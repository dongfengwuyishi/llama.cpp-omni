#!/usr/bin/env bash
# Launch the non-streaming batch inference server.
#
# Usage:
#     bash start_all.sh                 # use all visible GPUs
#     CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh
#
# Notes:
#   - This script spawns one worker per GPU + a single batch_server on the
#     gateway port. Workers and the server log to ./tmp/*.log .
#   - llama-server (C++) is auto-spawned by each worker when backend="cpp".
#   - Stop everything with:
#         pkill -f "batch_server.py|worker.py|llama-server"

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
if [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
fi

# Read ports from config (best-effort; fall back to defaults)
GATEWAY_PORT=$($PYTHON_BIN -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().gateway_port)" 2>/dev/null || echo "8080")
WORKER_BASE_PORT=$($PYTHON_BIN -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from config import get_config; print(get_config().worker_base_port)" 2>/dev/null || echo "22440")

# Detect GPUs
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
    if [[ "$NUM_GPUS" -le 0 ]]; then
        echo "No GPU detected (nvidia-smi failed). Set CUDA_VISIBLE_DEVICES explicitly."
        exit 1
    fi
    GPU_LIST=$(seq 0 $((NUM_GPUS - 1)) | tr '\n' ',' | sed 's/,$//')
else
    GPU_LIST="$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | wc -l)
fi

echo "=================================================="
echo "  MiniCPM-o Batch Server"
echo "=================================================="
echo "  GPUs:     $GPU_LIST ($NUM_GPUS)"
echo "  Gateway:  http://localhost:$GATEWAY_PORT"
echo "  Workers:  localhost:$WORKER_BASE_PORT..localhost:$((WORKER_BASE_PORT + NUM_GPUS - 1))"
echo "=================================================="

mkdir -p tmp

# Start workers, one per GPU
WORKER_ADDRS=""
GPU_IDX=0
for GPU_ID in $(echo "$GPU_LIST" | tr ',' ' '); do
    WORKER_PORT=$((WORKER_BASE_PORT + GPU_IDX))
    echo "[worker $GPU_IDX] starting on GPU $GPU_ID, port $WORKER_PORT ..."

    nohup env CUDA_VISIBLE_DEVICES=$GPU_ID PYTHONPATH="$PROJECT_DIR" \
        $PYTHON_BIN worker.py \
            --port "$WORKER_PORT" \
            --gpu-id "$GPU_ID" \
            --worker-index "$GPU_IDX" \
        > "tmp/worker_${GPU_IDX}.log" 2>&1 &
    echo $! > "tmp/worker_${GPU_IDX}.pid"

    if [[ -z "$WORKER_ADDRS" ]]; then
        WORKER_ADDRS="localhost:$WORKER_PORT"
    else
        WORKER_ADDRS="$WORKER_ADDRS,localhost:$WORKER_PORT"
    fi
    GPU_IDX=$((GPU_IDX + 1))
done

# Wait for workers to become healthy
echo ""
echo "Waiting for workers to load models ..."
sleep 5
for i in $(seq 0 $((NUM_GPUS - 1))); do
    WORKER_PORT=$((WORKER_BASE_PORT + i))
    MAX_RETRIES=3000
    RETRY=0
    while [[ $RETRY -lt $MAX_RETRIES ]]; do
        if curl -sf "http://localhost:$WORKER_PORT/health" 2>/dev/null | \
                python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('model_loaded') else 1)" 2>/dev/null; then
            echo "[worker $i] ready ($WORKER_PORT)"
            break
        fi
        RETRY=$((RETRY + 1))
        sleep 2
    done
    if [[ $RETRY -eq $MAX_RETRIES ]]; then
        echo "[worker $i] FAILED to start (see tmp/worker_${i}.log)"
    fi
done

# Start gateway / batch_server
echo ""
echo "[batch_server] starting on port $GATEWAY_PORT ..."
nohup env PYTHONPATH="$PROJECT_DIR" \
    $PYTHON_BIN batch_server.py \
        --port "$GATEWAY_PORT" \
        --workers "$WORKER_ADDRS" \
    > "tmp/batch_server.log" 2>&1 &
echo $! > "tmp/batch_server.pid"

sleep 2
if curl -sf "http://localhost:$GATEWAY_PORT/health" >/dev/null 2>&1; then
    echo "[batch_server] ready"
else
    echo "[batch_server] may still be starting; see tmp/batch_server.log"
fi

echo ""
echo "=================================================="
echo "  Running."
echo "  Endpoints:"
echo "    POST  http://localhost:$GATEWAY_PORT/v1/chat"
echo "    POST  http://localhost:$GATEWAY_PORT/v1/duplex_offline"
echo "    GET   http://localhost:$GATEWAY_PORT/v1/queue"
echo "    GET   http://localhost:$GATEWAY_PORT/v1/workers"
echo "    GET   http://localhost:$GATEWAY_PORT/v1/health"
echo "  Logs: tmp/batch_server.log  tmp/worker_*.log"
echo "  Stop: kill \$(cat tmp/*.pid 2>/dev/null) 2>/dev/null"
echo "=================================================="
