# MiniCPM-o Batch Inference Server

非流式批量推理服务，专为离线评测、刷数据等场景设计。
基于 `llama.cpp-omni` C++ 引擎（默认）或 PyTorch（可选），
对外提供 OpenAI 风格的 HTTP 接口：

| 接口 | 类型 | 输入 | 输出 |
|---|---|---|---|
| `POST /v1/chat`          | 单工（turn-based） | text + image + audio + video frames | text + 24kHz WAV |
| `POST /v1/duplex_offline`| 双工非流式         | 完整音频段 + 同步图像帧 + system_prompt | 每个 chunk 的 listen/speak/text/audio 时间线 |
| `GET  /v1/queue`         | -                  | -                                  | FIFO 队列状态 + ETA       |
| `GET  /v1/workers`       | -                  | -                                  | Worker 列表 + busy/idle   |
| `GET  /v1/health`        | -                  | -                                  | 集群健康 + 模型加载状态   |

与 `MiniCPM-o-Demo` 的关系：本目录从 Demo 的 `Comni` 分支（C++ 后端版本）
拆分而来，**完全没有前端、WebSocket、会话录制回放等在线交互特性**。
推理底层（`core/`、`worker.py` 内的推理原语）和 Demo 一致，只是入口层
（原 `gateway.py`）被替换成精简的 `batch_server.py`，去掉了 1500+ 行
前端 WS 代理代码。

---

## 与 Demo 的差异速查

| 文件 | 状态 |
|---|---|
| `core/`                  | 与 Demo 完全一致 |
| `worker.py`              | 与 Demo 一致，**新增 `POST /duplex_offline` 端点** |
| `gateway_modules/`       | 与 Demo 一致（worker_pool / models 都直接复用） |
| `batch_server.py`        | **新文件**，替代 Demo 的 `gateway.py`（约 300 行 vs Demo 1700+ 行） |
| `core/schemas/duplex_batch.py` | **新文件**，非流式双工 schema |
| `scripts/eval_runner.py` | **新文件**，批量评测客户端示例 |
| 前端 / VAD / 证书 / Docker | 未搬运 |

---

## TL;DR — 五条命令跑通

```bash
# 1. 在仓库根编译 C++ 引擎（依赖本分支已有的 omni stream API）
cd /path/to/llama.cpp-omni
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server -j

# 2. 安装 Python 依赖
cd server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. 配置（必须指定 cpp_backend.llamacpp_root + model_dir）
cp config.example.json config.json
${EDITOR:-vi} config.json

# 4. 启动（默认 cpp 后端，自动按 CUDA_VISIBLE_DEVICES 拉起 N 个 worker）
CUDA_VISIBLE_DEVICES=0,1 bash start_all.sh

# 5. 发请求（评测客户端示例）
python scripts/eval_runner.py \
    --endpoint http://localhost:8080 \
    --task duplex \
    --input-list eval_set.jsonl
```

---

## 接口契约

### `POST /v1/chat`（单工）

```jsonc
// 请求
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text",  "text": "描述视频里发生了什么"},
        {"type": "image", "data": "<base64 jpg>"},
        {"type": "audio", "data": "<base64 16kHz mono float32>"}
      ]
    }
  ],
  "tts":         { "enabled": true },
  "generation":  { "max_new_tokens": 256, "temperature": 0.7 }
}
// 响应
{
  "text":             "...",
  "audio_data":       "<base64 24kHz mono float32>",
  "audio_sample_rate": 24000,
  "duration_ms":      4180.7,
  "success":          true
}
```

### `POST /v1/duplex_offline`（双工非流式）

把完整的一段音视频丢给 server，server 内部按 `chunk_ms`（默认 1000ms）
切片 → 顺序 prefill+generate → 收集每个 chunk 的 listen/speak/text/audio
→ 一次性聚合返回。**不流式输出，调用方阻塞等结果**。

```jsonc
// 请求（详见 core/schemas/duplex_batch.py）
{
  "system_prompt":   "你是一个友好的语音助手，简短回复。",
  "user_audio_path": "/data/eval/sample_001.wav",   // 服务端可读路径
  "image_paths":     null,                          // 视频场景：每 chunk 一张
  "ref_audio_path":  null,                          // 不填走默认（按 system_prompt 语种）
  "config": {
    "force_listen_count":           3,
    "max_new_speak_tokens_per_chunk": 20,
    "chunk_ms":                     1000,
    "temperature":                  0.7,
    "top_k":                        20,
    "top_p":                        0.8,
    "length_penalty":               1.1
  },
  "stop_on_end_of_turn":  false,         // 默认 false：跑完所有音频 chunk
  "return_per_chunk_audio": true         // 默认 true：返回每个 chunk 的音频
}

// 响应（详见 core/schemas/duplex_batch.py: DuplexOfflineOutput）
{
  "success":         true,
  "full_text":       "你好，我听到你说...",
  "total_chunks":    18,
  "audio_duration_s": 12.3,
  "total_duration_ms": 28490.6,
  "chunks": [
    { "chunk_idx": 0, "phase": "user", "is_listen": true,  "text": "", ... },
    { "chunk_idx": 5, "phase": "response", "is_listen": false, "text": "你好",
      "audio_data": "<base64 24kHz float32>", "end_of_turn": false, ... },
    ...
  ],
  "merged_audio_data": "<base64 24kHz float32 拼接的整段>",  // 可选
  "queue_wait_ms":   42,
  "ticket_id":       "q_a1b2c3d4e5f6"
}
```

---

## 配置要点

- **后端选择**：`backend = "cpp"`（默认，吃 GGUF）或 `"pytorch"`（吃 HF 目录）
- **多 GPU**：`CUDA_VISIBLE_DEVICES` 决定起几个 worker，每个 worker 占一张卡
- **队列**：`service.max_queue_size` 默认 1000，按 FIFO 调度
- **超时**：单条 `duplex_offline` 默认 600s（视频/音频可能很长），可在 config 里改

---

## 性能/容量直觉

| 设置 | 并发数 | 单条延迟 |
|---|---|---|
| 1×L40S，C++ 后端 Q8_0 | 1（单 worker，单条串行） | 30s 音频约 25~40s 推理 |
| 4×L40S，C++ 后端 Q8_0 | 4 | 同上，4 路并发 |

模型本身不支持 batch；并发完全靠 worker 数量。

---

## 文档与测试

- 完整 API 使用手册：[`docs/API.md`](docs/API.md)
- 测试目录：[`tests/`](tests/)
  - `tests/unit/`：36 个，纯 pydantic / 调度逻辑，无 GPU 无模型，< 1 秒
  - `tests/integration/`：20 个，mock worker + 真 batch_server，无 GPU 无模型，约 12 秒
  - `tests/e2e/`：2 个，需要真实 GPU + GGUF，默认跳过

跑测试：

```bash
cd server
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio

# 单元 + 集成（推荐 CI 跑这两层）
PYTHONPATH=. pytest tests/unit tests/integration

# E2E 冒烟（需 GPU）
LLAMA_CPP_OMNI_ROOT=/abs/path  MODEL_DIR=/abs/gguf  TEST_AUDIO_WAV=/abs/x.wav \
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. pytest tests/e2e --run-e2e -s
```
