"""MiniCPM-o Batch Inference Server（非流式）

非流式批量推理入口，对外提供 OpenAI 风格 HTTP 接口：

    POST /v1/chat            单工（turn-based）
    POST /v1/duplex_offline  双工非流式（一次性返回整段结果）
    GET  /v1/queue           FIFO 队列状态
    GET  /v1/workers         worker 列表
    GET  /v1/health          集群健康
    GET  /health             向后兼容

设计上是 ``MiniCPM-o-Demo/gateway.py`` 的精简替代品：

- 没有前端、没有 WebSocket、没有会话录制 / 回放
- 没有 Admin、ETA 动态调整、ref audio 上传 等管理面
- 没有 HTTPS / 自签证书（评测内网部署用纯 HTTP 即可）
- 复用 ``gateway_modules/worker_pool.py`` 的 FIFO 队列与 worker 调度

启动方式：

    python batch_server.py --port 8080 --workers localhost:22440,localhost:22441
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# 让 worker_pool 走与 Demo 一致的 import 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gateway_modules.models import (
    EtaConfig,
    GatewayWorkerStatus,
    QueueStatus,
    ServiceStatus,
    WorkersResponse,
)
from gateway_modules.worker_pool import WorkerConnection, WorkerPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("batch_server")


# ============ 全局状态 ============

worker_pool: Optional[WorkerPool] = None
SERVER_CONFIG: Dict[str, Any] = {}


# ============ FastAPI 生命周期 ============

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 停止时管理 worker 连接池"""
    global worker_pool

    worker_addresses = SERVER_CONFIG["worker_addresses"]
    max_queue_size = SERVER_CONFIG["max_queue_size"]
    request_timeout = SERVER_CONFIG["request_timeout"]
    eta_config = SERVER_CONFIG.get("eta_config") or EtaConfig()

    logger.info(
        "Starting batch_server: workers=%s, max_queue_size=%d, request_timeout=%.0fs",
        worker_addresses, max_queue_size, request_timeout,
    )

    worker_pool = WorkerPool(
        worker_addresses=worker_addresses,
        max_queue_size=max_queue_size,
        request_timeout=request_timeout,
        eta_config=eta_config,
        ema_alpha=SERVER_CONFIG.get("ema_alpha", 0.3),
        ema_min_samples=SERVER_CONFIG.get("ema_min_samples", 3),
    )
    await worker_pool.start()
    logger.info("WorkerPool started: %d workers", len(worker_pool.workers))

    try:
        yield
    finally:
        logger.info("Shutting down WorkerPool ...")
        await worker_pool.stop()
        worker_pool = None


app = FastAPI(
    title="MiniCPM-o Batch Inference Server",
    description="Non-streaming HTTP server for chat / duplex_offline inference",
    version="0.1.0",
    lifespan=lifespan,
)


# ============ 通用工具：FIFO 排队 + 代理到 worker ============

async def _wait_for_worker(
    request: Request,
    ticket,
    future: "asyncio.Future[Optional[WorkerConnection]]",
) -> Optional[WorkerConnection]:
    """阻塞等 worker 分配，期间检测客户端断开"""
    if future.done():
        return future.result()
    while not future.done():
        if await request.is_disconnected():
            worker_pool.cancel(ticket.ticket_id)
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=2.0)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            worker_pool.cancel(ticket.ticket_id)
            return None
    return future.result()


async def _proxy_to_worker(
    request: Request,
    request_type: str,
    worker_path: str,
    payload: Dict[str, Any],
    dispatch_status: GatewayWorkerStatus,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """通用代理：入队 → 等 worker → 发请求 → 释放 worker"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    queue_start = datetime.now()
    try:
        ticket, future = worker_pool.enqueue(request_type)
    except WorkerPool.QueueFullError:
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({worker_pool.max_queue_size} requests)",
        )

    worker = await _wait_for_worker(request, ticket, future)
    if worker is None:
        raise HTTPException(status_code=499, detail="Client disconnected while queued")

    queue_done = datetime.now()
    queue_wait_ms = (queue_done - queue_start).total_seconds() * 1000.0
    estimated_queue_s = ticket.estimated_wait_s

    # Worker 已在 enqueue/dispatch 时被标记为 busy，这里只补一下日志
    task_start = datetime.now()
    eff_timeout = timeout or worker_pool.request_timeout

    try:
        async with httpx.AsyncClient(timeout=eff_timeout) as client:
            resp = await client.post(
                f"{worker.url}{worker_path}", json=payload, timeout=eff_timeout
            )

        worker.total_requests += 1
        worker.last_heartbeat = datetime.now()

        if resp.status_code != 200:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)

        result = resp.json()
        result["queue_wait_ms"] = round(queue_wait_ms)
        result["estimated_queue_wait_s"] = round(estimated_queue_s, 1)
        result["ticket_id"] = ticket.ticket_id
        result["worker_id"] = worker.worker_id
        return result

    except HTTPException:
        raise
    except httpx.TimeoutException:
        logger.error("[%s] worker %s timeout (limit=%.1fs)",
                     ticket.ticket_id, worker.worker_id, eff_timeout)
        raise HTTPException(status_code=504, detail="Worker timeout")
    except Exception as e:
        logger.error("[%s] proxy failed: %s", ticket.ticket_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        duration = (datetime.now() - task_start).total_seconds()
        worker_pool.release_worker(
            worker, request_type=request_type, duration_s=duration
        )


# ============ 业务端点 ============

@app.get("/health")
async def health_compat():
    """向后兼容 /health（与 /v1/health 等价）"""
    return await health()


@app.get("/v1/health")
async def health():
    """集群健康总览"""
    if worker_pool is None:
        return {"status": "starting", "ready": False}

    total = len(worker_pool.workers)
    idle = worker_pool.idle_count
    busy = worker_pool.busy_count
    loading = worker_pool.loading_count
    error = worker_pool.error_count
    return {
        "status": "ok",
        "ready": idle + busy > 0,
        "workers_total": total,
        "workers_idle": idle,
        "workers_busy": busy,
        "workers_loading": loading,
        "workers_error": error,
        "queue_length": worker_pool.queue_length,
        "max_queue_size": worker_pool.max_queue_size,
    }


@app.get("/v1/workers", response_model=WorkersResponse)
async def list_workers():
    """worker 列表"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return WorkersResponse(
        total=len(worker_pool.workers),
        workers=worker_pool.get_all_workers(),
    )


@app.get("/v1/queue", response_model=QueueStatus)
async def get_queue():
    """FIFO 队列状态"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return worker_pool.get_queue_status()


@app.get("/v1/status", response_model=ServiceStatus)
async def get_status():
    """聚合的服务状态（便于 dashboard）"""
    if worker_pool is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    qs = worker_pool.get_queue_status()
    return ServiceStatus(
        gateway_healthy=True,
        total_workers=len(worker_pool.workers),
        idle_workers=worker_pool.idle_count,
        busy_workers=worker_pool.busy_count,
        duplex_workers=worker_pool.duplex_count,
        loading_workers=worker_pool.loading_count,
        error_workers=worker_pool.error_count,
        offline_workers=worker_pool.offline_count,
        queue_length=qs.queue_length,
        max_queue_size=worker_pool.max_queue_size,
        running_tasks=qs.running_tasks,
    )


@app.post("/v1/chat")
async def chat(request: Request):
    """单工推理（非流式）

    输入：``ChatRequest``（见 ``core/schemas/chat.py``）
    输出：``ChatResponse``
    """
    body = await request.json()
    return await _proxy_to_worker(
        request=request,
        request_type="chat",
        worker_path="/chat",
        payload=body,
        dispatch_status=GatewayWorkerStatus.BUSY_CHAT,
    )


@app.post("/v1/duplex_offline")
async def duplex_offline(request: Request):
    """双工非流式推理（一次性返回整段结果）

    输入：``DuplexBatchRequest``（见 ``core/schemas/duplex_batch.py``）
    输出：``DuplexBatchResponse``

    超时：默认走 ``SERVER_CONFIG["duplex_offline_timeout"]``（默认 600s）。
    """
    body = await request.json()
    timeout = SERVER_CONFIG.get("duplex_offline_timeout", 600.0)
    # FIFO 队列里用 "audio_duplex" 类型（worker_pool 已有此请求类型与 ETA）
    return await _proxy_to_worker(
        request=request,
        request_type="audio_duplex",
        worker_path="/duplex_offline",
        payload=body,
        dispatch_status=GatewayWorkerStatus.DUPLEX_ACTIVE,
        timeout=timeout,
    )


# ============ 启动入口 ============

def _resolve_worker_addresses(args, cfg) -> list:
    """解析 --workers 或 --num-workers"""
    if args.workers:
        return [a.strip() for a in args.workers.split(",") if a.strip()]
    n = args.num_workers or cfg.num_workers
    return cfg.worker_addresses(n)


def main():
    from config import get_config

    cfg = get_config()

    parser = argparse.ArgumentParser(description="MiniCPM-o Batch Inference Server")
    parser.add_argument(
        "--port", type=int, default=cfg.gateway_port,
        help=f"Server port (default: from config = {cfg.gateway_port})",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument(
        "--workers", type=str, default=None,
        help='Worker addresses, comma-separated, e.g. "localhost:22440,localhost:22441"',
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of workers (used when --workers not set; uses worker_base_port).",
    )
    parser.add_argument(
        "--duplex-offline-timeout", type=float, default=600.0,
        help="Per-request timeout for /v1/duplex_offline (seconds, default 600)",
    )
    args = parser.parse_args()

    worker_addresses = _resolve_worker_addresses(args, cfg)

    SERVER_CONFIG.update(
        {
            "worker_addresses": worker_addresses,
            "max_queue_size": cfg.max_queue_size,
            "request_timeout": cfg.request_timeout,
            "ema_alpha": cfg.eta_ema_alpha,
            "ema_min_samples": cfg.eta_ema_min_samples,
            "duplex_offline_timeout": args.duplex_offline_timeout,
            "eta_config": EtaConfig(
                eta_chat_s=cfg.eta_chat_s,
                eta_streaming_s=cfg.eta_streaming_s,
                eta_half_duplex_s=cfg.eta_half_duplex_s,
                eta_audio_duplex_s=cfg.eta_audio_duplex_s,
                eta_omni_duplex_s=cfg.eta_omni_duplex_s,
                eta_duplex_s=cfg.eta_duplex_s,
            ),
        }
    )

    logger.info(
        "batch_server starting: port=%d host=%s workers=%s",
        args.port, args.host, worker_addresses,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
