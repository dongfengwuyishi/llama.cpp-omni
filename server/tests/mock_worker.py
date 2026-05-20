"""Mock Worker — lightweight stand-in for the real worker.py.

Implements all HTTP endpoints that ``batch_server.py`` talks to:

    GET  /health              health & worker_status
    POST /chat                ChatRequest → ChatResponse (mocked)
    POST /duplex_offline      DuplexBatchRequest → DuplexBatchResponse (mocked)
    POST /clear_cache         no-op
    GET  /cache_info          static reply

Designed to be embedded inside pytest's event loop via uvicorn.Server, so each
test can spin up N mock workers on random TCP ports without spawning extra
Python processes.

Behaviour knobs (set per-instance via ``configure()``):

- ``chat_delay`` (s):                how long /chat blocks
- ``duplex_offline_delay`` (s):      how long /duplex_offline blocks
- ``duplex_offline_chunks``:         how many fake chunk results to return
- ``fail_health`` (bool):            return non-200 on /health (simulate dead worker)
- ``raise_on_chat`` (bool):          return 500 on /chat
- ``raise_on_duplex`` (bool):        return 500 on /duplex_offline
"""

from __future__ import annotations

import asyncio
import base64
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response


# ============================================================
# Per-app state (each mock worker has its own instance)
# ============================================================


@dataclass
class MockWorkerState:
    worker_status: str = "idle"
    gpu_id: int = 0
    total_requests: int = 0
    current_session_id: Optional[str] = None

    # delays / failure injection
    chat_delay: float = 0.05
    duplex_offline_delay: float = 0.10
    duplex_offline_chunks: int = 4
    fail_health: bool = False
    raise_on_chat: bool = False
    raise_on_duplex: bool = False

    # tracking
    chat_calls: int = 0
    duplex_calls: int = 0
    last_chat_payload: Optional[Dict[str, Any]] = field(default=None)
    last_duplex_payload: Optional[Dict[str, Any]] = field(default=None)


# ============================================================
# FastAPI factory
# ============================================================


def build_app(state: MockWorkerState) -> FastAPI:
    app = FastAPI(title="MockWorker")

    @app.get("/health")
    async def health() -> Response:
        if state.fail_health:
            return Response(status_code=500, content="forced failure")
        return _json(
            {
                "status": "healthy",
                "worker_status": state.worker_status,
                "gpu_id": state.gpu_id,
                "model_loaded": True,
                "current_session_id": state.current_session_id,
                "total_requests": state.total_requests,
                "avg_inference_time_ms": 50.0,
                "kv_cache_length": 0,
            }
        )

    @app.post("/chat")
    async def chat(request: Request) -> Dict[str, Any]:
        body = await request.json()
        state.last_chat_payload = body
        state.chat_calls += 1
        state.total_requests += 1

        if state.raise_on_chat:
            raise HTTPException(status_code=500, detail="forced chat failure")

        state.worker_status = "busy_chat"
        try:
            await asyncio.sleep(state.chat_delay)
        finally:
            state.worker_status = "idle"

        # Synthesize a small base64-encoded audio payload (1024 samples, 24kHz, float32)
        synth = np.zeros(1024, dtype=np.float32)
        audio_b64 = base64.b64encode(synth.tobytes()).decode("utf-8")

        return {
            "text": _summarize_messages(body.get("messages", [])),
            "audio_data": audio_b64 if body.get("tts", {}).get("enabled") else None,
            "audio_sample_rate": 24000 if body.get("tts", {}).get("enabled") else None,
            "tokens_generated": 12,
            "duration_ms": state.chat_delay * 1000.0,
            "success": True,
            "token_stats": {
                "input_tokens": 10,
                "generated_tokens": 12,
                "total_tokens": 22,
                "cached_tokens": 0,
            },
        }

    @app.post("/duplex_offline")
    async def duplex_offline(request: Request) -> Dict[str, Any]:
        body = await request.json()
        state.last_duplex_payload = body
        state.duplex_calls += 1
        state.total_requests += 1

        if state.raise_on_duplex:
            raise HTTPException(status_code=500, detail="forced duplex failure")

        state.worker_status = "duplex_active"
        t0 = time.perf_counter()
        try:
            await asyncio.sleep(state.duplex_offline_delay)
        finally:
            state.worker_status = "idle"

        n = state.duplex_offline_chunks
        chunks: List[Dict[str, Any]] = []
        text_pieces: List[str] = []
        speak = 0
        listen = 0
        synth = np.zeros(2400, dtype=np.float32)  # 0.1s of 24kHz audio
        synth_b64 = base64.b64encode(synth.tobytes()).decode("utf-8")
        return_per_chunk = body.get("return_per_chunk_audio", True)
        return_merged = body.get("return_merged_audio", True)
        include_tl = body.get("include_text_timeline", True)

        for i in range(n):
            is_listen = i < (body.get("config", {}).get("force_listen_count", 3))
            if is_listen:
                listen += 1
                text = ""
                has_audio = False
            else:
                speak += 1
                text = f"chunk_{i} "
                text_pieces.append(text)
                has_audio = True

            if include_tl:
                chunks.append(
                    {
                        "chunk_idx": i,
                        "phase": "user" if is_listen else "response",
                        "is_listen": is_listen,
                        "text": text,
                        "has_audio": has_audio,
                        "audio_data": (synth_b64 if (has_audio and return_per_chunk) else None),
                        "end_of_turn": False,
                        "elapsed_ms": state.duplex_offline_delay * 1000.0 / n,
                    }
                )

        merged_b64 = None
        if return_merged and speak > 0:
            merged = np.tile(synth, speak)
            merged_b64 = base64.b64encode(merged.tobytes()).decode("utf-8")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": True,
            "full_text": "".join(text_pieces),
            "chunks": chunks,
            "merged_audio_data": merged_b64,
            "merged_audio_sample_rate": 24000 if merged_b64 else None,
            "total_chunks": speak + listen,
            "speak_chunks": speak,
            "listen_chunks": listen,
            "audio_duration_s": speak * 0.1,
            "total_duration_ms": elapsed_ms,
            "stopped_reason": "audio_exhausted",
            "request_id": body.get("request_id"),
        }

    @app.post("/clear_cache")
    async def clear_cache():
        return {"success": True, "message": "Cache cleared (mock)"}

    @app.get("/cache_info")
    async def cache_info():
        return {"status": "no_cache", "note": "mock worker"}

    return app


# ============================================================
# Helper: in-thread uvicorn launcher
# ============================================================


def _json(payload: Dict[str, Any]) -> Response:
    import json
    return Response(content=json.dumps(payload), media_type="application/json")


def _summarize_messages(messages: list) -> str:
    """Construct a deterministic text reply from the last user message."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return f"[mock-reply] {content[:60]}"
            if isinstance(content, list):
                texts = [c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"]
                if texts:
                    return f"[mock-reply] {' '.join(texts)[:60]}"
                return "[mock-reply] (multimodal)"
    return "[mock-reply] (no user message)"


class MockWorkerProcess:
    """Run a mock worker on a TCP port inside a background thread.

    Uses an isolated uvicorn.Server (asyncio loop in a thread) so the running
    pytest event loop can issue real HTTP requests against it without
    juggling lifespans manually.
    """

    def __init__(self, host: str, port: int, state: Optional[MockWorkerState] = None):
        self.host = host
        self.port = port
        self.state = state or MockWorkerState()
        self.app = build_app(self.state)
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def base_url(self) -> str:
        return f"http://{self.address}"

    def start(self) -> None:
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        # uvicorn.Server.run() builds its own loop; running it in a thread is fine.
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 5.0) -> None:
        import urllib.request
        start = time.time()
        while time.time() - start < timeout:
            try:
                urllib.request.urlopen(f"{self.base_url}/health", timeout=0.3)
                return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError(f"Mock worker on {self.address} failed to start")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._server = None
        self._thread = None
