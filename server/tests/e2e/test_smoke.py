"""End-to-end smoke tests.

These spin up a real worker + batch_server as subprocesses and send live HTTP
requests. They require a GPU and weights; run with ``pytest --run-e2e``.

Environment:
    LLAMA_CPP_OMNI_ROOT   -- repo root (must contain build/bin/llama-server)
    MODEL_DIR             -- GGUF directory
    LLM_MODEL             -- optional, e.g. "MiniCPM-o-4_5-Q8_0.gguf"
    TEST_AUDIO_WAV        -- a small (≤10s) 16kHz mono wav
    CUDA_VISIBLE_DEVICES  -- e.g. "0"
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

SERVER_ROOT = Path(__file__).resolve().parents[2]


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _require_env(*keys: str) -> dict:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        pytest.skip(f"E2E smoke needs env vars: {missing}")
    return {k: os.environ[k] for k in keys}


def _wait_url(url: str, timeout: float, predicate=None) -> None:
    import urllib.request
    start = time.time()
    last_err = None
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                data = json.loads(r.read().decode())
                if predicate is None or predicate(data):
                    return
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    raise TimeoutError(f"{url} not ready within {timeout}s (last err: {last_err})")


@pytest.fixture(scope="module")
def real_stack(tmp_path_factory):
    """Start a real worker + batch_server with one GPU."""
    env = _require_env("LLAMA_CPP_OMNI_ROOT", "MODEL_DIR")
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    worker_port = _pick_free_port()
    gateway_port = _pick_free_port()

    # Write a temp config.json for this test run
    cfg_dir = tmp_path_factory.mktemp("cfg")
    cfg_path = cfg_dir / "config.json"
    cfg_payload = {
        "backend": "cpp",
        "model": {"model_path": "unused-for-cpp-backend"},
        "cpp_backend": {
            "llamacpp_root": env["LLAMA_CPP_OMNI_ROOT"],
            "model_dir": env["MODEL_DIR"],
            "llm_model": os.environ.get("LLM_MODEL", ""),
            "ctx_size": 32768,
            "n_gpu_layers": 99,
        },
        "service": {
            "gateway_port": gateway_port,
            "worker_base_port": worker_port,
            "num_workers": 1,
            "max_queue_size": 10,
            "request_timeout": 900.0,
        },
        "recording": {"enabled": False},
    }
    cfg_path.write_text(json.dumps(cfg_payload, indent=2))

    proc_env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": cuda,
        "PYTHONPATH": str(SERVER_ROOT),
    }

    worker = subprocess.Popen(
        [
            sys.executable, "worker.py",
            "--port", str(worker_port),
            "--gpu-id", "0",
            "--worker-index", "0",
        ],
        cwd=str(SERVER_ROOT),
        env={**proc_env, "MINICPMO_CONFIG_PATH": str(cfg_path)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    server = subprocess.Popen(
        [
            sys.executable, "batch_server.py",
            "--port", str(gateway_port),
            "--workers", f"localhost:{worker_port}",
        ],
        cwd=str(SERVER_ROOT),
        env={**proc_env, "MINICPMO_CONFIG_PATH": str(cfg_path)},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        _wait_url(
            f"http://localhost:{worker_port}/health",
            timeout=180.0,
            predicate=lambda d: d.get("model_loaded") is True,
        )
        _wait_url(
            f"http://localhost:{gateway_port}/health",
            timeout=10.0,
            predicate=lambda d: d.get("workers_total", 0) >= 1,
        )
        yield {
            "gateway_url": f"http://localhost:{gateway_port}",
            "worker_port": worker_port,
        }
    finally:
        for p in (server, worker):
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                p.kill()


def test_chat_smoke(real_stack):
    import httpx
    body = {
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "generation": {"max_new_tokens": 32, "do_sample": False},
        "tts": {"enabled": False},
    }
    r = httpx.post(f"{real_stack['gateway_url']}/v1/chat", json=body, timeout=300)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p.get("success") is True
    assert p.get("text"), "chat reply text should be non-empty"


def test_duplex_offline_smoke(real_stack):
    audio = os.environ.get("TEST_AUDIO_WAV")
    if not audio or not Path(audio).exists():
        pytest.skip("set TEST_AUDIO_WAV to a small 16kHz mono wav")

    import httpx
    body = {
        "system_prompt": "请用一句话回应用户。",
        "user_audio_path": audio,
        "config": {"force_listen_count": 3, "chunk_ms": 1000, "max_new_speak_tokens_per_chunk": 30},
        "stop_on_end_of_turn": True,
        "return_per_chunk_audio": False,
        "return_merged_audio": True,
        "request_id": "e2e_001",
    }
    r = httpx.post(
        f"{real_stack['gateway_url']}/v1/duplex_offline",
        json=body,
        timeout=600,
    )
    assert r.status_code == 200, r.text
    p = r.json()
    assert p.get("success") is True
    assert p["total_chunks"] >= 1
    # In the smoke case we don't fail if the model decides to keep listening,
    # but we should at least have processed something.
    assert p["request_id"] == "e2e_001"
