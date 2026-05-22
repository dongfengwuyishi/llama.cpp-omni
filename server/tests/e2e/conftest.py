"""Shared fixtures for e2e tests.

Hosts the ``real_stack`` fixture that spins up an actual ``worker.py`` +
``batch_server.py`` against real GGUF weights on one GPU. Bumping the scope
to ``session`` lets multiple e2e test modules (``test_smoke``,
``test_logits_pd``, …) share one model-load — a single MiniCPM-o-4_5 boot
is ~30-60s, so the saving is real.

Required env vars (matches ``tests/e2e/README.md``):

    LLAMA_CPP_OMNI_ROOT  -- repo root containing build/bin/llama-server
    MODEL_DIR            -- GGUF directory
    LLM_MODEL            -- optional, e.g. "MiniCPM-o-4_5-Q8_0.gguf"
    TEST_AUDIO_WAV       -- absolute path to a small (≤10s) 16kHz mono wav
                            (only used by tests that exercise audio inputs)
    CUDA_VISIBLE_DEVICES -- e.g. "0"
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

SERVER_ROOT = Path(__file__).resolve().parents[2]


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _require_env(*keys: str) -> dict:
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        pytest.skip(f"E2E suite needs env vars: {missing}")
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


@pytest.fixture(scope="session")
def real_stack(tmp_path_factory):
    """Spin up a real worker + batch_server backed by the local GGUF.

    Session-scoped so the (slow) model load is shared across all e2e test
    modules in this directory.
    """
    env = _require_env("LLAMA_CPP_OMNI_ROOT", "MODEL_DIR")
    cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    worker_port = _pick_free_port()
    gateway_port = _pick_free_port()

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
        "MINICPMO_CONFIG_PATH": str(cfg_path),
    }

    worker = subprocess.Popen(
        [
            sys.executable, "worker.py",
            "--port", str(worker_port),
            "--gpu-id", "0",
            "--worker-index", "0",
        ],
        cwd=str(SERVER_ROOT),
        env=proc_env,
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
        env=proc_env,
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
