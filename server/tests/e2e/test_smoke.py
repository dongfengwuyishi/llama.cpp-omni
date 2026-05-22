"""End-to-end smoke tests.

These spin up a real worker + batch_server as subprocesses and send live HTTP
requests. They require a GPU and weights; run with ``pytest --run-e2e``.

Environment:
    LLAMA_CPP_OMNI_ROOT   -- repo root (must contain build/bin/llama-server)
    MODEL_DIR             -- GGUF directory
    LLM_MODEL             -- optional, e.g. "MiniCPM-o-4_5-Q8_0.gguf"
    TEST_AUDIO_WAV        -- a small (≤10s) 16kHz mono wav
    CUDA_VISIBLE_DEVICES  -- e.g. "0"

The actual ``real_stack`` fixture lives in ``tests/e2e/conftest.py`` (so
that ``test_logits_pd.py`` and any other e2e module can share one
model-load).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


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
