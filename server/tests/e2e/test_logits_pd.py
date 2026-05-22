"""End-to-end test: LLM logits capture covers BOTH prefill (P) and decode (D).

This test guards against the regression that motivated commit
``2a0a203 fix(no-tts): chat path emits index=0 system-prompt init …``:
before that fix the chat path silently dropped the first user message
(cnt=0 collided with C++ ``stream_prefill`` system-prompt init), so the
LLM decode loop ran zero sampling steps. The visible symptom on
``/v1/chat`` was an empty ``text`` field under ``--no-tts`` and a logits
``.safetensors`` whose ``n_tokens == n_prefill_tokens`` (only the system /
assistant prompt prefill positions were captured — the famous "~5%"
behaviour).

These tests exercise the running server stack provided by the
``real_stack`` fixture in ``test_smoke.py``:

- ``test_chat_logits_PD`` — single-turn chat, must produce non-empty text
  AND a logits payload with ``n_decode = n_tokens - n_prefill_tokens >= 2``.
- ``test_duplex_logits_PD`` — duplex_offline over a real audio file, must
  produce a multi-chunk logits payload with non-empty prefill capture; we
  do not hard-require decode tokens here (the model can legitimately
  decide to LISTEN every chunk for a short utterance), but we still
  surface a clear log line.

Both tests reuse the ``logits_format=file`` codepath because that is what
RL training pipelines actually consume.

Skip behaviour mirrors ``test_smoke``: no env -> skip cleanly, no GPU ->
skip cleanly via the underlying fixture.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# --------------------------------------------------------------------------
# Tiny safetensors reader (mirrors smoke_logits.py — we don't want to add a
# numpy/safetensors dep just for tests).
# --------------------------------------------------------------------------


def _read_safetensors(path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        body_offset = 8 + header_size
        tensors: Dict[str, np.ndarray] = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            dtype = spec["dtype"]
            shape = spec["shape"]
            start, end = spec["data_offsets"]
            f.seek(body_offset + start)
            raw = f.read(end - start)
            if dtype == "I32":
                tensors[name] = np.frombuffer(raw, dtype=np.int32).reshape(shape)
            elif dtype == "BF16":
                tensors[name] = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            elif dtype == "F32":
                tensors[name] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                raise NotImplementedError(f"unsupported dtype: {dtype}")
    return header.get("__metadata__", {}), tensors


def _split_pd(payload: Dict[str, Any], md: Dict[str, Any]) -> Tuple[int, int, int]:
    n_pref = int(md.get("n_prefill_tokens") or payload.get("n_prefill_tokens") or 0)
    n_total = int(md.get("n_tokens") or payload.get("n_tokens") or 0)
    return n_pref, n_total - n_pref, n_total


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_chat_logits_PD(real_stack, tmp_path):
    """``/v1/chat`` must capture both prefill and decode logits.

    Greedy decode (do_sample=False) should produce close to ``max_new_tokens``
    decode positions; we assert >= 2 to give the model breathing room while
    still failing hard on the historical "decode=0" regression.
    """
    import httpx

    out_dir = tmp_path / "logits_chat"
    out_dir.mkdir()
    body = {
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "generation": {"max_new_tokens": 24, "do_sample": False},
        "tts": {"enabled": False},
        "logits": {"enabled": True, "format": "file", "output_dir": str(out_dir)},
    }
    r = httpx.post(f"{real_stack['gateway_url']}/v1/chat", json=body, timeout=300)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p.get("success") is True
    text = (p.get("text") or "").strip()
    assert text, (
        "chat returned empty text — likely the no-tts cnt=0 dropped-content "
        "bug regressed (see commit 2a0a203 for context)"
    )

    lp = p.get("logits") or {}
    assert lp.get("success") is True, f"logits export failed: {lp}"
    file_path = lp.get("file")
    assert file_path and os.path.exists(file_path), f"safetensors missing: {file_path}"

    md, tensors = _read_safetensors(file_path)
    n_pref, n_decode, n_total = _split_pd(lp, md)

    # Visibility before assertions, so failures explain themselves.
    print(
        f"\n[chat logits] text={text!r}  n_total={n_total}  "
        f"n_prefill={n_pref}  n_decode={n_decode}  vocab={lp.get('vocab_size')}"
    )

    # Tensor shapes match header.
    assert tensors["token_ids"].shape == (n_total,)
    assert tensors["logits"].shape == (n_total, int(lp["vocab_size"]))

    # P phase MUST have captured tokens (system + assistant prompt + user prefill).
    assert n_pref > 0, f"prefill phase captured 0 tokens (n_prefill_tokens={n_pref})"

    # D phase MUST have captured tokens. This is the historical regression
    # surface — a passing chat with text but n_decode=0 would mean the
    # capture hooks aren't wired into the sampling loop anymore.
    assert n_decode >= 2, (
        f"decode phase captured only {n_decode} token(s); sampling loop barely ran. "
        f"This usually means stream_decode exited before/at the first iteration."
    )


def test_duplex_logits_PD(real_stack, tmp_path):
    """``/v1/duplex_offline`` must capture per-chunk logits with chunk_boundaries.

    Decode-side coverage isn't strictly required (the model is allowed to
    LISTEN every chunk for a short utterance) but prefill-side must be
    non-empty and ``chunk_boundaries`` must close out at ``n_tokens``.
    """
    audio = os.environ.get("TEST_AUDIO_WAV")
    if not audio or not Path(audio).exists():
        pytest.skip("set TEST_AUDIO_WAV to a small 16kHz mono wav")

    import httpx

    out_dir = tmp_path / "logits_duplex"
    out_dir.mkdir()
    body = {
        "system_prompt": "请用一句话回应。",
        "user_audio_path": audio,
        "config": {
            "force_listen_count": 3,
            "chunk_ms": 1000,
            "max_new_speak_tokens_per_chunk": 25,
        },
        "stop_on_end_of_turn": False,
        "return_per_chunk_audio": False,
        "return_merged_audio": False,
        "include_text_timeline": False,
        "logits": {"enabled": True, "format": "file", "output_dir": str(out_dir)},
        "request_id": "e2e_logits_pd_duplex",
    }
    r = httpx.post(
        f"{real_stack['gateway_url']}/v1/duplex_offline", json=body, timeout=900,
    )
    assert r.status_code == 200, r.text
    p = r.json()
    assert p.get("success") is True
    assert p.get("total_chunks", 0) >= 1

    lp = p.get("logits") or {}
    assert lp.get("success") is True, f"duplex logits export failed: {lp}"
    file_path = lp.get("file")
    assert file_path and os.path.exists(file_path), f"safetensors missing: {file_path}"

    md, tensors = _read_safetensors(file_path)
    n_pref, n_decode, n_total = _split_pd(lp, md)

    # Visibility.
    print(
        f"\n[duplex logits] total_chunks={p['total_chunks']}  speak={p.get('speak_chunks')}  "
        f"listen={p.get('listen_chunks')}  n_total={n_total}  n_prefill={n_pref}  "
        f"n_decode={n_decode}"
    )

    assert tensors["token_ids"].shape == (n_total,)
    assert tensors["logits"].shape == (n_total, int(lp["vocab_size"]))
    assert n_pref > 0, "duplex prefill phase captured 0 tokens"

    # chunk_boundaries closes consistent.
    cb = md.get("chunk_boundaries")
    if isinstance(cb, str):
        cb = json.loads(cb)
    assert cb, "chunk_boundaries metadata missing"
    assert cb[0] == 0
    assert cb[-1] == n_total, (
        f"chunk_boundaries[-1]={cb[-1]} doesn't match n_tokens={n_total}"
    )
    # one boundary per chunk + the closing offset
    assert len(cb) == p["total_chunks"] + 1, (
        f"chunk_boundaries has {len(cb)} entries, expected total_chunks+1="
        f"{p['total_chunks'] + 1}"
    )
