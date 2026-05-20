"""Unit tests for the LogitsExportSpec / LogitsPayload schemas and the
duplex_offline worker-side safetensors writer."""

from __future__ import annotations

import base64
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``server/`` importable so we can pull in ``worker.py`` helpers without
# triggering the FastAPI / torch import side-effects of running the app.
SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from core.schemas import (
    ChatRequest,
    ChatResponse,
    DuplexBatchRequest,
    DuplexBatchResponse,
    LogitsExportSpec,
    LogitsPayload,
    PLACEHOLDER_AUDIO,
    PLACEHOLDER_IMAGE,
    PLACEHOLDER_MODALITY,
)

pytestmark = pytest.mark.unit


# =========================================================================
# LogitsExportSpec
# =========================================================================


class TestLogitsExportSpec:
    def test_defaults(self):
        s = LogitsExportSpec()
        assert s.enabled is False
        assert s.format == "file"
        assert s.output_dir is None
        assert s.include_prefill is True

    def test_inline_explicit(self):
        s = LogitsExportSpec(enabled=True, format="inline")
        assert s.enabled is True
        assert s.format == "inline"

    def test_invalid_format(self):
        with pytest.raises(Exception):
            LogitsExportSpec(format="json")

    def test_round_trip_json(self):
        s = LogitsExportSpec(enabled=True, format="file", output_dir="/x")
        encoded = s.model_dump_json()
        back = LogitsExportSpec.model_validate_json(encoded)
        assert back == s


# =========================================================================
# LogitsPayload
# =========================================================================


class TestLogitsPayload:
    def test_minimal(self):
        p = LogitsPayload(n_tokens=10, n_prefill_tokens=3, vocab_size=128)
        assert p.success is True
        assert p.dtype == "bf16"
        assert p.token_ids_b64 is None
        assert p.logits_b64 is None
        assert p.file is None

    def test_inline_fields(self):
        p = LogitsPayload(
            n_tokens=2,
            n_prefill_tokens=1,
            vocab_size=4,
            token_ids_b64=base64.b64encode(b"\x00\x00\x00\x00\x01\x00\x00\x00").decode(),
            logits_b64=base64.b64encode(b"\x00" * 16).decode(),
        )
        assert p.token_ids_b64 is not None
        assert p.logits_b64 is not None
        # round-trip
        back = LogitsPayload.model_validate_json(p.model_dump_json())
        assert back.token_ids_b64 == p.token_ids_b64

    def test_file_fields(self):
        p = LogitsPayload(
            n_tokens=10,
            n_prefill_tokens=4,
            vocab_size=128,
            file="/tmp/x.safetensors",
            extra_metadata={"chunk_boundaries": [0, 3, 10]},
        )
        assert p.file == "/tmp/x.safetensors"
        assert p.extra_metadata["chunk_boundaries"] == [0, 3, 10]


# =========================================================================
# Placeholder constants
# =========================================================================


class TestPlaceholders:
    def test_distinct_negative(self):
        assert PLACEHOLDER_MODALITY == -1
        assert PLACEHOLDER_AUDIO == -2
        assert PLACEHOLDER_IMAGE == -3
        assert len({PLACEHOLDER_MODALITY, PLACEHOLDER_AUDIO, PLACEHOLDER_IMAGE}) == 3
        # Must be negative so they cannot collide with real Qwen3 vocab ids.
        for v in (PLACEHOLDER_MODALITY, PLACEHOLDER_AUDIO, PLACEHOLDER_IMAGE):
            assert v < 0


# =========================================================================
# Plumbing into ChatRequest / DuplexBatchRequest
# =========================================================================


class TestRequestPlumbing:
    def test_chat_request_default_disabled(self):
        r = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        assert r.logits.enabled is False

    def test_chat_request_enable(self):
        r = ChatRequest(
            messages=[{"role": "user", "content": "hi"}],
            logits={"enabled": True, "format": "inline"},
        )
        assert r.logits.enabled is True
        assert r.logits.format == "inline"

    def test_chat_response_logits_optional(self):
        r = ChatResponse(text="hi")
        assert r.logits is None
        r2 = ChatResponse(
            text="hi",
            logits=LogitsPayload(n_tokens=1, n_prefill_tokens=0, vocab_size=4),
        )
        assert r2.logits is not None
        assert r2.logits.n_tokens == 1

    def test_duplex_request_default_disabled(self):
        r = DuplexBatchRequest()
        assert r.logits.enabled is False

    def test_duplex_request_file_with_dir(self):
        r = DuplexBatchRequest(
            user_audio_path="/tmp/x.wav",
            logits={"enabled": True, "format": "file", "output_dir": "/tmp/out"},
        )
        assert r.logits.enabled is True
        assert r.logits.output_dir == "/tmp/out"

    def test_duplex_response_logits_optional(self):
        r = DuplexBatchResponse(success=True)
        assert r.logits is None


# =========================================================================
# worker._write_safetensors_logits — round-trip
# =========================================================================


class TestSafetensorsWriter:
    """The worker has a small Python safetensors writer for consolidating
    duplex per-chunk logits into one file. Round-trip it here so we don't
    depend on the C++ writer to verify format correctness.
    """

    def _import_writer(self):
        # worker.py is heavy (FastAPI app, ws endpoints). Only pull the two
        # helpers we need via runpy-style sourcing of a snippet would be
        # awkward; importing worker triggers the FastAPI app construction
        # which is fine for unit tests because lifespan only fires on uvicorn.
        # We just rely on ``import worker`` here.
        import importlib
        if "worker" in sys.modules:
            return sys.modules["worker"]
        return importlib.import_module("worker")

    def test_write_and_read_back(self, tmp_path):
        worker = self._import_writer()
        n_tokens = 8
        vocab = 32

        token_ids = np.arange(n_tokens, dtype=np.int32) - 1   # mix in -1 placeholder
        rng = np.random.default_rng(seed=0)
        logits_fp32 = rng.standard_normal((n_tokens, vocab), dtype=np.float32)
        # fp32 -> bf16 (truncation) to mimic what the C++ side writes
        u32 = logits_fp32.view(np.uint32)
        bf16 = (u32 >> 16).astype(np.uint16)
        body = bf16.tobytes()

        out = tmp_path / "out.safetensors"
        worker._write_safetensors_logits(
            path=str(out),
            token_ids=token_ids,
            logits_body_bf16=body,
            vocab_size=vocab,
            metadata={
                "format": "minicpm-o-omni-logits/v1",
                "n_prefill_tokens": "3",
                "vocab_size": str(vocab),
                "n_tokens": str(n_tokens),
                "chunk_boundaries": json.dumps([0, 3, 8]),
            },
        )

        # Read back
        with open(out, "rb") as f:
            hsz = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hsz))
            body_off = 8 + hsz
            tspec = header["token_ids"]
            lspec = header["logits"]
            f.seek(body_off + tspec["data_offsets"][0])
            tok_back = np.frombuffer(
                f.read(tspec["data_offsets"][1] - tspec["data_offsets"][0]),
                dtype=np.int32,
            )
            f.seek(body_off + lspec["data_offsets"][0])
            raw = f.read(lspec["data_offsets"][1] - lspec["data_offsets"][0])
            logits_back = np.frombuffer(raw, dtype=np.uint16).reshape(lspec["shape"])

        np.testing.assert_array_equal(tok_back, token_ids)
        np.testing.assert_array_equal(logits_back, bf16)
        assert header["token_ids"]["dtype"] == "I32"
        assert header["logits"]["dtype"] == "BF16"
        assert header["logits"]["shape"] == [n_tokens, vocab]
        md = header["__metadata__"]
        assert md["format"] == "minicpm-o-omni-logits/v1"
        assert md["n_prefill_tokens"] == "3"
        assert json.loads(md["chunk_boundaries"]) == [0, 3, 8]

    def test_header_padded_to_8(self, tmp_path):
        worker = self._import_writer()
        out = tmp_path / "pad.safetensors"
        worker._write_safetensors_logits(
            path=str(out),
            token_ids=np.zeros(1, dtype=np.int32),
            logits_body_bf16=b"\x00\x00",
            vocab_size=1,
            metadata={"n_prefill_tokens": "0"},
        )
        with open(out, "rb") as f:
            hsz = struct.unpack("<Q", f.read(8))[0]
        # The total prefix (8 + header_size) must be aligned to 8 bytes so
        # the body begins on an aligned offset.
        assert (8 + hsz) % 8 == 0
