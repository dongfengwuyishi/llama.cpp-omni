"""Schema unit tests — pure pydantic validation, no IO / no model."""

from __future__ import annotations

import json

import pytest

from core.schemas import (
    ChatRequest,
    ChatResponse,
    DuplexBatchRequest,
    DuplexBatchResponse,
    DuplexChunkResult,
    DuplexConfig,
    Message,
    Role,
)

pytestmark = pytest.mark.unit


# ============================================================
# DuplexBatchRequest
# ============================================================


class TestDuplexBatchRequest:
    def test_defaults(self):
        req = DuplexBatchRequest()
        assert req.system_prompt == "You are a helpful assistant."
        assert req.user_audio_path is None
        assert req.user_audio_base64 is None
        assert req.image_paths is None
        assert req.image_base64_list is None
        assert req.max_slice_nums == 1
        assert req.stop_on_end_of_turn is False
        assert req.return_per_chunk_audio is True
        assert req.return_merged_audio is True
        assert req.include_text_timeline is True

    def test_default_config_is_duplex_config(self):
        req = DuplexBatchRequest()
        assert isinstance(req.config, DuplexConfig)
        assert req.config.chunk_ms == 1000
        assert req.config.force_listen_count == 3
        assert req.config.sample_rate == 16000

    def test_accepts_path(self):
        req = DuplexBatchRequest(user_audio_path="/tmp/x.wav")
        assert req.user_audio_path == "/tmp/x.wav"

    def test_accepts_base64(self):
        req = DuplexBatchRequest(user_audio_base64="ABCD==")
        assert req.user_audio_base64 == "ABCD=="

    def test_accepts_both_paths_and_b64_for_images(self):
        req = DuplexBatchRequest(
            image_paths=["/a.png", "/b.png"],
            image_base64_list=["AA==", "BB=="],
        )
        # server will pick path > base64 at execution time; schema allows both.
        assert req.image_paths == ["/a.png", "/b.png"]
        assert req.image_base64_list == ["AA==", "BB=="]

    def test_config_override(self):
        req = DuplexBatchRequest(
            config={"chunk_ms": 500, "force_listen_count": 5, "temperature": 1.0}
        )
        assert req.config.chunk_ms == 500
        assert req.config.force_listen_count == 5
        assert req.config.temperature == 1.0
        # Defaults preserved for un-overridden fields:
        assert req.config.sample_rate == 16000

    def test_max_chunks_min(self):
        # Pydantic v2 should reject max_chunks=0 because of ge=1
        with pytest.raises(Exception):
            DuplexBatchRequest(max_chunks=0)

    def test_leading_silence_non_negative(self):
        with pytest.raises(Exception):
            DuplexBatchRequest(leading_silence_ms=-10)

    def test_round_trip_json(self):
        req = DuplexBatchRequest(
            system_prompt="hi",
            user_audio_path="/x.wav",
            config={"chunk_ms": 1000},
            request_id="case_001",
        )
        encoded = req.model_dump_json()
        # Survive JSON serialisation
        decoded = DuplexBatchRequest.model_validate_json(encoded)
        assert decoded.system_prompt == "hi"
        assert decoded.user_audio_path == "/x.wav"
        assert decoded.config.chunk_ms == 1000
        assert decoded.request_id == "case_001"


# ============================================================
# DuplexBatchResponse
# ============================================================


class TestDuplexBatchResponse:
    def test_success_minimal(self):
        rsp = DuplexBatchResponse(success=True)
        assert rsp.success
        assert rsp.error is None
        assert rsp.full_text == ""
        assert rsp.chunks == []
        assert rsp.total_chunks == 0
        assert rsp.stopped_reason == "audio_exhausted"

    def test_error_path(self):
        rsp = DuplexBatchResponse(success=False, error="boom")
        assert not rsp.success
        assert rsp.error == "boom"

    def test_chunk_round_trip(self):
        chunks = [
            DuplexChunkResult(
                chunk_idx=0,
                phase="user",
                is_listen=True,
            ),
            DuplexChunkResult(
                chunk_idx=1,
                phase="response",
                is_listen=False,
                text="hi",
                has_audio=True,
                elapsed_ms=12.3,
            ),
        ]
        rsp = DuplexBatchResponse(
            success=True, full_text="hi", chunks=chunks, speak_chunks=1, listen_chunks=1,
            total_chunks=2,
        )
        as_dict = rsp.model_dump()
        recreated = DuplexBatchResponse.model_validate(as_dict)
        assert len(recreated.chunks) == 2
        assert recreated.chunks[1].text == "hi"
        assert recreated.chunks[1].is_listen is False

    def test_audio_fields_default_none(self):
        rsp = DuplexBatchResponse(success=True)
        assert rsp.merged_audio_data is None
        assert rsp.merged_audio_sample_rate is None

    def test_serialises_to_plain_json(self):
        rsp = DuplexBatchResponse(success=True, full_text="ok")
        # Must be safe to feed back through fastapi's JSONResponse path
        payload = json.loads(rsp.model_dump_json())
        assert payload["success"] is True
        assert payload["full_text"] == "ok"


# ============================================================
# ChatRequest / ChatResponse — sanity (these came from upstream demo)
# ============================================================


class TestChatRequestSmoke:
    def test_text_only(self):
        req = ChatRequest(messages=[Message(role=Role.USER, content="hi")])
        assert req.messages[0].content == "hi"
        assert req.tts.enabled is False
        assert req.generation.max_new_tokens > 0

    def test_messages_required(self):
        with pytest.raises(Exception):
            ChatRequest(messages=[])  # min_length=1


class TestChatResponseSmoke:
    def test_minimal(self):
        rsp = ChatResponse(text="hi")
        assert rsp.text == "hi"
        assert rsp.success is True
        assert rsp.audio_data is None
