"""Unit tests for ``core.processors.simplex_backend.SimplexCppBackend``.

These tests mock out ``_StreamHttpClient`` and ``_OutputDirManager`` so
we can verify the **call sequence and per-call arguments** without a
live llama-server. The simplex contract is small but every step
matters - in particular:

  * ``begin_turn`` must call ``update_session_config`` once and then
    ``prefill(cnt=0)`` exactly once before any user content
  * ``push_*`` must use ``cnt=1, 2, 3, ...`` (cnt=0 is system-only)
  * ``decode_*`` must increment the round index after each call
  * ``end_turn(full_restart=True)`` must hit ``_CppServerProc.full_restart``
"""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def make_backend(tmp_path):
    """Factory: returns ``(backend, fake_proc, fake_http, fake_mgr)``.
    Each unit test gets a fresh quartet."""
    from core.processors import simplex_backend as sb

    def _factory(*, ref_audio="/nonexistent/ref.wav", use_tts=True):
        fake_proc = MagicMock()
        fake_proc.output_dir = str(tmp_path / "out")
        fake_http = MagicMock()
        fake_http.kv_cache_length = 0

        # Replace _OutputDirManager with a mock to avoid touching the FS
        backend = sb.SimplexCppBackend(
            proc=fake_proc,
            http=fake_http,
            ref_audio_path=ref_audio,
            worker_idx=2,
            use_tts=use_tts,
            output_dir=str(tmp_path / "out"),
            temp_dir=str(tmp_path / "tmp"),
        )
        os.makedirs(str(tmp_path / "tmp"), exist_ok=True)
        backend._dir_mgr = MagicMock()
        return backend, fake_proc, fake_http
    return _factory


def test_begin_turn_calls_update_then_cnt0_prefill(make_backend):
    backend, _, http = make_backend(ref_audio="/nonexistent/ref.wav")
    backend.begin_turn(
        system_content="你是面壁小钢炮",
        sampling={"force_listen_count": 3},
        return_logits=True,
    )
    # update_session_config -> prefill(cnt=0)
    assert http.update_session_config.call_count == 1
    cfg_kwargs = http.update_session_config.call_args.kwargs
    assert cfg_kwargs["duplex_mode"] is False
    assert cfg_kwargs["sampling"] == {"force_listen_count": 3}
    assert cfg_kwargs["return_logits"] is True
    assert "<|im_start|>system" in cfg_kwargs["voice_clone_prompt"]
    assert "面壁小钢炮" in cfg_kwargs["voice_clone_prompt"]

    assert http.prefill.call_count == 1
    pkw = http.prefill.call_args.kwargs
    assert pkw["cnt"] == 0
    # ref_audio is "/nonexistent/ref.wav" which doesn't exist, so backend
    # passes audio_path="" (relies on Phase 0 server.cpp patch). When the
    # path exists in production the backend passes it through.
    assert pkw["audio_path"] == ""

    # Internal counters
    assert backend._cnt == 1
    assert backend._round_idx == 0


def test_begin_turn_passes_existing_ref_audio(make_backend, tmp_path):
    """When ref_audio_path actually exists on disk, backend must pass it
    through to the system-init prefill."""
    ref_wav = tmp_path / "ref.wav"
    ref_wav.write_bytes(b"\x00" * 1024)
    backend, _, http = make_backend(ref_audio=str(ref_wav))
    backend.begin_turn(system_content="x")
    assert http.prefill.call_args.kwargs["audio_path"] == str(ref_wav)


def test_push_audio_uses_cnt1_then_increments(make_backend):
    backend, _, http = make_backend()
    backend.begin_turn()
    http.reset_mock()

    audio = np.zeros(2048, dtype=np.float32)
    backend.push_audio(audio)
    backend.push_audio(audio)

    cnts = [c.kwargs["cnt"] for c in http.prefill.call_args_list]
    assert cnts == [1, 2]
    # Each call carries an audio_path pointing into the temp dir
    for c in http.prefill.call_args_list:
        assert c.kwargs["audio_path"].endswith(".wav")


def test_push_text_skips_empty_string(make_backend):
    backend, _, http = make_backend()
    backend.begin_turn()
    http.reset_mock()

    backend.push_text("")  # no-op
    backend.push_text("hello")
    backend.push_text("world")
    assert http.prefill.call_count == 2
    assert http.prefill.call_args_list[0].kwargs["text"] == "hello"
    assert http.prefill.call_args_list[1].kwargs["text"] == "world"
    assert http.prefill.call_args_list[1].kwargs["cnt"] == 2


def test_decode_oneshot_uses_round_idx_then_increments(make_backend):
    backend, _, http = make_backend(use_tts=False)
    backend.begin_turn()

    # Mock the decode response
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.text = (
        'data: {"content": "hello"}\n\n'
        'data: [DONE]\n\n'
    )
    http.decode.return_value = fake_resp

    out = backend.decode_oneshot(want_audio=False)
    assert out["text"] == "hello"
    assert out["audio_data"] is None
    assert backend._round_idx == 1

    # Round 2: ensure round_idx increments
    fake_resp2 = MagicMock()
    fake_resp2.status_code = 200
    fake_resp2.text = 'data: {"content": "world"}\n\ndata: [DONE]\n\n'
    http.decode.return_value = fake_resp2
    backend.push_text("hi")
    out2 = backend.decode_oneshot(want_audio=False)
    assert out2["text"] == "world"
    assert backend._round_idx == 2

    # Verify decode received round_idx=0 then round_idx=1
    rounds = [c.kwargs["round_idx"] for c in http.decode.call_args_list]
    assert rounds == [0, 1]


def test_decode_streaming_yields_text_audio_done(make_backend):
    backend, _, http = make_backend(use_tts=True)
    backend.begin_turn()

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.text = 'data: {"content": "hi"}\n\ndata: [DONE]\n\n'
    http.decode.return_value = fake_resp
    backend._dir_mgr.iter_chunks.return_value = iter(["AAAA", "BBBB"])

    events = list(backend.decode_streaming(generate_audio=True))
    types = [e["type"] for e in events]
    assert types == ["text", "audio", "audio", "done"]
    assert events[0]["delta"] == "hi"
    assert events[1]["data"] == "AAAA"
    assert events[3]["text"] == "hi"


def test_decode_streaming_skips_audio_when_use_tts_false(make_backend):
    backend, _, http = make_backend(use_tts=False)
    backend.begin_turn()
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.text = 'data: {"content": "x"}\n\ndata: [DONE]\n\n'
    http.decode.return_value = fake_resp

    events = list(backend.decode_streaming(generate_audio=True))
    types = [e["type"] for e in events]
    assert types == ["text", "done"]
    backend._dir_mgr.iter_chunks.assert_not_called()


def test_end_turn_full_restart_calls_proc(make_backend):
    backend, proc, _ = make_backend()
    backend.end_turn(full_restart=True)
    assert proc.full_restart.called


def test_end_turn_default_does_not_restart(make_backend):
    backend, proc, _ = make_backend()
    backend.end_turn()
    assert not proc.full_restart.called


def test_break_now_routes_to_http(make_backend):
    backend, _, http = make_backend()
    backend.break_now(reason="user_cancel")
    http.break_.assert_called_once_with(reason="user_cancel")


def test_logits_metadata_flows_through_decode(make_backend):
    """When the caller asks for file-format logits, decode_oneshot must
    forward logit_format / logit_output_dir / logit_filename / extra
    metadata into the underlying decode call."""
    backend, _, http = make_backend(use_tts=False)
    backend.begin_turn(return_logits=True)
    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.text = 'data: [DONE]\n\n'
    http.decode.return_value = fake_resp
    backend.decode_oneshot(
        logit_format="file",
        logit_output_dir="/tmp/logits/",
        logit_filename="chat-w0-pid-seq.safetensors",
        logit_extra_metadata={"request_id": "rid-1"},
    )
    kw = http.decode.call_args.kwargs
    assert kw["logit_format"] == "file"
    assert kw["logit_output_dir"] == "/tmp/logits/"
    assert kw["logit_filename"] == "chat-w0-pid-seq.safetensors"
    assert kw["logit_extra_metadata"] == {"request_id": "rid-1"}
