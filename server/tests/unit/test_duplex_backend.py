"""Unit tests for ``core.processors.duplex_backend.DuplexCppBackend``.

Mocks ``_StreamHttpClient`` and ``_OutputDirManager`` so we can verify
the duplex call sequence without a live llama-server. The duplex
contract is more involved than simplex; key invariants tested here:

  * ``session_begin`` calls ``update_session_config`` once then
    ``prefill(cnt=0, audio=ref)`` exactly once
  * ``ref_audio_override`` wins over the worker-level default
  * ``push_frame`` uses ``cnt = self._frame_idx``, increments by
    ``1 + len(vision_frames)``, and forwards ``force_listen`` /
    ``round_idx`` to ``decode``
  * Vision frames get their own ``cnt`` slots after the audio
  * ``return_logits=True`` triggers ``logit_format="inline"`` on each
    decode call (per-chunk capture)
  * ``session_end`` issues a ``break_`` and ``full_restart`` (default)
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def make_backend(tmp_path):
    """Factory: returns ``(backend, fake_proc, fake_http)`` per test."""
    from core.processors import duplex_backend as db

    def _factory(*, ref_audio="/nonexistent/ref.wav", use_tts=True):
        fake_proc = MagicMock()
        fake_proc.output_dir = str(tmp_path / "out")
        fake_http = MagicMock()
        fake_http.kv_cache_length = 0

        backend = db.DuplexCppBackend(
            proc=fake_proc,
            http=fake_http,
            ref_audio_path=ref_audio,
            worker_idx=3,
            use_tts=use_tts,
            output_dir=str(tmp_path / "out"),
            temp_dir=str(tmp_path / "tmp"),
        )
        import os
        os.makedirs(str(tmp_path / "tmp"), exist_ok=True)
        backend._dir_mgr = MagicMock()
        backend._dir_mgr.collect_nowait.return_value = (None, "")
        return backend, fake_proc, fake_http
    return _factory


def _empty_decode_resp():
    """Build a fake decode SSE body with one event + DONE."""
    resp = MagicMock()
    resp.status_code = 200
    resp.text = (
        'data: {"is_listen": false, "end_of_turn": false, "content": "嗨"}\n\n'
        'data: [DONE]\n\n'
    )
    return resp


def test_session_begin_default_ref_audio_does_not_exist(make_backend):
    """When neither override nor worker default exists, backend still
    calls prefill(cnt=0) but with an empty path - relying on Phase 0
    server.cpp patch + ``ctx_omni->ref_audio_path`` fallback set by
    omni_init."""
    backend, _, http = make_backend(ref_audio="/nonexistent/ref.wav")
    backend.session_begin(system_content="你是面壁小钢炮", return_logits=False)

    assert http.update_session_config.call_count == 1
    cfg = http.update_session_config.call_args.kwargs
    assert cfg["duplex_mode"] is True
    assert cfg["return_logits"] is False
    assert "面壁小钢炮" in cfg["voice_clone_prompt"]

    assert http.prefill.call_count == 1
    pkw = http.prefill.call_args.kwargs
    assert pkw["cnt"] == 0
    assert pkw["audio_path"] == ""

    assert backend._frame_idx == 1
    assert backend._return_logits is False


def test_session_begin_with_override_uses_override(make_backend, tmp_path):
    """``ref_audio_override`` wins over worker-level default."""
    override = tmp_path / "override.wav"
    override.write_bytes(b"\x00" * 1024)
    default = tmp_path / "default.wav"
    default.write_bytes(b"\x00" * 1024)

    backend, _, http = make_backend(ref_audio=str(default))
    backend.session_begin(ref_audio_override=str(override))
    assert http.prefill.call_args.kwargs["audio_path"] == str(override)


def test_session_begin_falls_back_to_worker_default(make_backend, tmp_path):
    """No override, but worker default exists - backend uses default."""
    default = tmp_path / "default.wav"
    default.write_bytes(b"\x00" * 1024)
    backend, _, http = make_backend(ref_audio=str(default))
    backend.session_begin()
    assert http.prefill.call_args.kwargs["audio_path"] == str(default)


def test_session_begin_passes_sampling_through(make_backend):
    backend, _, http = make_backend()
    backend.session_begin(
        sampling={
            "force_listen_count": 2,
            "max_new_speak_tokens_per_chunk": 24,
            "llm_sampling": {"temp": 0.0, "seed": 42},
        },
        return_logits=True,
    )
    cfg = http.update_session_config.call_args.kwargs
    assert cfg["sampling"]["force_listen_count"] == 2
    assert cfg["sampling"]["max_new_speak_tokens_per_chunk"] == 24
    assert cfg["sampling"]["llm_sampling"] == {"temp": 0.0, "seed": 42}


def test_push_frame_uses_frame_idx_and_increments(make_backend):
    backend, _, http = make_backend(use_tts=False)
    backend.session_begin()
    http.reset_mock()
    http.decode.return_value = _empty_decode_resp()

    audio = np.zeros(2048, dtype=np.float32)

    out1 = backend.push_frame(audio_chunk=audio)
    out2 = backend.push_frame(audio_chunk=audio)
    out3 = backend.push_frame(audio_chunk=audio)

    # prefill cnts
    cnts = [c.kwargs["cnt"] for c in http.prefill.call_args_list]
    assert cnts == [1, 2, 3]
    # decode round_idx
    rounds = [c.kwargs["round_idx"] for c in http.decode.call_args_list]
    assert rounds == [1, 2, 3]
    # current_time field reflects the cnt at call time
    assert out1["current_time"] == 1
    assert out2["current_time"] == 2
    assert out3["current_time"] == 3
    # text comes through from SSE event
    assert out1["text"] == "嗨"
    # is_listen / end_of_turn are read from the same event
    assert out1["is_listen"] is False
    assert out1["end_of_turn"] is False


def test_push_frame_force_listen_flows_to_decode(make_backend):
    """D7: ``force_listen=True`` must end up on the decode body."""
    backend, _, http = make_backend(use_tts=False)
    backend.session_begin()
    http.reset_mock()
    http.decode.return_value = _empty_decode_resp()

    backend.push_frame(audio_chunk=np.zeros(2048, dtype=np.float32),
                       force_listen=True)
    assert http.decode.call_args.kwargs["force_listen"] is True

    backend.push_frame(audio_chunk=np.zeros(2048, dtype=np.float32),
                       force_listen=False)
    # When False, the field is omitted (None) - the C++ default applies
    assert http.decode.call_args.kwargs["force_listen"] is None


def test_push_frame_vision_frames_get_separate_cnts(make_backend):
    """When vision frames are passed, each one consumes its own cnt
    slot after the audio frame, and the next push_frame skips past them."""
    backend, _, http = make_backend(use_tts=False)
    backend.session_begin()
    http.reset_mock()
    http.decode.return_value = _empty_decode_resp()

    fake_pil = MagicMock()
    fake_pil.mode = "RGB"
    fake_pil.save = MagicMock()

    audio = np.zeros(2048, dtype=np.float32)
    out = backend.push_frame(audio_chunk=audio, vision_frames=[fake_pil, fake_pil])
    # cnt sequence: audio at 1, vision at 2, vision at 3, decode at round_idx=1
    cnts = [c.kwargs["cnt"] for c in http.prefill.call_args_list]
    assert cnts == [1, 2, 3]
    # frame_idx must skip past 1 (audio) + 2 (vision) -> next push starts at 4
    assert backend._frame_idx == 4
    assert out["n_vision_frames"] == 2

    # Second push_frame
    out2 = backend.push_frame(audio_chunk=audio)
    assert out2["current_time"] == 4
    assert backend._frame_idx == 5


def test_push_frame_skips_audio_prefill_when_chunk_empty(make_backend):
    """Empty audio chunk must still advance frame_idx and run decode,
    but does NOT issue a prefill (a cnt>0 prefill with empty body is
    rejected by the unpatched server, and the ``empty user audio = no
    new content`` semantics are preserved)."""
    backend, _, http = make_backend(use_tts=False)
    backend.session_begin()
    http.reset_mock()
    http.decode.return_value = _empty_decode_resp()

    backend.push_frame(audio_chunk=np.array([], dtype=np.float32))
    assert http.prefill.call_count == 0
    assert http.decode.call_count == 1
    assert backend._frame_idx == 2  # 1 + 0 vision + 1 (audio slot consumed)


def test_return_logits_triggers_inline_logit_format(make_backend):
    """When session_begin(return_logits=True), every push_frame must
    request inline logit capture on its decode call (so the
    per-chunk logits payload is parsed out of the SSE)."""
    backend, _, http = make_backend(use_tts=False)
    backend.session_begin(return_logits=True)
    http.reset_mock()
    http.decode.return_value = _empty_decode_resp()
    backend.push_frame(audio_chunk=np.zeros(2048, dtype=np.float32))
    assert http.decode.call_args.kwargs["logit_format"] == "inline"


def test_no_logits_when_capture_disabled(make_backend):
    backend, _, http = make_backend(use_tts=False)
    backend.session_begin(return_logits=False)
    http.reset_mock()
    http.decode.return_value = _empty_decode_resp()
    backend.push_frame(audio_chunk=np.zeros(2048, dtype=np.float32))
    assert http.decode.call_args.kwargs["logit_format"] is None


def test_session_end_breaks_then_full_restarts(make_backend):
    """Default session_end semantics: send /break to drain, then hard
    restart the subprocess."""
    backend, proc, http = make_backend()
    backend.session_begin()
    http.break_.reset_mock()
    backend.session_end()
    assert http.break_.called
    assert proc.full_restart.called


def test_session_end_no_restart_keeps_subprocess(make_backend):
    backend, proc, _ = make_backend()
    backend.session_end(full_restart=False)
    assert not proc.full_restart.called


def test_use_tts_false_skips_wav_collection(make_backend):
    """When TTS is disabled at the worker level, push_frame must not
    poll the output dir (no WAVs to collect)."""
    backend, _, http = make_backend(use_tts=False)
    backend.session_begin()
    http.decode.return_value = _empty_decode_resp()
    backend.push_frame(audio_chunk=np.zeros(2048, dtype=np.float32))
    backend._dir_mgr.collect_nowait.assert_not_called()


def test_use_tts_true_drains_audio_per_frame(make_backend):
    backend, _, http = make_backend(use_tts=True)
    backend.session_begin()
    http.decode.return_value = _empty_decode_resp()
    backend._dir_mgr.collect_nowait.return_value = ("AUDIO_B64", "")
    out = backend.push_frame(audio_chunk=np.zeros(2048, dtype=np.float32))
    assert out["audio_data"] == "AUDIO_B64"
