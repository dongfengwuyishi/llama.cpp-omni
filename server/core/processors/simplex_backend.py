"""Simplex chat backend for the C++ llama-server.

Aligned with the ``tools/omni/omni-cli.cpp`` calling convention:

    update_session_config(duplex_mode=False)
        - flushes the C++ KV cache and re-arms the system prompt
    prefill(cnt=0, audio=ref_audio_path)
        - triggers omni.cpp's system-prompt + ref-audio embedding rebuild
          (relies on the Phase 0 server.cpp patch that allows cnt==0 to
          carry only ref-audio in ``audio_path_prefix`` - or all-empty
          when ``ctx_omni->ref_audio_path`` is already populated by
          omni_init)
    push_audio / push_text / push_image  (cnt=1, 2, ...)
        - one prefill per content item
    decode_streaming / decode_oneshot (round_idx=...)
        - exactly one decode call per turn

Used by:
    POST /chat                  -> begin_turn + push_* + decode_oneshot
    WS   /ws/chat               -> begin_turn + push_* + decode_streaming
    WS   /ws/half_duplex        -> begin_turn (per VAD turn) + push_audio
                                    + decode_streaming
    WS   /ws/half_duplex_omni   -> same + push_image for video frames

The class deliberately stays free of FastAPI / WebSocket plumbing; turn
boundaries (VAD, queue back-pressure, recording, etc.) are upper-layer
concerns that ``worker.py`` keeps owning.
"""
from __future__ import annotations

import os
import time
import logging
import tempfile
from typing import Optional, Any, Dict, Iterator, List

import numpy as np

from .cpp_session import (
    _CppServerProc,
    _StreamHttpClient,
    _OutputDirManager,
    build_prompts_from_content,
    parse_sse_text,
    extract_logits_from_sse,
    save_audio_to_temp,
    save_pil_image_to_temp,
    cleanup_temp_files,
    _AUDIO_OUTPUT_SR,
)

logger = logging.getLogger("simplex_backend")


class SimplexCppBackend:
    """Per-turn simplex driver against ``_StreamHttpClient``.

    Each turn must follow ``begin_turn -> push_* -> decode_* -> end_turn``.
    The class holds enough state to count prefill ``cnt`` and the round
    index across a single turn; consecutive turns are independent (the
    worker.py layer decides whether to ``full_restart`` the subprocess
    between turns).
    """

    def __init__(
        self,
        *,
        proc: _CppServerProc,
        http: _StreamHttpClient,
        ref_audio_path: Optional[str],
        worker_idx: int,
        use_tts: bool,
        output_dir: Optional[str] = None,
        temp_dir: Optional[str] = None,
    ):
        self._proc = proc
        self._http = http
        self.ref_audio_path = ref_audio_path
        self.worker_idx = int(worker_idx)
        self.use_tts = bool(use_tts)
        self._output_dir = output_dir or proc.output_dir
        self._temp_dir = temp_dir or tempfile.mkdtemp(prefix="simplex_backend_")
        self._dir_mgr = _OutputDirManager(self._output_dir)

        # Turn-local state (reset on every begin_turn)
        self._cnt: int = 0
        self._round_idx: int = 0
        self._return_logits: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_turn(
        self,
        *,
        system_content: Any = None,
        sampling: Optional[Dict[str, Any]] = None,
        lang: str = "zh",
        media_type: int = 2,
        return_logits: bool = False,
    ) -> None:
        """Start a fresh simplex turn.

        Sequence:
          1. ``update_session_config(duplex=False, ...)`` clears KV +
             arms ``system_prompt_initialized=false``.
          2. ``prefill(cnt=0, audio=ref_audio_path)`` makes the C++
             runtime rebuild the system prompt + ref-audio embedding
             (omni.cpp:10096 contract). The Phase 0 server.cpp patch
             allows ``cnt==0`` even when ``audio_path_prefix`` is empty
             (server falls back to ``ctx_omni->ref_audio_path`` set by
             omni_init), but we still pass the ref path explicitly when
             we have one for clarity and to keep the contract symmetric
             with DuplexCppBackend.
          3. ``self._cnt = 1`` so subsequent ``push_*`` calls start at
             ``cnt=1`` - the C++ side silently drops user content from a
             ``cnt=0`` frame.
        """
        prompts = build_prompts_from_content(system_content, duplex=False, lang=lang)

        self._http.update_session_config(
            media_type=media_type,
            duplex_mode=False,
            voice_clone_prompt=prompts["voice_clone_prompt"],
            assistant_prompt=prompts["assistant_prompt"],
            sampling=sampling,
            return_logits=return_logits,
        )

        self._dir_mgr.reset()

        ref = ""
        if self.ref_audio_path and os.path.exists(self.ref_audio_path):
            ref = self.ref_audio_path
        # cnt=0 system-init slot. Phase 0 patch lets this body be all-empty;
        # we pass ref when available so the C++ log line shows the actual
        # path being used.
        self._http.prefill(audio_path=ref, cnt=0)

        self._cnt = 1
        self._round_idx = 0
        self._return_logits = bool(return_logits)

    def push_audio(self, audio_np: np.ndarray) -> None:
        path = save_audio_to_temp(self._temp_dir, audio_np, f"sx_{self._cnt}")
        try:
            self._http.prefill(audio_path=path, cnt=self._cnt)
        finally:
            cleanup_temp_files(path)
        self._cnt += 1

    def push_text(self, text: str) -> None:
        if not text:
            return
        self._http.prefill(text=text, cnt=self._cnt)
        self._cnt += 1

    def push_image(self, pil_image, max_slice_nums: int = -1) -> None:
        path = save_pil_image_to_temp(self._temp_dir, pil_image, f"sx_{self._cnt}")
        try:
            self._http.prefill(
                img_path=path, cnt=self._cnt, max_slice_nums=max_slice_nums
            )
        finally:
            cleanup_temp_files(path)
        self._cnt += 1

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_oneshot(
        self,
        *,
        length_penalty: float = 1.1,
        max_new_tokens: Optional[int] = None,
        logit_format: Optional[str] = None,
        logit_output_dir: Optional[str] = None,
        logit_filename: Optional[str] = None,
        logit_extra_metadata: Optional[Dict[str, Any]] = None,
        want_audio: bool = False,
    ) -> Dict[str, Any]:
        """One-shot decode: run ``/v1/stream/decode`` in stream mode but
        block until the SSE body is fully received, then optionally
        wait for the C++ TTS thread to finish writing WAVs and bundle
        them into a single base64 payload.

        Returns a flat dict so callers can map straight into either
        ``ChatResponse`` (POST /chat) or ``StreamingChunk`` (test
        scaffolding) without committing to a pydantic shape here.
        """
        resp = self._http.decode(
            stream=True,
            round_idx=self._round_idx,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            logit_format=logit_format,
            logit_output_dir=logit_output_dir,
            logit_filename=logit_filename,
            logit_extra_metadata=logit_extra_metadata,
            timeout=600.0,
        )
        self._round_idx += 1

        text = ""
        logits_payload = None
        if resp.status_code == 200:
            text = parse_sse_text(resp.text)
            if self._return_logits:
                logits_payload = extract_logits_from_sse(resp.text)
        else:
            logger.warning(
                f"decode non-200 status={resp.status_code} body={resp.text[:200]!r}"
            )

        wav_b64 = None
        if want_audio and self.use_tts:
            wav_b64, _ = self._dir_mgr.collect_blocking(sse_text=text)

        return {
            "text": text,
            "audio_data": wav_b64,
            "audio_sample_rate": _AUDIO_OUTPUT_SR if wav_b64 else None,
            "logits": logits_payload,
            "kv_cache_length": self._http.kv_cache_length,
        }

    def decode_streaming(
        self,
        *,
        length_penalty: float = 1.1,
        max_new_tokens: Optional[int] = None,
        logit_format: Optional[str] = None,
        logit_output_dir: Optional[str] = None,
        logit_filename: Optional[str] = None,
        logit_extra_metadata: Optional[Dict[str, Any]] = None,
        generate_audio: bool = True,
        wav_iter_timeout: float = 120.0,
    ) -> Iterator[Dict[str, Any]]:
        """Streaming decode: yields a sequence of ``{type, ...}`` dicts:

          * ``{"type": "text", "delta": "..."}``  per text chunk
          * ``{"type": "audio", "data": "<b64>"}`` per WAV file (if
            ``generate_audio`` and ``use_tts``)
          * ``{"type": "done", "text": "...", "logits": ..., ...}`` at
            end of stream

        Mirrors the legacy ``half_duplex_generate`` semantics but in
        a callback-free shape that the worker.py WS handlers can
        translate directly to their existing chunk events.

        Implementation note: the C++ /v1/stream/decode endpoint returns
        the SSE body all at once (it does NOT use chunked transfer
        encoding even with ``stream=true``), so we emit one ``text``
        event with the full assistant text plus interleaved ``audio``
        events for each WAV that has shown up by the time the body
        completes. This matches what ``half_duplex_generate`` did for
        chat WS clients.
        """
        resp = self._http.decode(
            stream=True,
            round_idx=self._round_idx,
            length_penalty=length_penalty,
            max_new_tokens=max_new_tokens,
            logit_format=logit_format,
            logit_output_dir=logit_output_dir,
            logit_filename=logit_filename,
            logit_extra_metadata=logit_extra_metadata,
            timeout=600.0,
        )
        self._round_idx += 1

        if resp.status_code != 200:
            yield {
                "type": "error",
                "message": f"decode non-200 status={resp.status_code}",
                "body": resp.text[:200] if resp.text else "",
            }
            return

        text = parse_sse_text(resp.text)
        logits_payload = extract_logits_from_sse(resp.text) if self._return_logits else None

        if text:
            yield {"type": "text", "delta": text}

        if generate_audio and self.use_tts:
            for wav_b64 in self._dir_mgr.iter_chunks(timeout=wav_iter_timeout):
                yield {
                    "type": "audio",
                    "data": wav_b64,
                    "sample_rate": _AUDIO_OUTPUT_SR,
                }

        yield {
            "type": "done",
            "text": text,
            "logits": logits_payload,
            "kv_cache_length": self._http.kv_cache_length,
        }

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def break_now(self, reason: str = "manual") -> None:
        self._http.break_(reason=reason)

    def end_turn(self, *, full_restart: bool = False) -> None:
        """Optional turn teardown.

        ``full_restart=True`` kills llama-server and re-runs ``omni_init``
        - matches the legacy ``full_reinit`` sequence the WS half-duplex
        handlers run on disconnect to guarantee a clean state for the
        next session. POST /chat doesn't need it (the next turn's
        ``begin_turn`` clears KV anyway).
        """
        if full_restart:
            self._proc.full_restart()

    @property
    def kv_cache_length(self) -> int:
        return self._http.kv_cache_length
