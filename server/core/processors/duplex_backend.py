"""Duplex (always-on conversational) backend for the C++ llama-server.

Aligned with the ``tools/omni/test/test-duplex.cpp`` calling convention:

    omni_init(duplex_mode=true)
    update_session_config(duplex=True, sampling, return_logits)
        - clears KV + arms ``system_prompt_initialized=false``
    prefill(cnt=0, audio=ref_audio_path)
        - rebuilds the system-prompt + ref-audio embedding
          (``omni.cpp::stream_prefill`` index=0 branch); equivalent to
          ``omni_duplex_session_begin``'s implicit init
    for chunk_i in 1..N:
        prefill(audio=chunk_i, cnt=frame_idx)            # push_frame
        for img in vision_frames:
            prefill(img=..., cnt=frame_idx + offset)
        decode(round_idx=frame_idx, force_listen=...,
               logit_format="inline")                    # wait_next_frame
    session_end -> full_restart                          # omni_duplex_session_end

This single backend class fixes nine independent bugs in the legacy
``CppBackendWorker.duplex_*`` methods at once:

    D2  ref_audio_path actually flows to prefill(cnt=0) - the legacy
        BUG FIX 2 path silently swallowed it under update_session_config
    D3  prefill(cnt=0) is no longer rejected by the server (Phase 0
        server.cpp patch) and the client raises on failure instead of
        logging
    D5  ``ref_audio_override`` per-session input lets each session use
        a different TTS voice without mutating the worker singleton
    D7  ``force_listen`` is plumbed all the way to /v1/stream/decode
    D8  ``round_idx`` is sent on every decode (test-duplex.cpp:11230
        contract); the legacy code passed it only on simplex chat
    D9  Full sampling whitelist (incl. ``llm_sampling`` LLM-main fields)
        is forwarded by ``_StreamHttpClient.update_session_config``

Used by:
    POST /duplex_offline    -> session_begin + loop push_frame + session_end
    WS   /ws/duplex         -> session_begin + per-chunk push_frame +
                                session_end (full_restart on disconnect)
"""
from __future__ import annotations

import os
import json
import time
import logging
import tempfile
from typing import Optional, Any, Dict, List

import numpy as np

from .cpp_session import (
    _CppServerProc,
    _StreamHttpClient,
    _OutputDirManager,
    build_prompts_from_content,
    extract_logits_from_sse,
    save_audio_to_temp,
    save_pil_image_to_temp,
    cleanup_temp_files,
)

logger = logging.getLogger("duplex_backend")


class DuplexCppBackend:
    """Per-session duplex driver.

    A duplex session is the unit of bring-up cost: each session starts
    from a fresh KV cache (the previous session's ``session_end`` is
    expected to ``full_restart`` the C++ subprocess - this is the
    ``omni_duplex_session_end`` equivalent). Within a session,
    ``push_frame`` is called once per ~1s audio slice and returns the
    listen/speak/text/audio decision for that frame.
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
        self._temp_dir = temp_dir or tempfile.mkdtemp(prefix="duplex_backend_")
        self._dir_mgr = _OutputDirManager(self._output_dir)

        # Session-local state, reset on every session_begin
        self._frame_idx: int = 0
        self._length_penalty: float = 1.1
        self._return_logits: bool = False
        self._last_break_time: float = 0.0

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def session_begin(
        self,
        *,
        system_content: Any = None,
        sampling: Optional[Dict[str, Any]] = None,
        lang: str = "zh",
        media_type: int = 2,
        ref_audio_override: Optional[str] = None,
        return_logits: bool = False,
        length_penalty: float = 1.1,
    ) -> None:
        """Start a fresh duplex session.

        Parameters mirror ``DuplexBatchRequest`` / ``DuplexPrepareRequest``:

          * ``system_content`` - request-level system prompt (string or
            ``[{type:"text"|"audio", ...}]`` shape).
          * ``sampling`` - whitelisted dict; backend layer is responsible
            for filtering down to ``listen_prob_scale`` /
            ``force_listen_count`` / ``max_new_speak_tokens_per_chunk`` /
            ``tts_temperature`` plus a nested ``llm_sampling`` for the
            LLM main sampler. ``_StreamHttpClient`` then passes them all
            through verbatim.
          * ``ref_audio_override`` - per-session ref audio path. Falls
            back to the worker-level default. The path is fed to the
            ``prefill(cnt=0)`` system-init slot **and** to the implicit
            inner prefill that ``update_session_config`` would otherwise
            run; the latter is bypassed here because the legacy
            ``voice_audio`` field on update_session_config triggered the
            BUG FIX 2 ``media_type`` corruption. Doing the ref-audio
            embedding via prefill(cnt=0) is the test-duplex.cpp
            convention.
          * ``return_logits`` - turn on per-chunk inline logits capture.
            Each ``push_frame`` will then carry a ``logits`` payload in
            its return dict.
          * ``length_penalty`` - sticky for all decode calls in the
            session; default 1.1 matches the legacy DuplexProcessor.
        """
        ref = ref_audio_override or self.ref_audio_path
        if not ref or not os.path.exists(ref):
            # The Phase 0 server.cpp patch lets cnt=0 be all-empty if
            # ``ctx_omni->ref_audio_path`` is already populated by
            # omni_init. We still emit a warning so the cause of any
            # subsequent silence is loud in the logs.
            logger.warning(
                "session_begin: ref_audio path missing or doesn't exist "
                f"(ref_audio_override={ref_audio_override!r}, "
                f"worker_default={self.ref_audio_path!r}); falling back to "
                "ctx_omni->ref_audio_path set by omni_init"
            )
            ref = ""

        prompts = build_prompts_from_content(system_content, duplex=True, lang=lang)

        self._http.update_session_config(
            media_type=media_type,
            duplex_mode=True,
            voice_clone_prompt=prompts["voice_clone_prompt"],
            assistant_prompt=prompts["assistant_prompt"],
            sampling=sampling,
            return_logits=return_logits,
        )

        self._dir_mgr.reset()

        # System-init prefill. The C++ side reads ref-audio from this
        # call's ``audio_path_prefix`` (or falls back to
        # ``ctx_omni->ref_audio_path``) and rebuilds
        # <|im_start|>system\n...<|audio_start|>[ref]<|audio_end|>...
        # <|im_end|>\n. After this, ``system_prompt_initialized`` is
        # true and subsequent push_frame calls go through the proper
        # duplex_prefill branch in omni.cpp.
        self._http.prefill(audio_path=ref, cnt=0)

        self._frame_idx = 1
        self._length_penalty = float(length_penalty)
        self._return_logits = bool(return_logits)
        self._last_break_time = 0.0

    # ------------------------------------------------------------------
    # Per-chunk frame
    # ------------------------------------------------------------------

    def push_frame(
        self,
        *,
        audio_chunk: Optional[np.ndarray] = None,
        vision_frames: Optional[List[Any]] = None,
        force_listen: bool = False,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        """Push one ~1s audio slice (and optional video frames) and
        immediately ask the model whether it wants to listen or speak.

        Returns a flat dict with the same fields the legacy
        ``DuplexGenerateResult`` exposes plus a few extras the upper
        layer translates to its WS event:

          * ``is_listen`` - bool, model's listen/speak decision
          * ``end_of_turn`` - bool, model thinks the user finished
          * ``text`` - any text emitted this frame (non-empty only when
            ``is_listen=False``)
          * ``audio_data`` - any new TTS WAV ready *as of this frame*
            (incremental polling via ``_OutputDirManager.collect_nowait``)
          * ``logits`` - per-chunk LogitsPayload if capture was on
          * ``cost_all_ms`` / ``kv_cache_length`` / ``current_time`` -
            instrumentation
          * ``n_vision_frames`` - count of vision frames that were
            prefilled this turn
        """
        t0 = time.perf_counter()
        cnt_at_call = self._frame_idx
        n_vision_frames = 0

        # 1. Audio prefill (cnt = current frame_idx).
        if audio_chunk is not None and len(audio_chunk) > 0:
            audio_path = save_audio_to_temp(
                self._temp_dir, audio_chunk, f"dx_{cnt_at_call}"
            )
            try:
                self._http.prefill(audio_path=audio_path, cnt=cnt_at_call)
            finally:
                cleanup_temp_files(audio_path)
        else:
            # Empty audio still needs a frame slot to keep cnt monotonic;
            # but cnt>0 with empty body would 400 (Phase 0 patch only
            # relaxes cnt==0). We skip the prefill entirely - the C++
            # decode will treat this as "no new user audio" which is
            # what an empty chunk should mean.
            pass

        # 2. Vision frames, each with its own cnt slot.
        if vision_frames:
            for offset, frame in enumerate(vision_frames):
                slot = cnt_at_call + 1 + offset
                img_path = save_pil_image_to_temp(
                    self._temp_dir, frame, f"dx_{cnt_at_call}_f{offset}"
                )
                try:
                    self._http.prefill(
                        img_path=img_path,
                        cnt=slot,
                        max_slice_nums=max_slice_nums,
                    )
                finally:
                    cleanup_temp_files(img_path)
                n_vision_frames += 1

        # 3. Decode this frame. round_idx mirrors the C++ test-duplex
        #    convention; force_listen lets the upper layer pin LISTEN
        #    when the VAD or the protocol says we shouldn't barge in.
        decode_resp = self._http.decode(
            stream=True,
            round_idx=cnt_at_call,
            length_penalty=self._length_penalty,
            force_listen=True if force_listen else None,
            logit_format="inline" if self._return_logits else None,
            timeout=600.0,
        )

        # 4. Advance frame counter past the audio + vision slots we used.
        self._frame_idx = cnt_at_call + 1 + n_vision_frames

        # 5. Parse the SSE body. Duplex decode emits one batch per call:
        #    a header event with is_listen / end_of_turn, optional
        #    content/text events, and (if capture is on) a final
        #    logits event right before [DONE].
        is_listen = True
        end_of_turn = False
        texts: List[str] = []
        cost_llm_ms = None
        cost_tts_prep_ms = None
        cost_tts_ms = None
        cost_token2wav_ms = None
        n_tokens = None
        n_tts_tokens = None
        logits_payload = None

        if decode_resp.status_code == 200:
            if self._return_logits:
                logits_payload = extract_logits_from_sse(decode_resp.text)
            for line in decode_resp.text.splitlines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if "is_listen" in event:
                    is_listen = bool(event["is_listen"])
                if "end_of_turn" in event:
                    end_of_turn = bool(event["end_of_turn"])
                if event.get("text"):
                    texts.append(event["text"])
                if event.get("content"):
                    texts.append(event["content"])
                # Cost / token stats (best-effort; C++ may rename later)
                for src, dst in (
                    ("cost_llm_ms", "cost_llm_ms"),
                    ("cost_tts_prep_ms", "cost_tts_prep_ms"),
                    ("cost_tts_ms", "cost_tts_ms"),
                    ("cost_token2wav_ms", "cost_token2wav_ms"),
                    ("n_tokens", "n_tokens"),
                    ("n_tts_tokens", "n_tts_tokens"),
                ):
                    if src in event and event[src] is not None:
                        v = event[src]
                        if dst.startswith("cost"):
                            cost_var = locals().get(dst)
                            if cost_var is None:
                                pass  # set below
                        if dst == "cost_llm_ms":
                            cost_llm_ms = float(v)
                        elif dst == "cost_tts_prep_ms":
                            cost_tts_prep_ms = float(v)
                        elif dst == "cost_tts_ms":
                            cost_tts_ms = float(v)
                        elif dst == "cost_token2wav_ms":
                            cost_token2wav_ms = float(v)
                        elif dst == "n_tokens":
                            n_tokens = int(v)
                        elif dst == "n_tts_tokens":
                            n_tts_tokens = int(v)
        else:
            logger.warning(
                f"duplex decode non-200 status={decode_resp.status_code} "
                f"body={decode_resp.text[:200]!r}"
            )

        text = "".join(texts)

        # 6. Drain any new TTS WAV chunks (if any). Non-blocking - the
        #    C++ TTS thread runs async and may still be writing files
        #    after this call returns; we just take what's ready.
        wav_b64 = None
        if self.use_tts:
            wav_b64, _ = self._dir_mgr.collect_nowait(sse_text=text)

        cost_all_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "is_listen": is_listen,
            "end_of_turn": end_of_turn,
            "text": text,
            "audio_data": wav_b64,
            "logits": logits_payload,
            "current_time": cnt_at_call,
            "cost_llm_ms": cost_llm_ms,
            "cost_tts_prep_ms": cost_tts_prep_ms,
            "cost_tts_ms": cost_tts_ms,
            "cost_token2wav_ms": cost_token2wav_ms,
            "cost_all_ms": round(cost_all_ms, 1),
            "n_tokens": n_tokens,
            "n_tts_tokens": n_tts_tokens,
            "n_vision_frames": n_vision_frames,
            "kv_cache_length": self._http.kv_cache_length,
        }

    # ------------------------------------------------------------------
    # Control + teardown
    # ------------------------------------------------------------------

    def break_now(self, reason: str = "duplex_stop") -> None:
        self._last_break_time = time.time()
        self._http.break_(reason=reason)

    def session_end(self, *, full_restart: bool = True) -> None:
        """Tear down the duplex session.

        ``full_restart=True`` (the default) hard-restarts llama-server
        so the next session starts from absolutely clean state. This
        matches the legacy ``worker.py`` /ws/duplex and /duplex_offline
        finally blocks and is the test-duplex equivalent of
        ``omni_duplex_session_end``.

        ``full_restart=False`` is provided as an escape hatch for tests
        and for cases where the upper layer wants to chain sessions
        on the same subprocess (rare; not currently used).
        """
        try:
            self.break_now(reason="session_end")
        except Exception:
            pass

        if full_restart:
            self._proc.full_restart()

    @property
    def kv_cache_length(self) -> int:
        return self._http.kv_cache_length
