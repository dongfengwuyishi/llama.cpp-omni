"""C++ llama-server backend infrastructure.

Two thin classes shared by ``SimplexCppBackend`` / ``DuplexCppBackend``:

  * ``_CppServerProc``   - llama-server subprocess lifecycle + ``omni_init``
  * ``_StreamHttpClient`` - thin wrapper over ``/v1/stream/*`` HTTP endpoints

These have no business semantics (no prompt template construction, no
per-request sampling whitelist, no logit-filename minting) - those concerns
live in the backend classes that consume this module.

This module is the new home for ``CppBackendWorker._start_cpp_server`` /
``_stop_cpp_server`` / ``full_reinit`` / ``_call_omni_init`` /
``_call_update_session_config`` / ``_call_prefill``. The legacy methods
remain in ``cpp_backend.py`` until Phase 4 of the refactor finishes
switching the worker.py endpoints to the new backends.

Notable behavior change vs the legacy methods:

  * ``prefill`` raises on non-200 instead of silently logging and
    returning. The legacy ``_call_prefill`` swallowed errors which masked
    the system-prompt re-init failure (D3 bug): with the unpatched
    server.cpp, ``prefill(cnt=0, audio="", img="", text="")`` returned 400
    but the Python side never noticed, so the C++ ``system_prompt_initialized``
    stayed false and subsequent user-content prefills degenerated to a
    legacy queue path that never spoke.
"""

from __future__ import annotations

import os
import time
import signal
import logging
import platform
import threading
import subprocess
from typing import Optional, Any, Dict

import httpx

logger = logging.getLogger("cpp_session")


class _CppServerProc:
    """llama-server subprocess lifecycle + initial omni_init wiring.

    Constructor params mirror the subset of ``CppBackendWorker.__init__``
    that the subprocess actually needs - it intentionally ignores
    ``worker_idx`` and other backend-only concerns.
    """

    def __init__(
        self,
        *,
        llamacpp_root: str,
        model_dir: str,
        gpu_id: int = 0,
        port: int = 19060,
        ctx_size: int = 32768,
        n_gpu_layers: int = 99,
        use_tts: bool = True,
        llm_model: str = "",
        ref_audio_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.llamacpp_root = llamacpp_root
        self.model_dir = model_dir
        self.gpu_id = gpu_id
        self.port = port
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        self.use_tts = bool(use_tts)
        self.llm_model = llm_model
        self.ref_audio_path = ref_audio_path
        self.output_dir = output_dir or os.path.join(
            llamacpp_root, f"tools/omni/output_{port}"
        )

        self._proc: Optional[subprocess.Popen] = None
        # Shared httpx client used by both this class (for omni_init) and
        # any _StreamHttpClient that wraps the same server. Created lazily
        # in ``start()``. ``trust_env=False`` is critical on Windows where
        # httpx would otherwise pick up IE's system proxy and tunnel local
        # 127.0.0.1 traffic through Clash/V2Ray (legacy BUG FIX 1).
        self._http_client: Optional[httpx.Client] = None
        self._last_kv_cache_length: int = 0
        self._last_omni_init_args: Dict[str, Any] = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def http_client(self) -> httpx.Client:
        if self._http_client is None:
            raise RuntimeError("_CppServerProc.start() must be called first")
        return self._http_client

    @property
    def kv_cache_length(self) -> int:
        return int(self._last_kv_cache_length)

    def maybe_update_kv_cache_length(self, payload: Any) -> None:
        if isinstance(payload, dict) and "kv_cache_length" in payload:
            try:
                self._last_kv_cache_length = int(
                    payload.get("kv_cache_length", 0) or 0
                )
            except (TypeError, ValueError):
                logger.debug(
                    "invalid kv_cache_length payload: %r",
                    payload.get("kv_cache_length"),
                )

    def start(self) -> None:
        """Launch llama-server subprocess and poll /health until ready."""
        if self._http_client is None:
            self._http_client = httpx.Client(
                timeout=httpx.Timeout(600.0, connect=30.0),
                trust_env=False,
            )

        server_bin = self._find_server_binary()
        model_path = os.path.join(self.model_dir, self.llm_model)
        if not os.path.exists(server_bin):
            raise RuntimeError(f"llama-server not found: {server_bin}")
        if not os.path.exists(model_path):
            raise RuntimeError(f"LLM model not found: {model_path}")

        env = os.environ.copy()
        # GPU pinning: only set CUDA_VISIBLE_DEVICES if the parent did not
        # already restrict device visibility. Blindly overwriting would
        # remap a parent's narrowed window (e.g. parent's "1") back to
        # physical GPU 0 inside the child.
        if not env.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            logger.info(
                f"[GPU {self.gpu_id}] parent CUDA_VISIBLE_DEVICES unset; "
                f"pinning C++ child to physical GPU {self.gpu_id}"
            )
        else:
            logger.info(
                f"[GPU {self.gpu_id}] parent CUDA_VISIBLE_DEVICES="
                f"{env['CUDA_VISIBLE_DEVICES']!r}; inheriting (not overriding)"
            )

        # OMNI_LLM_SAMPLE_TEMP / OMNI_LLM_REPEAT_PENALTY / OMNI_LLM_SAMPLE_SEED:
        # startup-time defaults for the LLM main sampler. Per-request override
        # goes through ``_StreamHttpClient.update_session_config(sampling=...)``
        # carrying a nested ``llm_sampling`` dict; the C++ side rebuilds
        # ``ctx_omni->ctx_sampler`` accordingly.
        sample_temp = os.environ.get("OMNI_LLM_SAMPLE_TEMP", "0.7")
        repeat_penalty = os.environ.get("OMNI_LLM_REPEAT_PENALTY", "1.05")
        sample_seed = os.environ.get("OMNI_LLM_SAMPLE_SEED")
        cmd = [
            server_bin,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--model", model_path,
            "--ctx-size", str(self.ctx_size),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--repeat-penalty", repeat_penalty,
            "--temp", sample_temp,
        ]
        if sample_seed is not None:
            cmd.extend(["--seed", sample_seed])
        if (
            sample_temp != "0.7"
            or repeat_penalty != "1.05"
            or sample_seed is not None
        ):
            logger.info(
                f"[LLM sampler override] temp={sample_temp} "
                f"repeat-penalty={repeat_penalty} seed={sample_seed!r} "
                "(env: OMNI_LLM_SAMPLE_TEMP / OMNI_LLM_REPEAT_PENALTY / "
                "OMNI_LLM_SAMPLE_SEED)"
            )

        logger.info(f"Starting C++ server: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=self.llamacpp_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )

        proc_ref = self._proc

        def _log_reader():
            try:
                for line in proc_ref.stdout:
                    stripped = line.rstrip()
                    if any(
                        kw in stripped
                        for kw in (
                            "TTS", "T2W", "LLM->TTS", "wav_", "tts_thread",
                            "generate_audio", "speek_done", "break_event",
                            "lang", "language", "omni_set_language", "prefill",
                            "change", "stream_decode", "LLM thread", "LLM Duplex",
                            "force_listen", "LLM decode", "EOS", "EOG", "sample",
                            "is_listen", "duplex_decode",
                        )
                    ):
                        logger.info(f"[CPP] {stripped}")
                    else:
                        logger.debug(f"[CPP] {stripped}")
            except Exception:
                pass

        threading.Thread(target=_log_reader, daemon=True).start()

        # ``requests`` for the boot health-poll: explicit no-proxy keeps
        # local 127.0.0.1 traffic from being hijacked by HTTP_PROXY env
        # set on shared workstations.
        import requests
        no_proxy = {"http": None, "https": None}
        for i in range(300):
            try:
                r = requests.get(
                    f"{self.url}/health", timeout=2, proxies=no_proxy
                )
                if r.status_code == 200:
                    logger.info(f"C++ server ready after {i+1}s")
                    return
            except Exception:
                pass
            time.sleep(1)

        raise RuntimeError("C++ server startup timeout (300s)")

    def health(self) -> bool:
        if self._http_client is None:
            return False
        try:
            r = self._http_client.get(f"{self.url}/health", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def stop(self) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except Exception:
                    proc.terminate()

                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        proc.kill()
                    proc.wait(timeout=5)
        except Exception as e:
            logger.warning(f"_CppServerProc.stop: {e}")
        finally:
            self._proc = None
            logger.info("llama-server stopped")

    def call_omni_init(
        self,
        *,
        media_type: int = 2,
        duplex_mode: bool = True,
        voice_clone_prompt: str = "",
        assistant_prompt: str = "",
    ) -> None:
        """Initial /v1/stream/omni_init call after server start.

        Prompt strings are pre-built by the backend layer (e.g. via the
        backend's ``_build_prompts_from_content``) - this method is
        intentionally dumb and does not embed any default templates.
        """
        tts_bin_dir = os.path.join(self.model_dir, "tts")
        os.makedirs(self.output_dir, exist_ok=True)

        req_body: Dict[str, Any] = {
            "media_type": media_type,
            "use_tts": self.use_tts,
            "duplex_mode": duplex_mode,
            "model_dir": self.model_dir,
            "tts_bin_dir": tts_bin_dir,
            "tts_gpu_layers": 100,
            "token2wav_device": "gpu:0",
            "output_dir": self.output_dir,
        }
        if self.ref_audio_path and os.path.exists(self.ref_audio_path):
            req_body["voice_audio"] = self.ref_audio_path
        if voice_clone_prompt:
            req_body["voice_clone_prompt"] = voice_clone_prompt
        if assistant_prompt:
            req_body["assistant_prompt"] = assistant_prompt

        logger.info(
            f"Calling omni_init: media_type={media_type}, "
            f"duplex={duplex_mode}, ref_audio={self.ref_audio_path!r}"
        )
        resp = self.http_client.post(
            f"{self.url}/v1/stream/omni_init",
            json=req_body,
            timeout=120.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"omni_init failed: {resp.text}")
        payload = resp.json()
        self.maybe_update_kv_cache_length(payload)
        self._last_omni_init_args = {
            "media_type": media_type,
            "duplex_mode": duplex_mode,
            "voice_clone_prompt": voice_clone_prompt,
            "assistant_prompt": assistant_prompt,
        }
        logger.info(f"omni_init success: {payload}")

    def full_restart(
        self,
        *,
        omni_init_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Hard restart: kill subprocess, relaunch, re-call omni_init.

        Uses the last cached omni_init args by default. Pass
        ``omni_init_args`` to override (e.g. duplex_mode flip). This is
        the equivalent of test-duplex.cpp's ``omni_duplex_session_end``
        followed by a fresh ``omni_duplex_session_begin``.
        """
        args = omni_init_args or self._last_omni_init_args or {
            "media_type": 2,
            "duplex_mode": True,
            "voice_clone_prompt": "",
            "assistant_prompt": "",
        }
        try:
            logger.info("full_restart: stopping llama-server...")
            self.stop()
            logger.info("full_restart: restarting llama-server...")
            self.start()
            self.call_omni_init(**args)
            logger.info("full_restart: omni context re-initialized")
        except Exception as e:
            logger.error(f"full_restart failed: {e}", exc_info=True)
            raise

    def _find_server_binary(self) -> str:
        is_win = platform.system() == "Windows"
        candidates = []
        if is_win:
            candidates += [
                os.path.join(self.llamacpp_root, "build", "bin", "Release", "llama-server.exe"),
                os.path.join(self.llamacpp_root, "build", "bin", "llama-server.exe"),
            ]
        candidates += [
            os.path.join(self.llamacpp_root, "build/bin/llama-server"),
            os.path.join(self.llamacpp_root, "build/bin/Release/llama-server"),
        ]
        if not is_win:
            candidates.append(
                os.path.join(
                    self.llamacpp_root,
                    "build-x64-linux-cuda-release/bin/llama-server",
                )
            )
        for c in candidates:
            if os.path.exists(c):
                return c
        return candidates[0]


class _StreamHttpClient:
    """Thin wrapper over the C++ /v1/stream/* HTTP endpoints.

    Intentionally has no business semantics:

      * prompt template construction lives in the backend layer
        (``SimplexCppBackend`` / ``DuplexCppBackend``).
      * the per-request sampling whitelist lives there too - this class
        just dumps whatever dict it is handed under ``sampling``.
      * logit-filename minting (collision-safe naming under multi-worker
        load) is the backend's responsibility.

    The single non-trivial behavior change vs the legacy
    ``CppBackendWorker._call_prefill`` is that ``prefill()`` raises on
    non-200 instead of logging and returning. The legacy behavior masked
    the D3 system-prompt re-init failure: with the unpatched server.cpp
    the bootstrap ``prefill(cnt=0, audio="", img="", text="")`` returned
    400 but the worker silently moved on and the C++ KV stayed without
    a system prompt.
    """

    def __init__(self, base_url: str, http_client: httpx.Client):
        self.base_url = base_url
        self.http = http_client
        self._last_kv_cache_length: int = 0

    @property
    def kv_cache_length(self) -> int:
        return int(self._last_kv_cache_length)

    def _maybe_update_kv(self, payload: Any) -> None:
        if isinstance(payload, dict) and "kv_cache_length" in payload:
            try:
                self._last_kv_cache_length = int(
                    payload.get("kv_cache_length", 0) or 0
                )
            except (TypeError, ValueError):
                pass

    def update_session_config(
        self,
        *,
        media_type: int = 2,
        duplex_mode: bool,
        voice_clone_prompt: str,
        assistant_prompt: str,
        sampling: Optional[Dict[str, Any]] = None,
        return_logits: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """POST /v1/stream/update_session_config.

        The caller is responsible for pre-building both prompt strings;
        this method does NOT call ``_build_prompts_from_content``.
        ``sampling`` is whitelisted by the backend layer before being
        passed in here - this method just merges it into the body.
        """
        # Issue an explicit /break before the config swap to drain any
        # in-flight TTS/T2W work. Matches the legacy behavior implemented
        # inside ``_call_update_session_config``; doing it here lets the
        # backend layer stay free of HTTP plumbing.
        try:
            self.http.post(
                f"{self.base_url}/v1/stream/break",
                json={"reason": "session_config_change"},
                timeout=10.0,
            )
            time.sleep(0.1)
        except Exception:
            pass

        req_body: Dict[str, Any] = {
            "media_type": media_type,
            "duplex_mode": duplex_mode,
            "voice_clone_prompt": voice_clone_prompt,
            "assistant_prompt": assistant_prompt,
        }
        if sampling:
            req_body.update(sampling)
        if return_logits is not None:
            req_body["return_logits"] = bool(return_logits)

        resp = self.http.post(
            f"{self.base_url}/v1/stream/update_session_config",
            json=req_body,
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"update_session_config failed: {resp.text}")
        payload = resp.json()
        self._maybe_update_kv(payload)
        return payload

    def prefill(
        self,
        *,
        audio_path: str = "",
        img_path: str = "",
        text: str = "",
        cnt: int,
        max_slice_nums: int = -1,
    ) -> Dict[str, Any]:
        """POST /v1/stream/prefill. Raises on non-200.

        With the Phase 0 server.cpp patch in place, ``cnt == 0`` may carry
        all-empty content fields and still succeed: the C++ side treats
        that case as a system-prompt-init placeholder, rebuilding
        ``<|im_start|>system\\n...<|audio_start|>[ref_audio]<|audio_end|>
        ...<|im_end|>\\n`` from ``ctx_omni->ref_audio_path`` (omni.cpp
        line ~10096). For ``cnt > 0`` the server still requires at least
        one non-empty field.
        """
        req_body: Dict[str, Any] = {
            "audio_path_prefix": audio_path,
            "img_path_prefix": img_path,
            "cnt": cnt,
        }
        if max_slice_nums > 0:
            req_body["max_slice_nums"] = max_slice_nums
        if text:
            req_body["text"] = text

        resp = self.http.post(
            f"{self.base_url}/v1/stream/prefill",
            json=req_body,
            timeout=30.0,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"prefill(cnt={cnt}, audio={audio_path!r}, img={img_path!r}, "
                f"text_len={len(text)}) failed: {resp.text}"
            )
        try:
            payload = resp.json()
        except Exception:
            payload = {}
        self._maybe_update_kv(payload)
        return payload

    def decode(
        self,
        *,
        stream: bool = True,
        round_idx: Optional[int] = None,
        length_penalty: float = 1.1,
        max_new_tokens: Optional[int] = None,
        force_listen: Optional[bool] = None,
        logit_format: Optional[str] = None,
        logit_output_dir: Optional[str] = None,
        logit_filename: Optional[str] = None,
        logit_extra_metadata: Optional[Dict[str, Any]] = None,
        timeout: float = 600.0,
    ):
        """POST /v1/stream/decode. Returns the raw httpx Response so the
        caller can either parse SSE incrementally or read the body in one
        shot."""
        body: Dict[str, Any] = {
            "stream": bool(stream),
            "length_penalty": float(length_penalty),
        }
        if round_idx is not None:
            body["round_idx"] = int(round_idx)
        if max_new_tokens is not None and max_new_tokens > 0:
            body["max_new_tokens"] = int(max_new_tokens)
        if force_listen is not None:
            body["force_listen"] = bool(force_listen)
        if logit_format:
            body["logit_format"] = logit_format
        if logit_output_dir:
            body["logit_output_dir"] = logit_output_dir
        if logit_filename:
            body["logit_filename"] = logit_filename
        if logit_extra_metadata:
            body["logit_extra_metadata"] = logit_extra_metadata

        return self.http.post(
            f"{self.base_url}/v1/stream/decode",
            json=body,
            timeout=timeout,
        )

    def break_(self, reason: str = "manual") -> None:
        try:
            self.http.post(
                f"{self.base_url}/v1/stream/break",
                json={"reason": reason},
                timeout=10.0,
            )
        except Exception as e:
            logger.warning(f"break_ failed: {e}")


# ============================================================================
# Shared helpers for SimplexCppBackend / DuplexCppBackend
# ----------------------------------------------------------------------------
# These functions are pulled out of the legacy ``CppBackendWorker`` so that
# the new backend classes don't have to inherit from a 2000-line god-class
# just to get SSE parsing or temp-file utilities. The functions are small
# and stateless; the only stateful piece (``_OutputDirManager``) wraps the
# ``round_NNN/tts_wav`` directory layout that the C++ server writes to.
# ============================================================================

import json
import re
import base64

import numpy as np


# ---- prompt template construction --------------------------------------------
# Two-segment chat template the C++ omni runtime expects (see
# omni.cpp::stream_prefill index=0 branch). The Python side owns the strings;
# the C++ side concatenates them with the ref-audio embedding sandwiched
# between ``<|audio_start|>`` and ``<|audio_end|>``.
_SYSTEM_PROMPTS: Dict[tuple, Dict[str, str]] = {
    (True, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    (True, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    (False, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
                              "请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
                              "<|im_end|>\n<|im_start|>user\n",
    },
    (False, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|>Please assist users while maintaining this voice style. "
                              "Please answer the user's questions seriously and in a high quality. "
                              "Please chat with the user in a highly human-like and oral style. "
                              "You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
                              "<|im_end|>\n<|im_start|>user\n",
    },
}


def get_system_prompts(duplex: bool, lang: str = "zh") -> Dict[str, str]:
    return _SYSTEM_PROMPTS.get((duplex, lang), _SYSTEM_PROMPTS[(duplex, "zh")])


def build_prompts_from_content(
    system_content: Any,
    duplex: bool,
    lang: str = "zh",
) -> Dict[str, str]:
    """Build ``voice_clone_prompt`` + ``assistant_prompt`` from a request's
    ``system_content``.

    Input shapes:
      * ``list[{type:"text"|"audio", ...}]`` (audio split point separates
        the two halves of the prompt)
      * ``str`` (treated as one text segment, no audio split)
      * empty / None → falls back to the language default

    The chunk before any ``audio`` item becomes the system text; the
    chunk after becomes the assistant prefix. For non-duplex turns the
    template ends with ``<|im_start|>user\\n`` so the next prefill
    appends user content cleanly.
    """
    if isinstance(system_content, str):
        system_content = [{"type": "text", "text": system_content}] if system_content.strip() else []

    if not system_content or not isinstance(system_content, list):
        return get_system_prompts(duplex, lang)

    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    before_parts = []
    after_parts = []
    seen_audio = False
    for item in system_content:
        t = _get(item, "type")
        t_str = getattr(t, "value", t)
        if t_str == "audio":
            seen_audio = True
        elif t_str == "text":
            text = (_get(item, "text") or "").strip()
            if not text:
                continue
            (after_parts if seen_audio else before_parts).append(text)

    before = "\n".join(before_parts).strip()
    after = "\n".join(after_parts).strip()

    if not before and not after:
        return get_system_prompts(duplex, lang)

    voice_clone_prompt = f"<|im_start|>system\n{before}\n<|audio_start|>"
    if duplex:
        assistant_prompt = f"<|audio_end|>{after}<|im_end|>\n" if after else "<|audio_end|><|im_end|>\n"
    else:
        tail = f"{after}<|im_end|>\n<|im_start|>user\n" if after else "<|im_end|>\n<|im_start|>user\n"
        assistant_prompt = f"<|audio_end|>{tail}"

    return {
        "voice_clone_prompt": voice_clone_prompt,
        "assistant_prompt": assistant_prompt,
    }


# ---- sampling whitelist ------------------------------------------------------
# Top-level keys the C++ /v1/stream/update_session_config currently
# recognizes (see server.cpp ~6377-6398). New entries must be added on
# both sides plus ``omni_context``.
_CPP_SAMPLING_KEYS = (
    "listen_prob_scale",
    "force_listen_count",
    "max_new_speak_tokens_per_chunk",
    "tts_temperature",
)


def sampling_from_duplex_config(cfg: Any) -> Dict[str, Any]:
    """Whitelist DuplexConfig fields the C++ server accepts."""
    if cfg is None:
        return {}
    out: Dict[str, Any] = {}
    for key in _CPP_SAMPLING_KEYS:
        val = None
        if hasattr(cfg, key):
            val = getattr(cfg, key, None)
        elif isinstance(cfg, dict):
            val = cfg.get(key)
        if val is not None:
            out[key] = val
    return out


def sampling_from_generation(gen: Any) -> Dict[str, Any]:
    """Map ``GenerationConfig`` fields to the C++
    ``update_session_config`` request body.

    Two kinds of fields:

    1. **Top-level session-level**: ``tts_temperature`` → same name.
    2. **Nested ``llm_sampling`` object**: ``do_sample`` / ``temperature``
       / ``top_p`` / ``top_k`` / ``seed`` / ``repetition_penalty`` etc.
       The C++ side rebuilds ``ctx_omni->ctx_sampler`` from this dict on
       every config update (see the ``[Python 透传 - LLM 主 sampler
       per-request 配置]`` block in server.cpp).

       HF semantics:
         * ``do_sample=False`` → ``temp=0`` (llama.cpp greedy)
         * ``do_sample=True`` (or unset) + explicit ``temperature`` →
           pass through unchanged
         * Other fields are independent of ``do_sample``.

    ``max_new_tokens`` is intentionally NOT mapped here - per-request
    decode budget rides on the ``decode`` body instead, since
    ``ctx_omni->chat_max_new_tokens`` is per-turn rather than per-session.
    """
    if gen is None:
        return {}
    out: Dict[str, Any] = {}

    def _pick(name: str):
        if isinstance(gen, dict):
            return gen.get(name)
        return getattr(gen, name, None)

    tts_t = _pick("tts_temperature")
    if tts_t is not None:
        out["tts_temperature"] = tts_t

    llm: Dict[str, Any] = {}
    do_sample = _pick("do_sample")
    temperature = _pick("temperature")
    if do_sample is False:
        llm["temp"] = 0.0
    elif temperature is not None:
        llm["temp"] = float(temperature)

    top_p = _pick("top_p")
    if top_p is not None:
        llm["top_p"] = float(top_p)

    top_k = _pick("top_k")
    if top_k is not None and int(top_k) > 0:
        llm["top_k"] = int(top_k)

    seed = _pick("seed")
    if seed is not None:
        llm["seed"] = int(seed)

    rep_pen = _pick("repetition_penalty")
    if rep_pen is None:
        rep_pen = _pick("repeat_penalty")
    if rep_pen is not None:
        llm["penalty_repeat"] = float(rep_pen)

    rep_last_n = _pick("repetition_penalty_last_n")
    if rep_last_n is not None:
        llm["penalty_last_n"] = int(rep_last_n)

    if llm:
        out["llm_sampling"] = llm
    return out


def parse_sse_text(resp_text: str) -> str:
    """Concatenate every ``data: { content: ... }`` chunk from an SSE body.

    Mirrors the legacy ``CppBackendWorker._parse_sse_text``. Used by both
    streaming and non-streaming decode paths to recover the assistant
    text after the C++ server is done streaming."""
    pieces = []
    for line in resp_text.splitlines():
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
        content = event.get("content", "")
        if content:
            pieces.append(content)
    return "".join(pieces)


def extract_logits_from_sse(resp_text: str):
    """Pull the final ``event: logits`` payload out of an SSE body.

    The C++ server emits exactly one such event right before ``[DONE]``
    when ``return_logits=true`` was set on update_session_config. Returns
    ``None`` if no such event was found, otherwise a ``LogitsPayload``
    instance with ``token_ids_b64+logits_b64`` (inline) or ``file``
    populated."""
    from core.schemas.logits import LogitsPayload

    last: Optional[Dict[str, Any]] = None
    for line in resp_text.splitlines():
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
        if isinstance(event, dict) and event.get("type") == "logits":
            last = event
    if last is None:
        return None
    try:
        return LogitsPayload(
            success=bool(last.get("success", True)),
            error=last.get("error"),
            n_tokens=int(last["n_tokens"]),
            n_prefill_tokens=int(last["n_prefill_tokens"]),
            vocab_size=int(last["vocab_size"]),
            dtype=str(last.get("dtype", "bf16")),
            token_ids_b64=last.get("token_ids_b64"),
            logits_b64=last.get("logits_b64"),
            file=last.get("file"),
            sha256=last.get("sha256"),
        )
    except Exception as e:
        logger.warning(
            f"failed to parse logits event: {e}; raw keys={list(last.keys())}"
        )
        return None


_AUDIO_INPUT_SR = 16000
_AUDIO_OUTPUT_SR = 24000


def save_audio_to_temp(temp_dir: str, audio_np: np.ndarray, prefix: str) -> str:
    """Dump a float32 mono waveform to ``{temp_dir}/{prefix}.wav`` (16 kHz
    PCM_16). Pads to ``MIN_SAMPLES`` so the C++ ASR doesn't choke on tiny
    fragments."""
    import soundfile as sf

    MIN_SAMPLES = 1600
    if len(audio_np) < MIN_SAMPLES:
        audio_np = np.pad(
            audio_np, (0, MIN_SAMPLES - len(audio_np)), mode="constant"
        )

    path = os.path.join(temp_dir, f"{prefix}.wav")
    audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
    sf.write(path, audio_np, _AUDIO_INPUT_SR, format="WAV", subtype="PCM_16")
    return path


def save_pil_image_to_temp(temp_dir: str, pil_image, prefix: str) -> str:
    path = os.path.join(temp_dir, f"{prefix}.png")
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    pil_image.save(path, format="PNG")
    return path


def cleanup_temp_files(*paths: str) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


class _OutputDirManager:
    """Tracks the C++ output directory layout and incrementally collects
    TTS WAVs as the server writes them.

    Layout:
        {output_dir}/
            round_000/tts_wav/wav_*.wav + generation_done.flag
            round_001/...
            tts_wav/wav_*.wav            (duplex-mode flat layout)

    The legacy ``CppBackendWorker`` exposed three almost-identical
    collectors (``_collect_wav_output``, ``_collect_wav_output_nowait``,
    ``_iter_wav_chunks_incremental``) each with its own subtle wait
    semantics. We keep all three as methods here; SimplexCppBackend uses
    ``collect_blocking`` for ``POST /chat`` non-streaming, ``iter_chunks``
    for ``WS /ws/chat`` streaming, and DuplexCppBackend uses
    ``collect_nowait`` for per-chunk peeks.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._sent: set = set()

    def reset(self) -> None:
        """Wipe ``output_dir`` (round_* subdirs + tts_wav) before a new
        session. Skips files in flight so it's safe to call mid-spawn."""
        if not os.path.exists(self.output_dir):
            return
        try:
            for entry in list(os.listdir(self.output_dir)):
                full = os.path.join(self.output_dir, entry)
                if not (
                    entry.startswith("round_") or entry == "tts_wav"
                ):
                    continue
                if os.path.isdir(full):
                    try:
                        import shutil
                        shutil.rmtree(full, ignore_errors=True)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"_OutputDirManager.reset: {e}")
        self._sent = set()

    def find_latest_round_dir(self) -> Optional[str]:
        if not os.path.exists(self.output_dir):
            return None
        rounds = sorted(
            [
                d for d in os.listdir(self.output_dir)
                if d.startswith("round_")
                and os.path.isdir(os.path.join(self.output_dir, d))
            ],
            reverse=True,
        )
        if rounds:
            return os.path.join(self.output_dir, rounds[0])
        return None

    def collect_blocking(
        self, sse_text: str = "", timeout_per_round: float = 120.0
    ) -> tuple:
        """Wait up to ``timeout_per_round`` seconds for
        ``generation_done.flag`` then concatenate every ``wav_*.wav``
        under the latest ``round_NNN/tts_wav`` (or root ``tts_wav`` if
        the model used the flat duplex layout). Returns
        ``(audio_b64_float32, combined_text)`` where ``combined_text``
        is ``sse_text`` (this overload doesn't read .txt sidecars to
        keep behavior aligned with simplex chat)."""
        import soundfile as sf

        round_dir = self.find_latest_round_dir()
        if not round_dir:
            direct_tts = os.path.join(self.output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                tts_wav_dir = direct_tts
            else:
                return None, sse_text
        else:
            tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            return None, sse_text

        flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
        t0 = time.time()
        while time.time() - t0 < timeout_per_round:
            if os.path.exists(flag_path):
                break
            time.sleep(0.1)

        wav_files = sorted(
            [
                f for f in os.listdir(tts_wav_dir)
                if f.startswith("wav_") and f.endswith(".wav")
            ],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1))
            if re.search(r"wav_(\d+)", f) else 0,
        )
        if not wav_files:
            return None, sse_text

        all_audio = []
        for wf in wav_files:
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, _sr = sf.read(wp)
                if len(data) > 0:
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    all_audio.append(data)
            except Exception:
                pass
        if not all_audio:
            return None, sse_text
        combined = np.concatenate(all_audio)
        audio_b64 = base64.b64encode(
            combined.astype(np.float32).tobytes()
        ).decode("utf-8")
        return audio_b64, sse_text

    def collect_nowait(self, sse_text: str = "") -> tuple:
        """Non-blocking peek: returns only WAVs that haven't been seen
        yet (tracked in ``self._sent``). Used by duplex per-chunk
        polling."""
        import soundfile as sf

        direct_tts = os.path.join(self.output_dir, "tts_wav")
        if os.path.isdir(direct_tts) and any(
            f.startswith("wav_") and f.endswith(".wav")
            for f in os.listdir(direct_tts)
        ):
            tts_wav_dir = direct_tts
        else:
            round_dir = self.find_latest_round_dir()
            if not round_dir:
                if os.path.isdir(direct_tts):
                    tts_wav_dir = direct_tts
                else:
                    return None, sse_text
            else:
                tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            return None, sse_text

        wav_files = sorted(
            [
                f for f in os.listdir(tts_wav_dir)
                if f.startswith("wav_") and f.endswith(".wav")
            ],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1))
            if re.search(r"wav_(\d+)", f) else 0,
        )
        new_files = [f for f in wav_files if f not in self._sent]
        if not new_files:
            return None, sse_text

        all_audio = []
        for wf in new_files:
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, _sr = sf.read(wp)
                if len(data) > 0:
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    all_audio.append(data)
                    self._sent.add(wf)
            except Exception:
                pass

        if not all_audio:
            return None, sse_text
        combined = np.concatenate(all_audio)
        audio_b64 = base64.b64encode(
            combined.astype(np.float32).tobytes()
        ).decode("utf-8")
        return audio_b64, sse_text

    def iter_chunks(self, timeout: float = 120.0):
        """Generator: yield each new WAV's base64 payload as soon as it
        appears in the latest ``round_NNN/tts_wav``, until
        ``generation_done.flag`` shows up (then drain remaining files)."""
        import soundfile as sf

        round_dir = None
        t_wait = time.time()
        while time.time() - t_wait < 15.0:
            round_dir = self.find_latest_round_dir()
            if round_dir:
                break
            direct_tts = os.path.join(self.output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self.output_dir
                break
            time.sleep(0.2)
        if not round_dir:
            logger.warning(
                "_OutputDirManager.iter_chunks: no round/tts_wav dir after 15s"
            )
            return

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
        sent: set = set()
        t0 = time.time()

        while time.time() - t0 < timeout:
            if not os.path.exists(tts_wav_dir):
                time.sleep(0.1)
                continue

            current_files = sorted(
                [
                    f for f in os.listdir(tts_wav_dir)
                    if f.startswith("wav_") and f.endswith(".wav")
                ],
                key=lambda f: int(re.search(r"wav_(\d+)", f).group(1))
                if re.search(r"wav_(\d+)", f) else 0,
            )

            for wf in current_files:
                if wf in sent:
                    continue
                wp = os.path.join(tts_wav_dir, wf)
                try:
                    data, _sr = sf.read(wp)
                    if len(data) == 0:
                        continue
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    yield base64.b64encode(data.tobytes()).decode("utf-8")
                    sent.add(wf)
                except Exception as e:
                    logger.warning(f"Failed to read {wf}: {e}")

            if os.path.exists(flag_path):
                final_files = sorted(
                    [
                        f for f in os.listdir(tts_wav_dir)
                        if f.startswith("wav_") and f.endswith(".wav")
                    ],
                    key=lambda f: int(re.search(r"wav_(\d+)", f).group(1))
                    if re.search(r"wav_(\d+)", f) else 0,
                )
                for wf in final_files:
                    if wf in sent:
                        continue
                    wp = os.path.join(tts_wav_dir, wf)
                    try:
                        data, _sr = sf.read(wp)
                        if len(data) > 0:
                            if data.dtype != np.float32:
                                data = data.astype(np.float32)
                            yield base64.b64encode(
                                data.tobytes()
                            ).decode("utf-8")
                            sent.add(wf)
                    except Exception:
                        pass
                return

            time.sleep(0.15)

        logger.warning(
            f"_OutputDirManager.iter_chunks timed out after {timeout}s"
        )
