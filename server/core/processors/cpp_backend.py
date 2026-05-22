"""C++ llama.cpp-omni 推理后端适配层

通过 HTTP 调用 C++ llama-server 的 omni 接口，实现与 MiniCPMOWorker 相同的方法签名，
作为 PyTorch 后端的 drop-in 替换。

生命周期映射：
    服务启动   → omni_init（加载 APM/VPM/TTS/Token2Wav，复用 LLM）
    新会话     → update_session_config（清空 KV cache，重新 prefill system prompt）
    每个 chunk → /v1/stream/prefill + /v1/stream/decode
    打断       → /v1/stream/break
    会话结束   → 清理输出目录
"""

import os
import re
import sys
import io
import gc
import json
import time
import base64
import shutil
import signal
import logging
import tempfile
import platform
import threading
import subprocess
from typing import Optional, List, Dict, Any, Iterator
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger("cpp_backend")

_AUDIO_INPUT_SR = 16000
_AUDIO_OUTPUT_SR = 24000

# System prompt 模板 — 来自 modeling_minicpmo.py audio_assistant 模式
# key: (duplex, lang) → (voice_clone_prompt, assistant_prompt)
_SYSTEM_PROMPTS: Dict[tuple, Dict[str, str]] = {
    # 双工模式 — 语言无关，固定英文 prompt
    (True, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    (True, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nStreaming Duplex Conversation! You are a helpful assistant.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|><|im_end|>\n",
    },
    # 非双工 — 中文
    (False, "zh"): {
        "voice_clone_prompt": "<|im_start|>system\n模仿音频样本的音色并生成新的内容。\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|>你的任务是用这种声音模式来当一个助手。请认真、高质量地回复用户的问题。"
                              "请用高自然度的方式和用户聊天。你是由面壁智能开发的人工智能助手：面壁小钢炮。"
                              "<|im_end|>\n<|im_start|>user\n",
    },
    # 非双工 — 英文
    (False, "en"): {
        "voice_clone_prompt": "<|im_start|>system\nClone the voice in the provided audio prompt.\n<|audio_start|>",
        "assistant_prompt":   "<|audio_end|>Please assist users while maintaining this voice style. "
                              "Please answer the user's questions seriously and in a high quality. "
                              "Please chat with the user in a highly human-like and oral style. "
                              "You are a helpful assistant developed by ModelBest: MiniCPM-Omni."
                              "<|im_end|>\n<|im_start|>user\n",
    },
}


def _get_system_prompts(duplex: bool, lang: str = "zh") -> Dict[str, str]:
    """根据模式和语言返回 voice_clone_prompt / assistant_prompt"""
    return _SYSTEM_PROMPTS.get((duplex, lang), _SYSTEM_PROMPTS[(duplex, "zh")])


def _build_prompts_from_content(
    system_content: Any,
    duplex: bool,
    lang: str = "zh",
) -> Dict[str, str]:
    """根据前端传入的 system_content 动态构造 C++ 需要的两段式 prompt。

    输入支持：
      - list: [{type:"text", text:...}, {type:"audio", data:...}, {type:"text", text:...}]
      - str: 纯文本 system prompt
      - None / 空: 返回硬编码默认模板

    输出结构：
      - voice_clone_prompt: "<|im_start|>system\\n{before}\\n<|audio_start|>"
      - assistant_prompt:  "<|audio_end|>{after}<|im_end|>\\n" (duplex)
                          "<|audio_end|>{after}<|im_end|>\\n<|im_start|>user\\n" (非 duplex)

    其中 before = audio 前所有 text 段拼接，after = audio 后所有 text 段拼接。
    若没有 audio 段，全部 text 归入 before。
    """
    # 字符串直接走单 text 分支
    if isinstance(system_content, str):
        system_content = [{"type": "text", "text": system_content}] if system_content.strip() else []

    if not system_content or not isinstance(system_content, list):
        return _get_system_prompts(duplex, lang)

    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    before_parts: List[str] = []
    after_parts: List[str] = []
    seen_audio = False
    for item in system_content:
        t = _get(item, "type")
        # pydantic 枚举可能是 ContentType.TEXT 形式
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
        return _get_system_prompts(duplex, lang)

    # 没有任何 text → 回退默认
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


# C++ /v1/stream/update_session_config 当前能识别的 sampling 字段。
# 与 omni_context 中的字段一一对应；新增需同步 server.cpp + omni.h。
_CPP_SAMPLING_KEYS = (
    "listen_prob_scale",
    "force_listen_count",
    "max_new_speak_tokens_per_chunk",
    "tts_temperature",
)


def _sampling_from_duplex_config(cfg: Any) -> Dict[str, Any]:
    """从 DuplexConfig（pydantic 模型 / dict / None）抽出 C++ 能用的 sampling 字段。"""
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


def _sampling_from_generation(gen: Any) -> Dict[str, Any]:
    """从 GenerationConfig（chat / half-duplex 用）映射到 C++
    ``/v1/stream/update_session_config`` 接受的字段。

    映射两类字段：

    1. **顶层 session 级 sticky**：``tts_temperature`` → 同名顶层字段，影响
       ``ctx_tts_sampler``。

    2. **嵌套 ``llm_sampling`` 对象**：do_sample / temperature / top_p / top_k /
       seed / repetition_penalty 这些 LLM 主链路字段，由 server.cpp 的
       ``handle_stream_update_session_config_impl`` 解析后**重建**
       ``ctx_omni->ctx_sampler``（见 server.cpp 的 ``[Python 透传 - LLM 主 sampler
       per-request 配置]`` 块）。语义对齐 HuggingFace generation：

       - ``do_sample=False`` → 透传 ``temp=0``（llama.cpp ``temp <= 0 = greedy``）
       - ``do_sample=True``  → 透传请求体上的 ``temperature`` 原值
       - 其它字段（top_p/top_k/seed）不依赖 do_sample，按字段直传

    历史上这里曾把整段 GenerationConfig 丢掉（见 commit 9bc3964 引入的
    ``OMNI_LLM_SAMPLE_TEMP`` 环境变量），导致 do_sample / temperature / seed
    在 chat 路径根本没有效果。本函数现在恢复语义并避免再依赖那个环境变量。

    ``max_new_tokens`` **不**进这里，而是在每轮 ``/v1/stream/decode`` 请求体上
    单独透传到 ``ctx_omni->chat_max_new_tokens`` —— 单轮上限是"按请求"语义，
    不是"按 session"语义；duplex 还要走 ``max_new_speak_tokens_per_chunk``。
    """
    if gen is None:
        return {}
    out: Dict[str, Any] = {}

    def _pick(name: str):
        if isinstance(gen, dict):
            return gen.get(name)
        return getattr(gen, name, None)

    # ---- TTS sampler (顶层) ----
    tts_t = _pick("tts_temperature")
    if tts_t is not None:
        out["tts_temperature"] = tts_t

    # ---- LLM 主 sampler (嵌套 llm_sampling) ----
    llm: Dict[str, Any] = {}
    do_sample = _pick("do_sample")
    temperature = _pick("temperature")
    if do_sample is False:
        # HF 语义：do_sample=False ⇒ greedy。映射到 llama.cpp 的 temp<=0=greedy。
        llm["temp"] = 0.0
    elif temperature is not None:
        # do_sample=True 或未指定，且显式给了 temperature → 按字段直传。
        # 注意 GenerationConfig.temperature 默认 0.7，不会触发 sampler 重建
        # （server.cpp 端做了"值变化才 rebuild"判断）。
        llm["temp"] = float(temperature)

    top_p = _pick("top_p")
    if top_p is not None:
        llm["top_p"] = float(top_p)

    top_k = _pick("top_k")
    if top_k is not None and int(top_k) > 0:
        # GenerationConfig.top_k=0 在 HF 语义里是"禁用"；在 llama.cpp 里
        # ``top_k <= 0`` 表示"用 vocab size"，行为不同。这里按 HF 语义跳过。
        llm["top_k"] = int(top_k)

    seed = _pick("seed")
    if seed is not None:
        # int -> uint32 转换由 C++ 端做（json::is_number_integer）
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


class CppBackendWorker:
    """C++ llama-server 推理后端

    实现与 MiniCPMOWorker 相同的方法签名，内部通过 HTTP 调用 C++ 服务。
    """

    def __init__(
        self,
        llamacpp_root: str,
        model_dir: str,
        gpu_id: int = 0,
        ref_audio_path: Optional[str] = None,
        duplex_pause_timeout: float = 60.0,
        llm_model: str = "",
        cpp_server_port: Optional[int] = None,
        ctx_size: int = 32768,
        n_gpu_layers: int = 99,
        use_tts: bool = True,
        worker_idx: int = 0,
        **kwargs,
    ):
        self.llamacpp_root = llamacpp_root
        self.model_dir = model_dir
        self.gpu_id = gpu_id
        self.ref_audio_path = ref_audio_path
        self.duplex_pause_timeout = duplex_pause_timeout
        self.llm_model = llm_model or self._auto_detect_llm_model(model_dir)
        self.ctx_size = ctx_size
        self.n_gpu_layers = n_gpu_layers
        # When False, omni_init is called with use_tts=False so the C++ side
        # skips loading TTS/Token2Wav weights and never spawns the TTS / T2W
        # threads. Used for RL-training rollouts that only care about the
        # LLM token stream + logits (no audio synthesis).
        self.use_tts = bool(use_tts)
        # 0-based index of this worker in the batch_server pool, injected by
        # ``worker.py``'s ``args.worker_index``. **Only** used to mint
        # collision-free logits filenames in ``make_logits_filename`` —
        # multiple workers under the same gateway share one date-bucket dir,
        # so without an idx in the filename two workers would race on the
        # same name and overwrite each other's ``.safetensors``. Kept as a
        # plain int (not parsed from gpu_id) because ``CUDA_VISIBLE_DEVICES``
        # remaps gpu_id to local 0 on every worker.
        self.worker_idx = int(worker_idx)

        from worker import WorkerState, WorkerStatus
        self.state = WorkerState()
        self.processor = None  # compatibility — used for kv_cache_length etc.

        self._cpp_server_port = cpp_server_port or (19060 + gpu_id)
        self._cpp_server_url = f"http://127.0.0.1:{self._cpp_server_port}"
        self._cpp_process: Optional[subprocess.Popen] = None
        self._http_client = None  # httpx.Client (sync)
        self._temp_dir = tempfile.mkdtemp(prefix="cpp_backend_")
        self._output_dir = os.path.join(llamacpp_root, f"tools/omni/output_{self._cpp_server_port}")
        self._last_duplex_mode: Optional[bool] = None
        self._last_media_type: int = 2
        self._last_lang: str = "zh"
        self._duplex_length_penalty: float = 1.1

        self._duplex_chunk_counter: int = 0
        self._current_session_id: Optional[str] = None
        self._round_number: int = 0
        self._sent_wav_files: set = set()
        self._last_kv_cache_length: int = 0

        # New layered backend infrastructure (Phase 4 of the simplex/duplex
        # refactor). ``_proc`` and ``_http`` are bound during ``load_model``;
        # ``simplex`` and ``duplex`` are the public entry points the
        # worker.py endpoints now go through. Legacy method stubs above
        # remain only as thin proxies for the WS half-duplex / chat /
        # duplex paths until those callers migrate.
        from .cpp_session import _CppServerProc, _StreamHttpClient
        self._proc = _CppServerProc(
            llamacpp_root=llamacpp_root,
            model_dir=model_dir,
            gpu_id=gpu_id,
            port=self._cpp_server_port,
            ctx_size=ctx_size,
            n_gpu_layers=n_gpu_layers,
            use_tts=self.use_tts,
            llm_model=self.llm_model,
            ref_audio_path=ref_audio_path,
            output_dir=self._output_dir,
        )
        self._http: Optional[_StreamHttpClient] = None
        self.simplex = None  # filled in load_model
        self.duplex = None   # filled in load_model

    # ================================================================
    # Model loading (maps to omni_init)
    # ================================================================

    def load_model(self) -> None:
        """Boot the C++ llama-server and arm the omni context.

        Wiring:
          1. ``_CppServerProc.start()`` spawns llama-server, polls /health.
          2. ``call_omni_init`` loads APM/VPM/TTS/Token2Wav inside C++.
             Default duplex mode + the canonical English duplex prompt;
             per-request prompts override this on every turn.
          3. Mirror ``_proc``'s subprocess + httpx onto the legacy
             ``self._cpp_process`` / ``self._http_client`` /
             ``self._cpp_server_url`` fields so the remaining legacy
             helpers (``_call_*``, ``_collect_wav_output``, etc.) still
             work during the migration window.
          4. Build ``self.simplex`` and ``self.duplex`` - the new
             entry points worker.py now drives.
        """
        from worker import WorkerStatus
        from .cpp_session import (
            _StreamHttpClient,
            get_system_prompts,
        )
        from .simplex_backend import SimplexCppBackend
        from .duplex_backend import DuplexCppBackend

        self.state.status = WorkerStatus.LOADING
        logger.info(f"[GPU {self.gpu_id}] Starting C++ llama-server...")

        self._proc.start()

        boot_prompts = get_system_prompts(duplex=True, lang=self._last_lang)
        self._proc.call_omni_init(
            media_type=2,
            duplex_mode=True,
            voice_clone_prompt=boot_prompts["voice_clone_prompt"],
            assistant_prompt=boot_prompts["assistant_prompt"],
        )
        self._last_duplex_mode = True

        # Legacy field mirrors so the old _call_* methods keep working.
        self._http_client = self._proc.http_client
        self._cpp_process = self._proc._proc
        self._cpp_server_url = self._proc.url

        self._http = _StreamHttpClient(self._proc.url, self._proc.http_client)
        self.simplex = SimplexCppBackend(
            proc=self._proc,
            http=self._http,
            ref_audio_path=self.ref_audio_path,
            worker_idx=self.worker_idx,
            use_tts=self.use_tts,
            output_dir=self._output_dir,
            temp_dir=self._temp_dir,
        )
        self.duplex = DuplexCppBackend(
            proc=self._proc,
            http=self._http,
            ref_audio_path=self.ref_audio_path,
            worker_idx=self.worker_idx,
            use_tts=self.use_tts,
            output_dir=self._output_dir,
            temp_dir=self._temp_dir,
        )

        self.state.status = WorkerStatus.IDLE
        logger.info(f"[GPU {self.gpu_id}] C++ backend ready")

    @property
    def kv_cache_length(self) -> int:
        return int(self._last_kv_cache_length)

    def _maybe_update_kv_cache_length(self, payload: Any) -> None:
        if isinstance(payload, dict) and "kv_cache_length" in payload:
            try:
                self._last_kv_cache_length = int(payload.get("kv_cache_length", 0) or 0)
            except (TypeError, ValueError):
                logger.debug("invalid kv_cache_length payload: %r", payload.get("kv_cache_length"))

    # ================================================================
    # Duplex
    # ================================================================

    def duplex_prepare(
        self,
        system_prompt_text: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        media_type: int = 2,
        lang: Optional[str] = None,
        system_content: Any = None,
        length_penalty: float = 1.1,
        sampling: Optional[Dict[str, Any]] = None,
        return_logits: bool = False,
    ) -> str:
        """Open a fresh duplex session via ``DuplexCppBackend.session_begin``.

        Now a thin shim around the new backend - all of the legacy
        ``update_session_config`` short-circuiting and BUG-FIX commentary
        is gone. The ``ref_audio_path`` request override flows directly
        into ``session_begin(ref_audio_override=...)`` (D2/D5), the
        sampling dict is forwarded as-is (D9), and ``return_logits``
        switches the per-chunk inline capture on.

        ``prompt_wav_path`` is accepted for signature compatibility with
        the legacy worker.py call site but is not currently used: the
        omni model uses the same ref-audio for both ASR conditioning and
        TTS voice cloning, and that ref is set via ``ref_audio_path``.
        """
        self._duplex_length_penalty = float(length_penalty)
        self._sent_wav_files = set()
        self._round_number = 0

        self.duplex.session_begin(
            system_content=system_content,
            sampling=sampling,
            lang=lang or self._last_lang or "zh",
            media_type=media_type,
            ref_audio_override=ref_audio_path,
            return_logits=return_logits,
            length_penalty=length_penalty,
        )

        self._last_duplex_mode = True
        self._last_media_type = media_type
        if lang:
            self._last_lang = lang
        # Pending state for the legacy 2-phase prefill -> generate API.
        self._pending_audio: Optional[np.ndarray] = None
        self._pending_frames: Optional[list] = None
        self._pending_max_slice_nums: int = 1

        os.makedirs(os.path.join(self._output_dir, "tts_wav"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "tts_txt"), exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "llm_debug"), exist_ok=True)
        return system_prompt_text or "Streaming Duplex Conversation."

    def duplex_prefill(
        self,
        audio_waveform: Optional[np.ndarray] = None,
        frame_list: Optional[list] = None,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        """Stage one chunk's worth of input for the next ``duplex_generate``.

        The legacy API split each duplex tick into ``prefill`` + ``decode``;
        the new ``DuplexCppBackend.push_frame`` does both in one HTTP
        round-trip. We keep the split worker.py call shape by stashing
        the inputs here and consuming them on the very next
        ``duplex_generate`` call. This preserves the WS protocol exactly:
        worker.py still sees one ``DuplexGenerateResult`` per WS chunk."""
        self._pending_audio = audio_waveform
        self._pending_frames = list(frame_list) if frame_list else None
        self._pending_max_slice_nums = int(max_slice_nums) if max_slice_nums else 1
        n_vision_images = len(frame_list) if frame_list else 0
        return {"n_vision_images": n_vision_images}

    def duplex_generate(self, force_listen: bool = False) -> "DuplexGenerateResult":
        """Run one duplex tick: push the staged frame and read back the
        listen/speak/text/audio/logits decision."""
        from core.schemas.duplex import DuplexGenerateResult

        result = self.duplex.push_frame(
            audio_chunk=self._pending_audio,
            vision_frames=self._pending_frames,
            force_listen=force_listen,
            max_slice_nums=self._pending_max_slice_nums,
        )
        # Reset pending state so a forgotten duplex_prefill doesn't quietly
        # replay the previous chunk on the next generate.
        self._pending_audio = None
        self._pending_frames = None
        # Mirror the kv_cache_length onto the legacy field so any code
        # reading ``self._last_kv_cache_length`` still sees fresh values.
        kv = result.get("kv_cache_length")
        if isinstance(kv, int):
            self._last_kv_cache_length = kv
        self._duplex_chunk_counter = result.get("current_time", 0) + 1

        return DuplexGenerateResult(
            is_listen=bool(result["is_listen"]),
            text=result.get("text", "") or "",
            audio_data=result.get("audio_data"),
            end_of_turn=bool(result["end_of_turn"]),
            current_time=int(result.get("current_time", 0)),
            cost_llm_ms=result.get("cost_llm_ms"),
            cost_tts_prep_ms=result.get("cost_tts_prep_ms"),
            cost_tts_ms=result.get("cost_tts_ms"),
            cost_token2wav_ms=result.get("cost_token2wav_ms"),
            cost_all_ms=result.get("cost_all_ms"),
            n_tokens=result.get("n_tokens"),
            n_tts_tokens=result.get("n_tts_tokens"),
            logits=result.get("logits"),
        )

    def duplex_finalize(self) -> None:
        """C++ manages duplex KV state internally; no-op kept for the
        legacy worker.py call site that runs it after each chunk."""
        pass

    def duplex_stop(self) -> None:
        """Drain pending TTS/T2W work via ``/v1/stream/break``.

        Records ``_last_break_time`` so ``full_reinit`` later knows it
        should wait for ``generation_done.flag`` before tearing down.
        """
        self._last_break_time = time.time()
        try:
            self.duplex.break_now(reason="duplex_stop")
        except Exception as e:
            logger.warning(f"duplex_stop break call failed: {e}")

    def duplex_cleanup(self) -> None:
        """Wipe ``output_dir`` (round_NNN/tts_wav) and prep for next session.

        The legacy version also re-issued ``update_session_config`` here
        which double-cleared the KV cache; the new ``DuplexCppBackend``
        always re-arms KV at ``session_begin``, so that step is dead
        weight. We only reset the output dir + sent-WAV bookkeeping.
        """
        try:
            self.duplex._dir_mgr.reset()
        except Exception as e:
            logger.warning(f"duplex_cleanup dir reset failed: {e}")
        self._sent_wav_files = set()
        gc.collect()

    def is_cpp_healthy(self) -> bool:
        """Check whether the underlying llama-server subprocess is alive.

        Avoids hitting ``/health`` on purpose: a watchdog that polls
        HTTP would compete with an in-flight prefill/decode request and
        the server occasionally resets the responder under that contention,
        producing spurious "unhealthy" verdicts. Subprocess liveness via
        ``proc.poll()`` is enough.
        """
        proc = self._cpp_process
        if proc is None or proc.poll() is not None:
            return False
        return True

    def full_reinit(self) -> None:
        """Hard restart the C++ subprocess; matches the legacy semantics.

        Used by:
          * ``worker.py`` CppWatchdog when the subprocess looks unhealthy
          * the WS half-duplex / duplex finally blocks for a clean
            next-session state

        Implementation now delegates to ``_CppServerProc.full_restart``
        and simply resyncs the legacy mirror fields so the remaining
        ``_call_*`` helpers keep functioning. The pre-restart
        T2W-completion-flag wait remains useful because a duplex/turn
        teardown may still have async TTS/T2W work in flight.
        """
        self._round_number = 0
        self._last_duplex_mode = None
        self._last_media_type = 2
        t_break = getattr(self, '_last_break_time', 0.0)
        if t_break > 0.0:
            flag_paths = [
                os.path.join(self._output_dir, "generation_done.flag"),
                os.path.join(self._output_dir, "tts_wav", "generation_done.flag"),
            ]

            def _flag_exists() -> bool:
                latest_round = self._find_latest_round_dir()
                if latest_round:
                    latest_flag = os.path.join(latest_round, "tts_wav", "generation_done.flag")
                    if latest_flag not in flag_paths:
                        flag_paths.append(latest_flag)
                return any(os.path.exists(p) for p in flag_paths)

            if not _flag_exists():
                for _ in range(20):
                    time.sleep(0.5)
                    if _flag_exists():
                        break
                else:
                    logger.warning(
                        "full_reinit: T2W completion flag not seen within 10s, proceeding; "
                        f"checked_paths={flag_paths}"
                    )
        try:
            self._proc.full_restart()
            # Keep the legacy mirrors fresh so any code path still using
            # ``self._cpp_process`` / ``self._http_client`` sees the new
            # subprocess + client.
            self._cpp_process = self._proc._proc
            self._http_client = self._proc.http_client
            self._last_duplex_mode = True
            self._last_media_type = 2
            logger.info("full_reinit: omni context re-initialized successfully")
        except Exception as e:
            logger.error(f"full_reinit failed: {e}", exc_info=True)
            raise

    # ================================================================
    # Half-Duplex
    # ================================================================

    def half_duplex_prefill(self, request) -> str:
        """Half-Duplex prefill: begin a fresh simplex turn for this VAD
        segment and push every user content item from ``request.messages``
        into the C++ context.

        Each VAD turn is its own clean simplex turn (the previous turn's
        KV cache is wiped by ``begin_turn``'s update_session_config),
        so we never accumulate state across speech segments. The cached
        config snapshot from ``reset_half_duplex_session`` is consulted
        for system content / sampling / language."""
        from core.processors.base import MiniCPMOProcessorMixin

        snapshot = getattr(self, "_hdx_config", None) or {}
        system_content = snapshot.get("system_content")
        # Fall back to the first system message in this request if the
        # caller never ran ``reset_half_duplex_session``.
        if system_content is None:
            for m in request.messages:
                role = getattr(m, "role", None)
                role_str = role.value if hasattr(role, "value") else role
                if role_str == "system":
                    system_content = getattr(m, "content", None)
                    break

        self.simplex.begin_turn(
            system_content=system_content,
            sampling=snapshot.get("sampling"),
            lang=snapshot.get("lang") or self._last_lang or "zh",
            return_logits=False,
        )

        mixin = MiniCPMOProcessorMixin()
        for msg in request.messages:
            role = getattr(msg, "role", None)
            role_str = role.value if hasattr(role, "value") else role
            if role_str == "system":
                continue
            for item in mixin._convert_content_to_model_format(msg.content):
                if isinstance(item, np.ndarray):
                    self.simplex.push_audio(item)
                elif isinstance(item, str):
                    self.simplex.push_text(item)
                elif hasattr(item, "size"):
                    self.simplex.push_image(item)
        return "prefilled"

    def half_duplex_init_tts(self, ref_audio_data: Optional[np.ndarray] = None) -> None:
        """TTS is already initialized inside ``omni_init``; this is a
        no-op kept only for backward-compatible call sites."""
        pass

    def _parse_sse_text(self, resp_text: str) -> str:
        """从 C++ decode SSE 响应中提取所有文本内容"""
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
            self._maybe_update_kv_cache_length(event)
            content = event.get("content", "")
            if content:
                pieces.append(content)
        return "".join(pieces)

    def _extract_logits_from_sse(self, resp_text: str):
        """Scan a /v1/stream/decode SSE body and pull out the final
        ``event: logits`` payload (if any) into a LogitsPayload pydantic obj.

        The C++ server emits exactly one such event right before ``[DONE]``
        when ``return_logits=true`` was set on update_session_config. The
        event shape matches LogitsPayload fields (n_tokens, n_prefill_tokens,
        vocab_size, dtype, plus either token_ids_b64+logits_b64 or file).
        """
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
            logger.warning(f"failed to parse logits event: {e}; raw keys={list(last.keys())}")
            return None

    def half_duplex_generate(
        self,
        session_id: str,
        generate_audio: bool = True,
        max_new_tokens: int = 256,
        length_penalty: float = 1.1,
    ) -> "Iterator[StreamingChunk]":
        """Half-Duplex / chat WS streaming decode.

        Translates the SimplexCppBackend's event stream into the legacy
        ``StreamingChunk`` shape that worker.py's WS endpoints already
        send to clients, so the WS protocol is unchanged. The legacy
        body (~140 lines of hand-rolled SSE parsing + WAV polling) is
        no longer needed; ``decode_streaming`` does both."""
        from core.schemas.streaming import StreamingChunk

        t0 = time.perf_counter()
        chunk_idx = 0
        last_text = ""

        try:
            for evt in self.simplex.decode_streaming(
                length_penalty=float(length_penalty),
                max_new_tokens=int(max_new_tokens) if max_new_tokens else None,
                generate_audio=bool(generate_audio),
            ):
                t_now = round((time.perf_counter() - t0) * 1000, 1)
                if evt["type"] == "text":
                    last_text = evt["delta"]
                    yield StreamingChunk(
                        chunk_index=chunk_idx,
                        text_delta=evt["delta"],
                        is_final=False,
                        duration_ms=t_now,
                    )
                    chunk_idx += 1
                elif evt["type"] == "audio":
                    yield StreamingChunk(
                        chunk_index=chunk_idx,
                        audio_data=evt["data"],
                        is_final=False,
                        duration_ms=t_now,
                    )
                    chunk_idx += 1
                elif evt["type"] == "done":
                    if chunk_idx == 0 and evt.get("text"):
                        yield StreamingChunk(
                            chunk_index=0,
                            text_delta=evt["text"],
                            is_final=True,
                            duration_ms=t_now,
                        )
                    else:
                        yield StreamingChunk(
                            chunk_index=chunk_idx,
                            is_final=True,
                            duration_ms=t_now,
                        )
                    return
                elif evt["type"] == "error":
                    logger.error(
                        f"[HalfDuplex] decode error: {evt.get('message')!r}"
                    )
                    yield StreamingChunk(chunk_index=chunk_idx, is_final=True)
                    return
        except Exception as e:
            logger.error(f"[HalfDuplex] decode_streaming failed: {e}", exc_info=True)
            yield StreamingChunk(chunk_index=chunk_idx, is_final=True)
            return
        # If decode_streaming exited without a 'done' event (shouldn't
        # happen but stay defensive), emit a final chunk so the caller
        # WS loop terminates.
        yield StreamingChunk(chunk_index=chunk_idx, is_final=True)
        # Note: legacy code below is kept as a no-op safety net only when
        # ``return`` is missed; the decode_streaming generator always
        # emits a 'done' event at the end.

    def reset_half_duplex_session(
        self,
        lang: Optional[str] = None,
        ref_audio_path: Optional[str] = None,
        system_content: Any = None,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache the half-duplex session-level config snapshot.

        Each subsequent ``half_duplex_prefill`` / ``half_duplex_omni_prefill``
        starts a fresh simplex turn (see SimplexCppBackend.begin_turn) and
        looks up these values to feed into ``update_session_config``. We
        deliberately do NOT call C++ here - per-turn ``begin_turn`` will
        do it once it has the user audio in hand, which keeps the C++
        KV cache in lockstep with what the model is about to see.

        ``ref_audio_path`` is also cached: SimplexCppBackend's
        ``begin_turn`` reads ``self.ref_audio_path`` from the backend
        instance, so we update that too if the request specified one.
        """
        if lang:
            self._last_lang = lang
        if ref_audio_path:
            # Per-session ref-audio override winds up in the next
            # begin_turn's prefill(cnt=0) audio_path; mutating the
            # backend instance is safe because half-duplex is single
            # session per worker.
            self.simplex.ref_audio_path = ref_audio_path
        self._hdx_config = {
            "system_content": system_content,
            "sampling": sampling,
            "lang": lang or self._last_lang or "zh",
        }
        self._duplex_chunk_counter = 0
        self._round_number = 0

    def half_duplex_omni_prefill(
        self,
        audio_waveform: np.ndarray,
        frame_list: Optional[list] = None,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        """Half-Duplex omni prefill: open a fresh simplex turn, push the
        VAD'ed user audio plus any sampled vision frames.

        ``max_slice_nums`` is forwarded to image prefill so the C++ side
        can decide between hi-res (slice) and fast (no-slice) packing
        per request. Returns the legacy ``{n_vision_images: int}`` dict
        so worker.py's metrics path keeps working unchanged."""
        snapshot = getattr(self, "_hdx_config", None) or {}

        self.simplex.begin_turn(
            system_content=snapshot.get("system_content"),
            sampling=snapshot.get("sampling"),
            lang=snapshot.get("lang") or self._last_lang or "zh",
            return_logits=False,
        )

        if audio_waveform is not None and len(audio_waveform) > 0:
            self.simplex.push_audio(audio_waveform)

        n_vision_images = 0
        if frame_list:
            for frame in frame_list:
                self.simplex.push_image(frame, max_slice_nums=max_slice_nums)
                n_vision_images += 1
        return {"n_vision_images": n_vision_images}

    # ================================================================
    # Chat
    # ================================================================

    def chat(self, request) -> "ChatResponse":
        """Drive one stateless simplex turn through ``self.simplex``.

        This is the production path for ``POST /chat``. The legacy
        body that hand-rolled ``update_session_config + prefill(0) +
        decode + WAV poll`` has been replaced by the dedicated
        ``SimplexCppBackend`` (``begin_turn -> push_* -> decode_oneshot``).
        Two of the historical bugs that lived here are now structurally
        impossible:

          * ``prefill(cnt=0)`` is no longer a placeholder that silently
            fails when the server rejects empty bodies (D3 was masked
            by ``_call_prefill`` swallowing 400s).
          * ``request_id`` flows through to ``logit_filename`` via
            ``make_logits_filename`` to keep multi-worker logits files
            collision-free.
        """
        from core.schemas.chat import ChatResponse
        from core.processors.base import MiniCPMOProcessorMixin
        from core.schemas.logits import LogitsExportSpec
        from .cpp_session import sampling_from_generation
        from .logits_retention import resolve_output_dir, make_logits_filename

        generation = getattr(request, "generation", None)
        length_penalty = float(getattr(generation, "length_penalty", 1.1) or 1.1)
        sampling = sampling_from_generation(generation)

        system_content: Any = None
        for m in request.messages:
            role = getattr(m, "role", None)
            role_str = role.value if hasattr(role, "value") else role
            if role_str == "system":
                system_content = getattr(m, "content", None)
                break

        logits_spec: LogitsExportSpec = getattr(request, "logits", None) or LogitsExportSpec()
        max_new = None
        if generation is not None:
            max_new = getattr(generation, "max_new_tokens", None)
            if max_new is None and isinstance(generation, dict):
                max_new = generation.get("max_new_tokens")
        try:
            max_new_int = int(max_new) if max_new is not None else 0
        except (TypeError, ValueError):
            max_new_int = 0

        sx = self.simplex
        sx.begin_turn(
            system_content=system_content,
            sampling=sampling,
            return_logits=logits_spec.enabled,
        )

        mixin = MiniCPMOProcessorMixin()
        for msg in request.messages:
            role = getattr(msg, "role", None)
            role_str = role.value if hasattr(role, "value") else role
            if role_str == "system":
                continue
            for item in mixin._convert_content_to_model_format(msg.content):
                if isinstance(item, np.ndarray):
                    sx.push_audio(item)
                elif isinstance(item, str):
                    sx.push_text(item)
                elif hasattr(item, "size"):
                    sx.push_image(item)

        logit_format = None
        logit_output_dir = None
        logit_filename = None
        logit_extra_metadata = None
        if logits_spec.enabled:
            logit_format = logits_spec.format
            if logits_spec.format == "file":
                bucket_dir = resolve_output_dir(logits_spec.output_dir)
                logit_output_dir = (
                    bucket_dir if bucket_dir.endswith(os.sep) else bucket_dir + os.sep
                )
                override = getattr(request, "_logit_filename", None)
                logit_filename = override or make_logits_filename(
                    "chat", self.worker_idx,
                    getattr(request, "request_id", None),
                )
            logit_extra_metadata = getattr(request, "_logit_extra_metadata", None)

        want_audio = bool(getattr(request, "tts", None) and request.tts.enabled)
        out = sx.decode_oneshot(
            length_penalty=length_penalty,
            max_new_tokens=max_new_int if max_new_int > 0 else None,
            logit_format=logit_format,
            logit_output_dir=logit_output_dir,
            logit_filename=logit_filename,
            logit_extra_metadata=logit_extra_metadata,
            want_audio=want_audio,
        )

        return ChatResponse(
            text=out["text"],
            audio_data=out["audio_data"],
            audio_sample_rate=out["audio_sample_rate"],
            success=True,
            logits=out["logits"],
            request_id=getattr(request, "request_id", None),
        )

    def chat_prefill(self, session_id, msgs, omni_mode=False, max_slice_nums=None,
                     use_tts_template=False, enable_thinking=False, lang: Optional[str] = None,
                     ref_audio_path: Optional[str] = None,
                     reset_context: bool = True,
                     system_content: Any = None,
                     sampling: Optional[Dict[str, Any]] = None) -> str:
        """Chat WS prefill: open one fresh simplex turn and push every
        content item from the **last** message in ``msgs``.

        Mirrors the legacy contract:
          * ``reset_context=True`` is the only supported mode now (the
            simplex backend always resets KV per turn; carrying state
            across turns was an artifact of the old direct ``_call_*``
            path and never used in production).
          * ``system_content`` falls back to the system message inside
            ``msgs`` when the caller didn't pass one explicitly.
          * Returns the legacy sentinel string ``"prefilled"`` so
            worker.py's check ``if prefill_result == "prefilled":`` still
            works untouched.
        """
        effective_system = system_content
        if effective_system is None and msgs:
            for m in msgs:
                role = getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
                role_str = role.value if hasattr(role, "value") else role
                if role_str == "system":
                    content = (
                        getattr(m, "content", None)
                        or (m.get("content") if isinstance(m, dict) else None)
                    )
                    if content is not None:
                        effective_system = content
                    break

        # Per-session ref-audio override propagates to simplex.begin_turn.
        if ref_audio_path:
            self.simplex.ref_audio_path = ref_audio_path

        self.simplex.begin_turn(
            system_content=effective_system,
            sampling=sampling,
            lang=lang or self._last_lang or "zh",
            return_logits=False,
        )
        if lang:
            self._last_lang = lang
        logger.info(
            f"[ChatPrefill] session={session_id} omni_mode={omni_mode} "
            f"lang={lang or self._last_lang} reset_context={reset_context} "
            f"ref_audio={self.simplex.ref_audio_path!r}"
        )

        last_msg = msgs[-1] if msgs else None
        if last_msg is not None:
            content_list = last_msg.get("content", [])
            if not isinstance(content_list, list):
                content_list = [content_list]
            for item in content_list:
                if isinstance(item, np.ndarray):
                    self.simplex.push_audio(item)
                elif isinstance(item, str):
                    if not item:
                        continue
                    self.simplex.push_text(item)
                elif hasattr(item, "size"):
                    self.simplex.push_image(item, max_slice_nums=max_slice_nums or -1)
        return "prefilled"

    def chat_non_streaming_generate(self, session_id, **kwargs):
        """Chat WS non-streaming decode.

        Returns either ``sse_text`` (no audio) or ``(sse_text, waveform)``
        when audio was generated, matching the legacy return-shape so
        worker.py's POST /chat WS code keeps its existing two-branch
        unpacking unchanged."""
        length_penalty = float(kwargs.get("length_penalty", 1.1) or 1.1)
        try:
            max_new_int = int(kwargs.get("max_new_tokens") or 0)
        except (TypeError, ValueError):
            max_new_int = 0
        generate_audio = bool(kwargs.get("generate_audio", True))

        out = self.simplex.decode_oneshot(
            length_penalty=length_penalty,
            max_new_tokens=max_new_int if max_new_int > 0 else None,
            want_audio=generate_audio,
        )
        sse_text = out.get("text", "") or ""
        wav_b64 = out.get("audio_data")
        if wav_b64:
            audio_bytes = base64.b64decode(wav_b64)
            waveform = np.frombuffer(audio_bytes, dtype=np.float32)
            return sse_text, waveform
        return sse_text

    def chat_streaming_generate(self, session_id, generate_audio=True,
                                max_new_tokens=256, length_penalty=1.1):
        """Chat WS streaming decode (alias to ``half_duplex_generate``).

        Both the chat WS endpoint and the half-duplex VAD endpoint
        translate to the same simplex streaming primitive; keeping the
        two method names is purely for readability at the call site."""
        yield from self.half_duplex_generate(
            session_id=session_id,
            generate_audio=generate_audio,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
        )

    # ================================================================
    # Internal: C++ server management
    # ----------------------------------------------------------------
    # The legacy ``_start_cpp_server`` / ``_stop_cpp_server`` /
    # ``_find_server_binary`` / ``_call_omni_init`` /
    # ``_call_update_session_config`` / ``_call_prefill`` helpers
    # have been replaced by ``_CppServerProc`` (subprocess + omni_init)
    # and ``_StreamHttpClient`` (HTTP primitives) in cpp_session.py.
    # See ``load_model`` for the wiring; ``simplex`` / ``duplex`` are
    # the public entry points worker.py drives.
    # ================================================================

    # ================================================================
    # Internal: data conversion helpers
    # ================================================================

    def _save_audio_to_temp(self, audio_np: np.ndarray, prefix: str) -> str:
        import soundfile as sf

        MIN_SAMPLES = 1600
        if len(audio_np) < MIN_SAMPLES:
            audio_np = np.pad(audio_np, (0, MIN_SAMPLES - len(audio_np)), mode="constant")

        path = os.path.join(self._temp_dir, f"{prefix}.wav")
        audio_np = np.clip(audio_np, -1.0, 1.0).astype(np.float32)
        sf.write(path, audio_np, _AUDIO_INPUT_SR, format="WAV", subtype="PCM_16")
        return path

    def _save_pil_image_to_temp(self, pil_image, prefix: str) -> str:
        path = os.path.join(self._temp_dir, f"{prefix}.png")
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pil_image.save(path, format="PNG")
        return path

    def _cleanup_temp_files(self, *paths: str) -> None:
        for p in paths:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    # ================================================================
    # Internal: collect WAV output from C++ tts_wav directory
    # ================================================================

    def _iter_wav_chunks_incremental(self, timeout: float = 120.0) -> "Iterator[str]":
        """增量式收集 WAV：每出现一个新 WAV 文件就立即 yield base64 音频，不等全部完成"""
        import soundfile as sf

        # Wait up to 15s for the round directory to appear (C++ TTS creates it async)
        round_dir = None
        t_wait = time.time()
        while time.time() - t_wait < 15.0:
            round_dir = self._find_latest_round_dir()
            if round_dir:
                break
            # Also check base output dir for duplex-mode WAV files
            direct_tts = os.path.join(self._output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self._output_dir
                break
            time.sleep(0.2)
        if not round_dir:
            logger.warning("_iter_wav_chunks_incremental: no round/tts_wav dir found after 15s")
            return

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
        sent_files: set = set()
        t0 = time.time()

        while time.time() - t0 < timeout:
            if not os.path.exists(tts_wav_dir):
                time.sleep(0.1)
                continue

            current_files = sorted(
                [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
                key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
            )

            new_files = [f for f in current_files if f not in sent_files]
            for wf in new_files:
                wp = os.path.join(tts_wav_dir, wf)
                try:
                    data, _sr = sf.read(wp)
                    if len(data) == 0:
                        continue
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    yield base64.b64encode(data.tobytes()).decode("utf-8")
                    sent_files.add(wf)
                except Exception as e:
                    logger.warning(f"Failed to read {wf}: {e}")

            if os.path.exists(flag_path):
                final_files = sorted(
                    [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
                    key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
                )
                for wf in final_files:
                    if wf in sent_files:
                        continue
                    wp = os.path.join(tts_wav_dir, wf)
                    try:
                        data, _sr = sf.read(wp)
                        if len(data) > 0:
                            if data.dtype != np.float32:
                                data = data.astype(np.float32)
                            yield base64.b64encode(data.tobytes()).decode("utf-8")
                            sent_files.add(wf)
                    except Exception:
                        pass
                return

            time.sleep(0.15)

        logger.warning(f"_iter_wav_chunks_incremental timed out after {timeout}s")

    def _find_latest_round_dir(self) -> Optional[str]:
        """找到最新的 round_NNN 目录"""
        if not os.path.exists(self._output_dir):
            return None
        rounds = sorted(
            [d for d in os.listdir(self._output_dir)
             if d.startswith("round_") and os.path.isdir(os.path.join(self._output_dir, d))],
            reverse=True,
        )
        if rounds:
            return os.path.join(self._output_dir, rounds[0])
        return None

    def _wait_for_generation_done(self, round_dir: str, timeout: float = 120.0) -> bool:
        """等待 C++ TTS 异步生成完成（generation_done.flag 出现）"""
        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        flag_path = os.path.join(tts_wav_dir, "generation_done.flag")
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(flag_path):
                return True
            time.sleep(0.1)
        logger.warning(f"Timed out waiting for generation_done.flag ({timeout}s)")
        return False

    def _collect_wav_output_nowait(self, sse_text: str = "") -> tuple:
        """非阻塞版 WAV 收集：只拿新增的 WAV 文件，跳过已发送的，不做任何等待。

        用于 duplex 场景——TTS 异步生成 WAV，每个 chunk 只取增量部分。
        """
        import soundfile as sf

        # 双工模式下 C++ 把 WAV 写到根级 tts_wav/，优先检查
        direct_tts = os.path.join(self._output_dir, "tts_wav")
        if os.path.isdir(direct_tts) and any(
            f.startswith("wav_") and f.endswith(".wav") for f in os.listdir(direct_tts)
        ):
            tts_wav_dir = direct_tts
        else:
            round_dir = self._find_latest_round_dir()
            if not round_dir:
                if os.path.isdir(direct_tts):
                    tts_wav_dir = direct_tts
                else:
                    return None, sse_text
            else:
                tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            return None, sse_text

        all_files = os.listdir(tts_wav_dir)
        if all_files:
            logger.info(f"[WAV nowait] dir={tts_wav_dir}, all_files={sorted(all_files)[:10]}, sent={len(self._sent_wav_files)}")

        wav_files = sorted(
            [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
        )
        new_files = [f for f in wav_files if f not in self._sent_wav_files]
        if not new_files:
            return None, sse_text

        all_audio = []
        for wf in new_files:
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, sr = sf.read(wp)
                if len(data) > 0:
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    all_audio.append(data)
                    self._sent_wav_files.add(wf)
            except Exception:
                pass

        if not all_audio:
            return None, sse_text

        combined = np.concatenate(all_audio)
        audio_b64 = base64.b64encode(combined.astype(np.float32).tobytes()).decode("utf-8")
        return audio_b64, sse_text

    def _collect_wav_output(self, sse_text: str = "") -> tuple:
        """收集所有 WAV 文件，合并为一个 base64 float32 PCM 字符串 + 文本

        Returns:
            (audio_base64_float32, combined_text)
        """
        import soundfile as sf

        round_dir = self._find_latest_round_dir()
        # Also check base output dir for duplex-mode WAV files
        if not round_dir:
            direct_tts = os.path.join(self._output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self._output_dir
        if not round_dir:
            # Wait briefly — TTS may still be creating the directory
            for _ in range(30):
                time.sleep(0.2)
                round_dir = self._find_latest_round_dir()
                if round_dir:
                    break
                direct_tts = os.path.join(self._output_dir, "tts_wav")
                if os.path.isdir(direct_tts):
                    round_dir = self._output_dir
                    break
        if not round_dir:
            return None, sse_text

        self._wait_for_generation_done(round_dir)

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            return None, sse_text

        wav_files = sorted(
            [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
        )

        if not wav_files:
            return None, sse_text

        all_audio = []
        for wf in wav_files:
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, sr = sf.read(wp)
                if len(data) > 0:
                    if data.dtype != np.float32:
                        data = data.astype(np.float32)
                    all_audio.append(data)
            except Exception as e:
                logger.warning(f"Failed to read {wf}: {e}")

        if not all_audio:
            return None, sse_text

        combined = np.concatenate(all_audio)
        audio_b64 = base64.b64encode(combined.astype(np.float32).tobytes()).decode("utf-8")
        return audio_b64, sse_text

    def _collect_all_wav_chunks(self, sse_text: str = "") -> List[tuple]:
        """收集所有 WAV 文件，每个文件作为独立 chunk

        Returns:
            [(audio_base64_float32, text), ...]
        """
        import soundfile as sf

        round_dir = self._find_latest_round_dir()
        if not round_dir:
            direct_tts = os.path.join(self._output_dir, "tts_wav")
            if os.path.isdir(direct_tts):
                round_dir = self._output_dir
        if not round_dir:
            if sse_text:
                return [(None, sse_text)]
            return []

        self._wait_for_generation_done(round_dir)

        tts_wav_dir = os.path.join(round_dir, "tts_wav")
        if not os.path.exists(tts_wav_dir):
            if sse_text:
                return [(None, sse_text)]
            return []

        wav_files = sorted(
            [f for f in os.listdir(tts_wav_dir) if f.startswith("wav_") and f.endswith(".wav")],
            key=lambda f: int(re.search(r"wav_(\d+)", f).group(1)) if re.search(r"wav_(\d+)", f) else 0,
        )

        results = []
        for i, wf in enumerate(wav_files):
            wp = os.path.join(tts_wav_dir, wf)
            try:
                data, sr = sf.read(wp)
                if len(data) == 0:
                    continue
                if data.dtype != np.float32:
                    data = data.astype(np.float32)
                audio_b64 = base64.b64encode(data.tobytes()).decode("utf-8")
                text = sse_text if i == 0 else None
                results.append((audio_b64, text))
            except Exception as e:
                logger.warning(f"Failed to read {wf}: {e}")

        if not results and sse_text:
            results.append((None, sse_text))
        return results

    def _read_llm_text(self, llm_debug_dir: str) -> str:
        """从 llm_debug 目录读取所有文本并拼接"""
        text_file = os.path.join(llm_debug_dir, "llm_text.txt")
        if os.path.exists(text_file):
            try:
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                texts = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r"\[chunk_\d+\]\s*(.*)", line)
                    texts.append(m.group(1).strip() if m else line)
                return "".join(texts)
            except Exception:
                pass

        # fallback: read per-chunk text files
        texts = []
        for i in range(100):
            chunk_dir = os.path.join(llm_debug_dir, f"chunk_{i}")
            txt_path = os.path.join(chunk_dir, "llm_text.txt")
            if not os.path.exists(txt_path):
                break
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read().strip())
            except Exception:
                break
        return "".join(texts)

    def _read_llm_text_lines(self, llm_debug_dir: str) -> List[str]:
        """从 llm_debug 目录按 chunk 读取文本列表"""
        text_file = os.path.join(llm_debug_dir, "llm_text.txt")
        if os.path.exists(text_file):
            try:
                with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                results = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.match(r"\[chunk_\d+\]\s*(.*)", line)
                    results.append(m.group(1).strip() if m else line)
                return results
            except Exception:
                pass

        results = []
        for i in range(100):
            chunk_dir = os.path.join(llm_debug_dir, f"chunk_{i}")
            txt_path = os.path.join(chunk_dir, "llm_text.txt")
            if not os.path.exists(txt_path):
                break
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    results.append(f.read().strip())
            except Exception:
                break
        return results

    # ================================================================
    # Internal: output directory management
    # ================================================================

    def _reset_output_dir(self) -> None:
        if os.path.exists(self._output_dir):
            for item in os.listdir(self._output_dir):
                item_path = os.path.join(self._output_dir, item)
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                except Exception:
                    pass
        os.makedirs(self._output_dir, exist_ok=True)
        os.makedirs(os.path.join(self._output_dir, "round_000", "tts_wav"), exist_ok=True)

    # ================================================================
    # Internal: auto detect LLM model
    # ================================================================

    @staticmethod
    def _auto_detect_llm_model(model_dir: str) -> str:
        import glob

        # 优先 Q8，再回退到 Q4 / F16（与显式配置 llm_model 时的推荐一致）
        patterns = ["*Q8_0*.gguf", "*Q4_K_M*.gguf", "*Q4_K_S*.gguf", "*F16*.gguf"]
        for pat in patterns:
            matches = glob.glob(os.path.join(model_dir, pat))
            root = [m for m in matches if os.path.dirname(m) == model_dir]
            if root:
                return os.path.basename(sorted(root)[0])

        all_gguf = glob.glob(os.path.join(model_dir, "*.gguf"))
        candidates = [f for f in all_gguf
                      if not any(x in os.path.basename(f).lower()
                                 for x in ("audio", "vision", "tts", "projector"))]
        if candidates:
            return os.path.basename(sorted(candidates)[0])

        raise RuntimeError(f"No LLM GGUF found in {model_dir}")

    # ================================================================
    # Cleanup
    # ================================================================

    def shutdown(self) -> None:
        if self._cpp_process:
            logger.info("Stopping C++ server...")
            self._cpp_process.terminate()
            try:
                self._cpp_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._cpp_process.kill()
            self._cpp_process = None

        if self._http_client:
            self._http_client.close()
            self._http_client = None

        if os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
