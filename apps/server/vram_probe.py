"""Detect available GPU VRAM so we can pick a sane default ctx_size.

Background
----------
On a 12 GB consumer NVIDIA card (e.g. RTX 5070) running MiniCPM-o-4_5 in
duplex mode:

    Q4_K_M LLM weights ............ ~5.0 GB
    LLM KV cache @ 4K (fp16) ...... ~2.4 GB    →  total ~13 GB  (just fits)
    LLM KV cache @ 8K (fp16) ...... ~4.8 GB    →  total ~15 GB  (overflow)
    + vision encoder, audio encoder,
      TTS LLM, Token2Wav (flow + voc_hg2)

When the total exceeds physical VRAM, CUDA silently spills part of the
Token2Wav vocoder weights into host memory. Each subsequent vocoder pass
then drags weights back over PCIE and the realtime-factor jumps from
~0.17 (5070, 4K) to ~3.0 (5070, 8K) — an 18× regression we directly
measured in the comni_service-5070 logs.

The vocoder slowdown is the *visible* failure: audio chunks back up,
the duplex session drops chunks, and the user perceives it as "卡".
Picking a smaller ctx_size keeps everything in VRAM and avoids the
PCIE-swap death spiral.

Probing strategy
----------------
We try in order:

    1. ``nvidia-smi`` CLI (universally available with the driver, no
       Python dep). Reports ``memory.total`` and ``memory.free`` in MiB
       per GPU.
    2. Give up — return ``ProbeResult(total_gb=0, free_gb=0, source="none")``.

We deliberately do NOT depend on ``pynvml`` / ``nvidia-ml-py`` / ``torch``
to keep the launcher import-light: those packages either need to be
bundled (pynvml has C bindings) or pull in CUDA at import time, both of
which are landmines in the embedded-Python desktop build.

On Apple Silicon the GPU is part of the unified-memory pool, so VRAM
isn't a separate budget — callers should fall back to ``_detect_system_ram_gb()``
on Darwin and use this module only as a Linux/Windows NVIDIA detector.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of a VRAM probe.

    ``total_gb`` and ``free_gb`` are 0.0 when no NVIDIA GPU could be
    queried (e.g. macOS, AMD-only Windows box, broken driver). Callers
    must treat 0.0 as "unknown", NOT "0 bytes".

    ``source`` is a short human-readable tag for logging, one of:
        "nvidia-smi", "none"
    """
    total_gb: float
    free_gb: float
    source: str
    gpu_name: str = ""

    @property
    def known(self) -> bool:
        return self.total_gb > 0


# Subprocess timeout. nvidia-smi is normally instant, but a half-broken
# driver can hang the call indefinitely — we'd rather report "unknown"
# than freeze the launcher splash screen.
_PROBE_TIMEOUT_S = 3.0


def _probe_nvidia_smi() -> ProbeResult | None:
    """Query the *first* visible NVIDIA GPU via nvidia-smi.

    Returns None if nvidia-smi isn't on PATH or the call fails. We pick
    GPU 0 because that's what cpp_backend defaults to (CUDA_VISIBLE_DEVICES=0
    in _start_cpp_server). If the user has a multi-GPU box and pins a
    different index, the recommendation is still in the right ballpark
    — gaming GPUs in the same machine usually have similar VRAM.
    """
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return None

    cmd = [
        nvsmi,
        "--query-gpu=memory.total,memory.free,name",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        # creationflags=CREATE_NO_WINDOW on Windows so we don't flash a
        # console window in PyInstaller-windowed builds.
        kwargs: dict = dict(
            timeout=_PROBE_TIMEOUT_S,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        if sys.platform == "win32":
            # 0x08000000 = CREATE_NO_WINDOW; defined here to avoid
            # importing subprocess.CREATE_NO_WINDOW in non-frozen contexts
            # where the constant might be missing on older Pythons.
            kwargs["creationflags"] = 0x08000000
        out = subprocess.check_output(cmd, **kwargs).strip()
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("nvidia-smi probe failed: %s", e)
        return None

    line = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return None
    try:
        total_mib = float(parts[0])
        free_mib = float(parts[1])
    except ValueError:
        return None
    name = parts[2] if len(parts) >= 3 else ""

    return ProbeResult(
        total_gb=total_mib / 1024.0,
        free_gb=free_mib / 1024.0,
        source="nvidia-smi",
        gpu_name=name,
    )


def detect_vram() -> ProbeResult:
    """Best-effort GPU VRAM probe. Always returns; never raises.

    Caller pattern:

        v = detect_vram()
        if v.known and v.total_gb < 14:
            ctx = 4096
        elif v.known:
            ctx = 8192
        else:
            ctx = recommend_from_ram(...)
    """
    r = _probe_nvidia_smi()
    if r is not None:
        return r
    return ProbeResult(total_gb=0.0, free_gb=0.0, source="none")


# Empirically derived from the 5070 12GB regression: 8K ctx + the full
# omni stack (LLM + vision + audio + TTS LLM + Token2Wav) needs roughly
# 14 GB of headroom before the vocoder stops getting evicted to host
# memory. Below this threshold we keep ctx_size at 4K.
SAFE_8K_VRAM_GB = 14.0


def recommend_ctx_size(*, vram_total_gb: float, ram_gb: float) -> int:
    """Pick 4096 vs 8192 based on whichever memory probe is most relevant.

    Priority:
      1. NVIDIA VRAM, if known. This is the only signal that actually
         predicts the Token2Wav vocoder slowdown — the regression we
         observed on the 5070 happens with plenty of system RAM free.
      2. System RAM, as a fallback (macOS unified memory, CPU-only boxes,
         or NVIDIA boxes where nvidia-smi failed).

    Anything we can't classify defaults to 4K — a smaller ctx is always
    safe; 8K on a small GPU is the failure mode we're trying to prevent.
    """
    if vram_total_gb > 0:
        return 8192 if vram_total_gb >= SAFE_8K_VRAM_GB else 4096
    if ram_gb >= 8:
        return 8192
    return 4096
