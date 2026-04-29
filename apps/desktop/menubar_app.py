#!/usr/bin/env python3
"""Comni — macOS Desktop App

带主窗口 + 菜单栏图标的原生 macOS 应用。
主窗口显示服务状态、模型信息、操作按钮和实时日志。
菜单栏图标提供快捷操作。

模型管理：
    - 默认模型目录 ~/.comni/models/
    - 支持 symlink 导入已有模型文件夹（不复制，节省磁盘）
    - 主界面只展示模型名称和组件状态
    - 次级窗口可管理模型（导入/删除/查看路径）
"""

import os
import sys
import json
import time
import socket
import subprocess
import webbrowser
import threading
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, List

# Ensure apps/ is on sys.path so `from server.model_hub import ...` works
# when run from the .app bundle (CWD = Contents/Resources/)
_apps_root_str = str(Path(__file__).resolve().parent.parent)
if _apps_root_str not in sys.path:
    sys.path.insert(0, _apps_root_str)

import objc
from AppKit import (
    NSApplication, NSApp, NSObject, NSWindow, NSView, NSTextField,
    NSButton, NSStatusBar, NSMenu, NSMenuItem, NSImage,
    NSApplicationActivationPolicyRegular, NSApplicationActivationPolicyAccessory,
    NSBackingStoreBuffered, NSWindowStyleMaskTitled, NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable, NSMakeRect, NSFont, NSColor,
    NSTextAlignmentCenter, NSTextAlignmentLeft, NSTextAlignmentRight,
    NSBezelStyleRounded,
    NSOpenPanel, NSVariableStatusItemLength, NSOnState, NSOffState,
    NSRunningApplication, NSApplicationActivateIgnoringOtherApps,
    NSWindowStyleMaskResizable, NSScrollView, NSTextView,
    NSBorderlessWindowMask, NSAlert, NSAlertStyleInformational,
    NSAlertStyleWarning, NSAlertStyleCritical, NSBox, NSProgressIndicator,
    NSLineBreakByTruncatingMiddle,
)
from Foundation import NSTimer, NSRunLoop, NSDefaultRunLoopMode, NSURL

# ── Paths (needed before logger setup) ───────────────────
_APP_DIR = Path(__file__).resolve().parent
_APPS_ROOT = _APP_DIR.parent
_SERVER_DIR = _APPS_ROOT / "server"
_REPO_ROOT = _APPS_ROOT.parent
_CONFIG_PATH = _SERVER_DIR / "config.json"
_CONFIG_EXAMPLE = _SERVER_DIR / "config.example.json"

_COMNI_HOME = Path.home() / ".comni"
_MODELS_HOME = _COMNI_HOME / "models"
_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "Comni"
_APP_SUPPORT.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _APP_SUPPORT / "comni_service.log"
_APP_LOG_PATH = _APP_SUPPORT / "comni_app.log"

# ── Logging ──────────────────────────────────────────────
_log_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("comni")
logger.setLevel(logging.DEBUG)
_fh = logging.handlers.RotatingFileHandler(
    str(_APP_LOG_PATH), maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_log_fmt)
logger.addHandler(_fh)
_sh = logging.StreamHandler()
_sh.setLevel(logging.INFO)
_sh.setFormatter(_log_fmt)
logger.addHandler(_sh)

# Legacy paths (auto-migrate)
_LEGACY_APP_SUPPORT = Path.home() / "Library" / "Application Support" / "LlamaOmni"
_LEGACY_LOG_PATH = _LEGACY_APP_SUPPORT / "omni_service.log"

DEFAULT_WORKER_BASE_PORT = 22700
DEFAULT_GATEWAY_PORT = 8006
LEGACY_WORKER_PORT = 22400

WIN_W, WIN_H = 540, 720
_PAD = 20
_CARD_W = WIN_W - 2 * _PAD
_CARD_INSET = 16
_INNER_L = _PAD + _CARD_INSET
_INNER_W = _CARD_W - 2 * _CARD_INSET

# Context window choices exposed in the main window. Larger ctx → bigger KV
# cache → more (unified) memory. We deliberately cap the picker at 8K on the
# desktop build:
#   * MiniCPM-o is shipped for end-side use — voice/video chats almost never
#     need more than ~8K of live context once the C++ duplex sliding-window
#     prune kicks in.
#   * 16K/32K KV cache fights with the vision/audio encoders and pushes
#     16-24 GB Macs into swap or wired-memory pressure.
#   * Power users on 64GB+ workstations can still hand-edit ctx_size in
#     ~/.comni/config.json — this picker just hides the foot-gun by default.
CTX_SIZE_CHOICES = (4096, 8192)
CTX_SIZE_MAX = max(CTX_SIZE_CHOICES)


def _coerce_ctx_size(cs: object) -> int | None:
    """Map any incoming ctx_size value onto an allowed choice.

    Returns None for values we can't parse at all (caller decides whether to
    fall back to a recommended default). Values larger than CTX_SIZE_MAX are
    clamped down — this is what migrates 16K/32K configs left over from
    earlier Comni builds without forcing the user to re-pick.
    """
    if not isinstance(cs, int):
        return None
    if cs in CTX_SIZE_CHOICES:
        return cs
    if cs > CTX_SIZE_MAX:
        return CTX_SIZE_MAX
    return min(CTX_SIZE_CHOICES, key=lambda c: abs(c - cs))

# Download source choices: (config value, UI label). "auto" walks the chain
# HF → HF mirror → ModelScope so it works for both overseas and domestic
# users out of the box; users behind aggressive firewalls (where neither HF
# nor hf-mirror reaches) can pin "modelscope" to skip the discovery step.
DOWNLOAD_SOURCE_CHOICES = (
    ("auto", "Auto (recommended)"),
    ("hf", "HuggingFace"),
    ("hf_mirror", "HF Mirror (hf-mirror.com)"),
    ("modelscope", "ModelScope (魔搭, China)"),
)


def _detect_system_ram_gb() -> float:
    """Total physical RAM in GB; 0.0 if unknown.

    Used to pick a sensible default ctx_size on first launch. Falls back
    silently rather than raising — a wrong ctx_size is recoverable, but a
    crashed launcher is not.
    """
    import platform as _plat
    sysname = _plat.system()
    try:
        if sysname == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=2).strip()
            return int(out) / (1024 ** 3)
        if sysname == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024 ** 2)
        if sysname == "Windows":
            import ctypes

            class _MemStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("sullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MemStatusEx()
            stat.dwLength = ctypes.sizeof(_MemStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def _load_build_info() -> dict:
    """Read apps/build_info.json shipped in the .app bundle.

    Empty dict in dev mode (running from source); the dict is populated
    by build_dmg.sh and includes "version", "build_commit", "build_time".
    """
    path = _APPS_ROOT / "build_info.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _format_version_tag() -> str:
    """Short human-readable version, e.g. 'v1.0.2' or 'dev'."""
    bi = _load_build_info()
    v = bi.get("version")
    if isinstance(v, str) and v:
        return f"v{v}" if not v.startswith("v") else v
    return "dev"


_VRAM_PROBE_CACHE = None  # populated lazily by _detect_vram()


def _detect_vram():
    """Cached VRAM probe. On Apple Silicon nvidia-smi is absent and the
    probe returns ``known=False``, which is the right outcome — Macs
    have unified memory, so the RAM heuristic is what we actually want.
    Bootcamped Macs / external NVIDIA enclosures will trigger the probe
    and benefit from VRAM-aware defaults.
    """
    global _VRAM_PROBE_CACHE
    if _VRAM_PROBE_CACHE is not None:
        return _VRAM_PROBE_CACHE
    try:
        from server.vram_probe import detect_vram  # type: ignore
        _VRAM_PROBE_CACHE = detect_vram()
    except Exception:
        logger.exception("VRAM probe failed; falling back to RAM heuristic")
        from types import SimpleNamespace
        _VRAM_PROBE_CACHE = SimpleNamespace(
            total_gb=0.0, free_gb=0.0, source="error", gpu_name="", known=False)
    return _VRAM_PROBE_CACHE


def _recommend_ctx_size(ram_gb: float) -> int:
    """Default ctx_size for first launch.

    On Macs the relevant signal is unified RAM (since the GPU shares it).
    On bootcamped / eGPU Macs with a discrete NVIDIA card the VRAM probe
    will fire and gate 8K behind a 14 GB total-VRAM threshold — see the
    Windows version of this docstring for why (Token2Wav vocoder ends up
    PCIE-swapping when the omni stack overflows VRAM).
    """
    try:
        from server.vram_probe import recommend_ctx_size  # type: ignore
        v = _detect_vram()
        return recommend_ctx_size(vram_total_gb=v.total_gb, ram_gb=ram_gb)
    except Exception:
        if 0 < ram_gb < 8:
            return 4096
        return 8192


class ServiceState:
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


STATUS_DISPLAY = {
    ServiceState.STOPPED:  ("Stopped",   (0.55, 0.55, 0.55)),
    ServiceState.STARTING: ("Starting…", (0.95, 0.70, 0.10)),
    ServiceState.RUNNING:  ("Running",   (0.20, 0.78, 0.35)),
    ServiceState.STOPPING: ("Stopping…", (0.95, 0.70, 0.10)),
    ServiceState.ERROR:    ("Error",     (0.90, 0.25, 0.20)),
}


# ── Model Management Helpers ─────────────────────────────

def find_llama_server() -> Optional[str]:
    for c in [_REPO_ROOT / "build/bin/llama-server",
              _REPO_ROOT / "build/bin/Release/llama-server"]:
        if c.exists():
            return str(c)
    return None


def get_model_dir_from_config() -> Optional[str]:
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                return json.load(f).get("cpp_backend", {}).get("model_dir")
        except Exception:
            pass
    return None


def check_model_dir(model_dir: str) -> dict:
    """Check a model directory for completeness. Uses registry if available."""
    from server.model_hub import verify_model, match_spec_by_dir

    result = {"valid": True, "missing": [], "llm": None,
              "has_audio": False, "has_tts": False, "has_vision": False,
              "has_vision_ane": False}
    model_path = Path(model_dir)
    if not model_path.exists():
        result["valid"] = False
        result["missing"].append("Directory not found")
        return result

    try:
        vr = verify_model(model_dir)
        result["llm"] = vr.llm
        result["has_audio"] = vr.has_audio
        result["has_tts"] = vr.has_tts
        result["has_vision"] = vr.has_vision
        result["has_vision_ane"] = vr.has_vision_ane
        result["missing"] = vr.missing + vr.size_mismatch
        result["valid"] = vr.complete
        return result
    except Exception:
        pass

    # Fallback: direct scan if model_hub import fails
    for pattern in ["*Q4_K_M*.gguf", "*Q4_K_S*.gguf", "*Q8_0*.gguf", "*F16*.gguf"]:
        matches = [m for m in model_path.glob(pattern) if m.parent == model_path]
        if matches:
            result["llm"] = matches[0].name
            break
    if not result["llm"]:
        all_gguf = list(model_path.glob("*.gguf"))
        llm_candidates = [f for f in all_gguf
                          if not any(x in f.stem.lower()
                                     for x in ("audio", "vision", "tts", "projector"))]
        if llm_candidates:
            result["llm"] = llm_candidates[0].name
        else:
            result["valid"] = False
            result["missing"].append("LLM GGUF model file")
    for d in model_path.iterdir():
        if not d.is_dir():
            continue
        gguf_files = list(d.glob("*.gguf"))
        name_lower = d.name.lower()
        if "audio" in name_lower and gguf_files:
            result["has_audio"] = True
        elif "tts" in name_lower and gguf_files:
            result["has_tts"] = True
        elif "vision" in name_lower and gguf_files:
            result["has_vision"] = True
            mlmodelc = list(d.glob("*.mlmodelc"))
            if mlmodelc and mlmodelc[0].is_dir():
                result["has_vision_ane"] = True
    return result


def scan_models() -> List[dict]:
    """Scan ~/.comni/models/ for model directories."""
    if not _MODELS_HOME.exists():
        return []
    results = []
    for d in sorted(_MODELS_HOME.iterdir()):
        if not d.is_dir():
            continue
        check = check_model_dir(str(d))
        size_gb = 0.0
        if check.get("llm"):
            try:
                size_gb = round((d / check["llm"]).stat().st_size / (1024**3), 1)
            except Exception:
                pass
        results.append({
            "name": d.name,
            "path": str(d),
            "is_symlink": d.is_symlink(),
            "real_path": str(d.resolve()) if d.is_symlink() else str(d),
            "size_gb": size_gb,
            **check,
        })
    return results


def import_model_link(source_dir: str) -> tuple:
    """Create symlink in models home. Returns (link_path, error_msg)."""
    _MODELS_HOME.mkdir(parents=True, exist_ok=True)
    src = Path(source_dir).resolve()
    link = _MODELS_HOME / src.name
    if link.exists():
        if link.is_symlink() and link.resolve() == src:
            return str(link), None
        return None, f"'{src.name}' already exists in models home"
    try:
        os.symlink(src, link)
        return str(link), None
    except Exception as e:
        return None, str(e)


def get_active_model_dir() -> Optional[str]:
    """Auto-detect: models home → config → None."""
    for m in scan_models():
        if m["valid"]:
            return m["path"]
    cfg = get_model_dir_from_config()
    if cfg and Path(cfg).exists():
        return cfg
    return None


def _prettify_model_family(raw: str) -> str:
    """Turn a raw GGUF family stem like 'MiniCPM-o-4_5' into 'MiniCPM-o 4.5'.

    Rules:
      1. Strip trailing '-gguf' / '_gguf'.
      2. Convert version-like segments: '-4_5' → ' 4.5', '-2_5' → ' 2.5'.
         Pattern: a hyphen followed by one or more digit_digit groups at the tail.
      3. Keep other hyphens intact (e.g. 'MiniCPM-o' stays 'MiniCPM-o').
    """
    import re
    s = re.sub(r'[-_]gguf$', '', raw, flags=re.IGNORECASE)
    s = re.sub(r'-(\d+)_(\d+)_(\d+)(?=-|$)', r' \1.\2.\3', s)
    s = re.sub(r'-(\d+)_(\d+)(?=-|$)', r' \1.\2', s)
    return s.strip()


def get_model_display_name(model_dir: Optional[str] = None) -> str:
    """Derive a human-friendly model name.

    Priority: registry display_name > parse from filename > directory name.
    Examples:
        MiniCPM-o-4_5-Q4_K_M.gguf  →  "MiniCPM-o 4.5  ·  Q4_K_M"
        Qwen-2_5-7B-Q8_0.gguf      →  "Qwen 2.5-7B  ·  Q8_0"
    """
    if not model_dir:
        model_dir = get_active_model_dir()
    if not model_dir:
        return "No model"

    # Try registry display_name first
    try:
        from server.model_hub import match_spec_by_dir
        spec = match_spec_by_dir(Path(model_dir).name)
        if spec:
            display = spec.get("display_name", "")
            if display:
                check = check_model_dir(model_dir)
                llm = check.get("llm", "")
                if llm:
                    base = llm.replace(".gguf", "")
                    quants = ("Q4_K_M", "Q4_K_S", "Q8_0", "Q4_0", "Q4_1",
                              "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q6_K", "F16")
                    for q in quants:
                        if base.endswith(q) or f"-{q}" in base:
                            return f"{display}  ·  {q}"
                return display
    except Exception:
        pass

    # Fallback: parse from filename
    check = check_model_dir(model_dir)
    llm = check.get("llm", "")
    if llm:
        base = llm.replace(".gguf", "")
        quants = ("Q4_K_M", "Q4_K_S", "Q8_0", "Q4_0", "Q4_1",
                  "Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1", "Q6_K", "F16",
                  "IQ4_XS", "IQ4_NL", "IQ3_M", "IQ3_S", "IQ2_M")
        for q in quants:
            idx = base.rfind(q)
            if idx > 0:
                family = base[:idx].rstrip("-_ ")
                pretty = _prettify_model_family(family)
                if pretty:
                    return f"{pretty}  ·  {q}"
                return q
        return _prettify_model_family(base)
    return _prettify_model_family(Path(model_dir).name)


def get_component_status_text(model_dir: Optional[str] = None) -> str:
    if not model_dir:
        model_dir = get_active_model_dir()
    if not model_dir:
        return "No model installed"
    check = check_model_dir(model_dir)
    parts = []
    parts.append("✓ LLM" if check.get("llm") else "✗ LLM")
    parts.append("✓ Audio" if check["has_audio"] else "✗ Audio")
    parts.append("✓ TTS" if check["has_tts"] else "✗ TTS")
    parts.append("✓ Vision" if check["has_vision"] else "✗ Vision")
    if check.get("has_vision_ane"):
        parts.append("✓ ANE")
    return "  ".join(parts)


def save_config(model_dir: str, gateway_port: int = DEFAULT_GATEWAY_PORT,
                worker_base_port: int = DEFAULT_WORKER_BASE_PORT):
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            config = json.load(f)
    elif _CONFIG_EXAMPLE.exists():
        with open(_CONFIG_EXAMPLE) as f:
            config = json.load(f)
    else:
        config = {"backend": "cpp", "model": {"model_path": "unused-for-cpp-backend"}}
    config["backend"] = "cpp"
    config.setdefault("cpp_backend", {})
    config["cpp_backend"]["model_dir"] = model_dir
    config["cpp_backend"].setdefault("llamacpp_root", str(_REPO_ROOT))
    config.setdefault("service", {})
    config["service"]["gateway_port"] = gateway_port
    config["service"]["worker_base_port"] = worker_base_port

    try:
        from server.model_hub import load_comni_config, save_comni_config
        comni_cfg = load_comni_config()
        vb = comni_cfg.get("vision_backend", "auto")
        if vb in ("auto", "metal", "coreml"):
            config["cpp_backend"]["vision_backend"] = vb
        raw_cs = comni_cfg.get("ctx_size")
        coerced = _coerce_ctx_size(raw_cs)
        if coerced is not None:
            config["cpp_backend"]["ctx_size"] = coerced
            # Migrate the user's persisted choice in-place when we had to
            # clamp it (e.g. 16384 → 8192 after the desktop build dropped
            # the bigger options). Keeps the GUI dropdown and runtime in
            # sync, and avoids a confusing "I picked 32K, why am I on 8K?"
            if raw_cs != coerced:
                comni_cfg["ctx_size"] = coerced
                try:
                    save_comni_config(comni_cfg)
                    logger.info(
                        "Migrated ctx_size in ~/.comni/config.json: %r -> %d",
                        raw_cs, coerced)
                except Exception:
                    logger.exception("ctx_size migration save failed")
    except Exception:
        pass

    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def get_system_info() -> str:
    import platform
    lines = [f"macOS {platform.mac_ver()[0]}  ·  {platform.machine()}"]
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        lines[0] = chip
    except Exception:
        pass
    try:
        mem = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
        lines.append(f"{round(mem / (1024**3))} GB")
    except Exception:
        pass
    return "  ·  ".join(lines)


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def next_free_port(start: int, max_tries: int = 128) -> tuple:
    orig = int(start)
    for i in range(max_tries):
        p = orig + i
        if p > 65535:
            break
        if not is_port_in_use(p):
            note = f"Port {orig} busy → using {p}\n" if p != orig else ""
            return p, note
    return orig, f"Warning: could not find free port from {orig}\n"


def resolve_gateway_worker_ports(preferred_gw, preferred_wk):
    notes = []
    gw, wk = int(preferred_gw), int(preferred_wk)
    if gw == LEGACY_WORKER_PORT:
        gw = DEFAULT_GATEWAY_PORT
    if wk == LEGACY_WORKER_PORT:
        wk = DEFAULT_WORKER_BASE_PORT
    ng, n1 = next_free_port(gw)
    if n1: notes.append(n1)
    gw = ng
    nw, n2 = next_free_port(wk)
    if n2: notes.append(n2)
    wk = nw
    if gw == wk:
        nw, n3 = next_free_port(gw + 1)
        if n3: notes.append(n3)
        wk = nw
    return gw, wk, notes


def _parse_port(s, fallback):
    try:
        p = int(str(s).strip())
        if 1024 <= p <= 65535:
            return p
    except (TypeError, ValueError):
        pass
    return fallback


# ============================================================
# AppDelegate
# ============================================================

class AppDelegate(NSObject):

    def applicationDidFinishLaunching_(self, notification):
        logger.info("App launching…")
        self._state = ServiceState.STOPPED
        self._worker_proc = None
        self._gateway_proc = None
        self._log_file = None
        self._model_manager_window = None
        self._retained_views = []
        self._downloader = None
        self._dl_progress_label = None
        self._dl_progress_bar = None
        self._quant_popups = {}

        gw, wk = self._load_ports_from_config()
        self._port = gw
        self._worker_port = wk
        self._log_tail_thread = None
        self._log_running = False
        self._log_buffer = []

        self._auto_migrate_legacy()
        self._auto_import_legacy_config_model()
        self._ensure_ctx_size_default()

        self._build_menu_bar()
        self._build_main_window()
        self._update_ui()

        self._log_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.5, self, "_flushLogBuffer", None, True)
        NSRunLoop.currentRunLoop().addTimer_forMode_(self._log_timer, NSDefaultRunLoopMode)

        NSApp.activateIgnoringOtherApps_(True)
        self._run_first_launch_check()
        logger.info("App launched. ports=%s/%s model=%s",
                     self._port, self._worker_port, get_active_model_dir())

    # ── Migration & Auto-detect ──────────────────────────────

    def _auto_migrate_legacy(self):
        """Migrate ~/Library/Application Support/LlamaOmni → Comni."""
        if _LEGACY_APP_SUPPORT.exists() and not _APP_SUPPORT.exists():
            try:
                _LEGACY_APP_SUPPORT.rename(_APP_SUPPORT)
            except Exception:
                pass
        if _LEGACY_LOG_PATH.exists() and not _LOG_PATH.exists():
            try:
                import shutil
                shutil.move(str(_LEGACY_LOG_PATH), str(_LOG_PATH))
            except Exception:
                pass

    def _auto_import_legacy_config_model(self):
        """If config has a model_dir not in models home, auto-symlink it."""
        if scan_models():
            return
        cfg_dir = get_model_dir_from_config()
        if not cfg_dir or not Path(cfg_dir).exists():
            return
        check = check_model_dir(cfg_dir)
        if not check["valid"]:
            return
        path, err = import_model_link(cfg_dir)
        if path:
            save_config(path, self._port, self._worker_port)

    # ── Config ───────────────────────────────────────────────

    def _load_ports_from_config(self):
        gw, wk = DEFAULT_GATEWAY_PORT, DEFAULT_WORKER_BASE_PORT
        if _CONFIG_PATH.exists():
            try:
                with open(_CONFIG_PATH) as f:
                    cfg = json.load(f)
                gw = cfg.get("service", {}).get("gateway_port", gw)
                wk = cfg.get("service", {}).get("worker_base_port", wk)
            except Exception:
                pass
        if wk == LEGACY_WORKER_PORT: wk = DEFAULT_WORKER_BASE_PORT
        if gw == LEGACY_WORKER_PORT: gw = DEFAULT_GATEWAY_PORT
        return gw, wk

    def _read_ports_from_ui(self):
        gw = _parse_port(self._gw_field.stringValue(), self._port)
        wk = _parse_port(self._wk_field.stringValue(), self._worker_port)
        return gw, wk

    # ── First Launch ─────────────────────────────────────────

    @objc.python_method
    def _ensure_ctx_size_default(self):
        """Stamp a RAM-aware ctx_size into ~/.comni/config.json on first run.

        Also migrates legacy values (16K / 32K from earlier Comni builds) to
        the current allowed range. The desktop picker now caps at 8K
        (see CTX_SIZE_CHOICES); without the clamp, the dropdown would
        silently fall back to the recommended default and the user's
        runtime would mysteriously shrink. Migrating in-place keeps the
        change visible and one-time only.
        """
        try:
            from server.model_hub import load_comni_config, save_comni_config
            cfg = load_comni_config()
            raw_cs = cfg.get("ctx_size")
            coerced = _coerce_ctx_size(raw_cs)
            if coerced is not None and raw_cs == coerced:
                return  # already a legal value
            if coerced is not None and raw_cs != coerced:
                cfg["ctx_size"] = coerced
                save_comni_config(cfg)
                logger.info("Migrated ctx_size %r → %d", raw_cs, coerced)
                return
            ram_gb = _detect_system_ram_gb()
            vram = _detect_vram()
            recommended = _recommend_ctx_size(ram_gb)
            cfg["ctx_size"] = recommended
            save_comni_config(cfg)
            logger.info(
                "First-run ctx_size default: %d "
                "(RAM=%.1f GB, VRAM=%.1f GB src=%s gpu=%r)",
                recommended, ram_gb,
                getattr(vram, "total_gb", 0.0),
                getattr(vram, "source", "unknown"),
                getattr(vram, "gpu_name", ""))
        except Exception:
            logger.exception("_ensure_ctx_size_default failed")

    def _run_first_launch_check(self):
        issues = []
        if not find_llama_server():
            issues.append(
                "llama-server binary not found.\n"
                f"  cd {_REPO_ROOT}\n"
                "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
                "  cmake --build build --target llama-server -j")

        model_dir = get_active_model_dir()
        if not model_dir:
            issues.append(
                "No model installed.\n"
                "Click 'Manage Models' to download or import a model.")
        else:
            try:
                from server.model_hub import verify_model
                vr = verify_model(model_dir)
                if not vr.complete:
                    parts = []
                    for m in vr.missing:
                        parts.append(f"  - Missing: {m}")
                    for m in vr.size_mismatch:
                        parts.append(f"  - Size mismatch: {m}")
                    issues.append(
                        f"Model integrity issue ({Path(model_dir).name}):\n"
                        + "\n".join(parts) + "\n"
                        "  Click 'Manage Models' → 'Download' to fix.")
                    logger.warning("Startup verify: model incomplete — %s",
                                   vr.missing + vr.size_mismatch)
                else:
                    logger.info("Startup verify OK: %d files", len(vr.verified))
            except Exception:
                logger.debug("Startup verify skipped (model_hub not available)")

        if issues:
            self._append_log("=== Startup Check ===\n")
            for i in issues:
                self._append_log(f"  {i}\n")
            self._append_log("=====================\n")

    # ── Menu Bar ─────────────────────────────────────────────

    def _build_menu_bar(self):
        sb = NSStatusBar.systemStatusBar()
        self._status_item = sb.statusItemWithLength_(NSVariableStatusItemLength)
        self._status_item.setTitle_("Comni")
        menu = NSMenu.alloc().init()
        # First (disabled) menu item is the version label. Single source of
        # truth for "which build is this" — easy to read off when triaging
        # bug reports: "open menu, read first line".
        version_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            f"Comni {_format_version_tag()}", None, "")
        version_item.setEnabled_(False)
        menu.addItem_(version_item)
        menu.addItem_(NSMenuItem.separatorItem())
        self._menu_status = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Status: Stopped", None, "")
        self._menu_status.setEnabled_(False)
        menu.addItem_(self._menu_status)
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Start Server", "menuStart:", "s"))
        menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Stop Server", "menuStop:", ""))
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Open Web UI", "menuOpenBrowser:", "o"))
        menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Show Window", "menuShowWindow:", "w"))
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "About Comni…", "menuAbout:", ""))
        menu.addItem_(NSMenuItem.separatorItem())
        menu.addItem_(NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit", "menuQuit:", "q"))
        self._status_item.setMenu_(menu)

    # ── Main Window ──────────────────────────────────────────

    def _build_main_window(self):
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                 | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable)
        self._window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(200, 200, WIN_W, WIN_H), style, NSBackingStoreBuffered, False)
        self._window.setTitle_("Comni")
        self._window.setMinSize_((480, 580))
        self._window.setReleasedWhenClosed_(False)
        self._window.setDelegate_(self)
        self._window.center()
        self._window.setBackgroundColor_(NSColor.underPageBackgroundColor())
        cv = self._window.contentView()
        y = WIN_H - 40

        # ── Title ──
        y -= 30
        cv.addSubview_(self._make_label("Comni", x=0, y=y, w=WIN_W, h=28,
                                         size=22, bold=True, align=NSTextAlignmentCenter))
        # 主窗右上角的 About 入口 —— 否则 About 只能从菜单栏状态项的下拉里点到，
        # 用户在主窗里完全看不到。位置贴边，不抢标题视觉重心。
        about_btn = self._make_button(
            "About", x=WIN_W - _PAD - 64, y=y + 2, w=64, h=22,
            action="menuAbout:")
        cv.addSubview_(about_btn)
        y -= 20
        cv.addSubview_(self._make_label("Omni inference in C/C++",
                                         x=0, y=y, w=WIN_W, h=18, size=11.5,
                                         color=NSColor.secondaryLabelColor(),
                                         align=NSTextAlignmentCenter))
        y -= 16

        # ── Status Card ──
        sc_h = 158
        sc_y = y - sc_h
        cv.addSubview_(self._make_card(x=_PAD, y=sc_y, w=_CARD_W, h=sc_h))

        iy = y - _CARD_INSET - 24
        self._status_label = self._make_label("●  Stopped", x=_INNER_L, y=iy, w=300, h=24,
                                               size=20, bold=True)
        cv.addSubview_(self._status_label)
        self._open_btn = self._make_button("Open Web UI  →",
                                            x=_PAD + _CARD_W - _CARD_INSET - 130, y=iy - 1,
                                            w=130, h=28, action="onOpenBrowser:")
        self._open_btn.setHidden_(True)
        cv.addSubview_(self._open_btn)

        iy -= 20
        self._sysinfo_label = self._make_label(get_system_info(), x=_INNER_L, y=iy,
                                                w=_INNER_W, h=16, size=11,
                                                color=NSColor.secondaryLabelColor())
        cv.addSubview_(self._sysinfo_label)

        iy -= 16
        found = find_llama_server()
        self._server_label = self._make_label(
            "llama-server  ✓" if found else "llama-server  ✗  Not built",
            x=_INNER_L, y=iy, w=_INNER_W, h=15, size=11,
            color=NSColor.secondaryLabelColor() if found else NSColor.systemRedColor())
        cv.addSubview_(self._server_label)

        iy -= 16
        self._progress_bar = NSProgressIndicator.alloc().initWithFrame_(
            NSMakeRect(_INNER_L, iy + 4, _INNER_W - 120, 6))
        self._progress_bar.setStyle_(0)
        self._progress_bar.setIndeterminate_(True)
        self._progress_bar.setHidden_(True)
        cv.addSubview_(self._progress_bar)
        self._progress_text = self._make_label("Loading models…",
                                                x=_INNER_L + _INNER_W - 115, y=iy, w=115, h=14,
                                                size=11, color=NSColor.secondaryLabelColor(),
                                                align=NSTextAlignmentRight)
        self._progress_text.setHidden_(True)
        cv.addSubview_(self._progress_text)

        btn_y = sc_y + _CARD_INSET
        self._start_btn = self._make_button("▶  Start Server", x=_INNER_L, y=btn_y,
                                             w=180, h=36, action="onStart:")
        self._start_btn.setKeyEquivalent_("\r")
        cv.addSubview_(self._start_btn)
        self._stop_btn = self._make_button("■  Stop", x=_INNER_L + 192, y=btn_y,
                                            w=100, h=36, action="onStop:")
        self._stop_btn.setEnabled_(False)
        cv.addSubview_(self._stop_btn)

        y = sc_y - 12

        # ── Model Card ──
        mc_h = 72
        mc_y = y - mc_h
        cv.addSubview_(self._make_card(x=_PAD, y=mc_y, w=_CARD_W, h=mc_h))

        model_dir = get_active_model_dir()
        self._model_name_label = self._make_label(
            get_model_display_name(model_dir),
            x=_INNER_L, y=mc_y + mc_h - _CARD_INSET - 18, w=_INNER_W - 140, h=18,
            size=13, bold=True)
        cv.addSubview_(self._model_name_label)

        manage_btn = self._make_button("Manage Models",
                                        x=_PAD + _CARD_W - _CARD_INSET - 120,
                                        y=mc_y + mc_h - _CARD_INSET - 20, w=120, h=24,
                                        action="onManageModels:")
        cv.addSubview_(manage_btn)

        self._component_label = self._make_label(
            get_component_status_text(model_dir),
            x=_INNER_L, y=mc_y + _CARD_INSET, w=_INNER_W, h=16, size=11,
            color=NSColor.secondaryLabelColor())
        cv.addSubview_(self._component_label)

        y = mc_y - 12

        # ── Service Card (两行：桌面 URL + 移动 URL) ──
        svc_h = 82
        svc_y = y - svc_h
        cv.addSubview_(self._make_card(x=_PAD, y=svc_y, w=_CARD_W, h=svc_h))

        # 第一行（靠上）：桌面 URL + Copy
        row1_y = svc_y + svc_h - _CARD_INSET - 22
        cv.addSubview_(self._make_label("Desktop",
                                         x=_INNER_L, y=row1_y, w=60, h=20,
                                         size=11, bold=True))
        self._url_label = self._make_label(
            f"https://localhost:{self._port}",
            x=_INNER_L + 64, y=row1_y, w=_INNER_W - 64 - 80, h=20,
            size=12, color=NSColor.secondaryLabelColor())
        cv.addSubview_(self._url_label)
        cv.addSubview_(self._make_button(
            "Copy", x=_PAD + _CARD_W - _CARD_INSET - 72,
            y=row1_y - 2, w=72, h=24, action="onCopyURL:"))

        # 第二行（靠下）：Mobile URL（局域网 IP） + QR 按钮
        row2_y = svc_y + _CARD_INSET
        cv.addSubview_(self._make_label("Mobile",
                                         x=_INNER_L, y=row2_y, w=60, h=20,
                                         size=11, bold=True))
        lan_ip = self._get_lan_ip()
        mobile_url = (f"https://{lan_ip}:{self._port}/mobile/"
                      if lan_ip else "(no LAN IP detected)")
        self._mobile_url_label = self._make_label(
            mobile_url,
            x=_INNER_L + 64, y=row2_y, w=_INNER_W - 64 - 80, h=20,
            size=12, color=NSColor.secondaryLabelColor())
        cv.addSubview_(self._mobile_url_label)
        cv.addSubview_(self._make_button(
            "QR", x=_PAD + _CARD_W - _CARD_INSET - 72,
            y=row2_y - 2, w=72, h=24, action="onShowMobileQR:"))

        y = svc_y - 8

        # ── Ports (subtle) ──
        y -= 22
        cv.addSubview_(self._make_label("Ports", x=_INNER_L, y=y + 2, w=42, h=20,
                                         size=11, color=NSColor.tertiaryLabelColor()))
        cv.addSubview_(self._make_label("Gateway", x=_INNER_L + 44, y=y + 2, w=56, h=20,
                                         size=10, color=NSColor.tertiaryLabelColor()))
        self._gw_field = self._make_port_field(self._port, x=_INNER_L + 100, y=y, w=60)
        cv.addSubview_(self._gw_field)
        cv.addSubview_(self._make_label("Worker", x=_INNER_L + 170, y=y + 2, w=50, h=20,
                                         size=10, color=NSColor.tertiaryLabelColor()))
        self._wk_field = self._make_port_field(self._worker_port, x=_INNER_L + 220, y=y, w=60)
        cv.addSubview_(self._wk_field)
        cv.addSubview_(self._make_label("auto-resolves if busy",
                                         x=_INNER_L + 290, y=y + 3, w=140, h=16,
                                         size=9, color=NSColor.tertiaryLabelColor()))
        y -= 10

        # ── Vision Backend (macOS only) ──
        import platform as _plat_mod
        if _plat_mod.system() == "Darwin":
            y -= 24
            cv.addSubview_(self._make_label("Vision", x=_INNER_L, y=y + 2, w=46, h=20,
                                             size=11, color=NSColor.tertiaryLabelColor()))
            from AppKit import NSPopUpButton
            self._vision_backend_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
                NSMakeRect(_INNER_L + 48, y, 120, 22), False)
            for title in ["Auto", "Metal (GPU)", "CoreML (ANE)"]:
                self._vision_backend_popup.addItemWithTitle_(title)
            try:
                from server.model_hub import load_comni_config
                vb = load_comni_config().get("vision_backend", "auto")
                idx = {"auto": 0, "metal": 1, "coreml": 2}.get(vb, 0)
                self._vision_backend_popup.selectItemAtIndex_(idx)
            except Exception:
                pass
            self._vision_backend_popup.setTarget_(self)
            self._vision_backend_popup.setAction_(b"onVisionBackendChanged:")
            self._vision_backend_popup.setFont_(NSFont.systemFontOfSize_(11))
            cv.addSubview_(self._vision_backend_popup)

            ane_status = ""
            model_dir_for_ane = get_active_model_dir()
            if model_dir_for_ane:
                check_ane = check_model_dir(model_dir_for_ane)
                if check_ane.get("has_vision_ane"):
                    ane_status = "ANE model ready"
                else:
                    ane_status = "ANE model not installed"
            self._ane_status_label = self._make_label(
                ane_status, x=_INNER_L + 174, y=y + 3, w=200, h=16,
                size=9, color=NSColor.tertiaryLabelColor())
            cv.addSubview_(self._ane_status_label)
            y -= 4

        # ── Context Size (cross-platform: tunes KV cache footprint) ──
        y -= 24
        cv.addSubview_(self._make_label("Context", x=_INNER_L, y=y + 2, w=54, h=20,
                                         size=11, color=NSColor.tertiaryLabelColor()))
        from AppKit import NSPopUpButton
        self._ctx_size_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(_INNER_L + 56, y, 110, 22), False)
        for cs in CTX_SIZE_CHOICES:
            label = f"{cs // 1024}K  ({cs})" if cs >= 1024 else str(cs)
            self._ctx_size_popup.addItemWithTitle_(label)
        try:
            from server.model_hub import load_comni_config
            raw_cs = load_comni_config().get("ctx_size")
            cur_cs = _coerce_ctx_size(raw_cs)
            if cur_cs is None:
                cur_cs = _recommend_ctx_size(_detect_system_ram_gb())
            idx = CTX_SIZE_CHOICES.index(cur_cs)
            self._ctx_size_popup.selectItemAtIndex_(idx)
        except Exception:
            self._ctx_size_popup.selectItemAtIndex_(CTX_SIZE_CHOICES.index(8192))
        self._ctx_size_popup.setTarget_(self)
        self._ctx_size_popup.setAction_(b"onContextSizeChanged:")
        self._ctx_size_popup.setFont_(NSFont.systemFontOfSize_(11))
        cv.addSubview_(self._ctx_size_popup)

        ram_gb = _detect_system_ram_gb()
        vram = _detect_vram()
        # On bootcamped Macs / eGPU rigs we have a real NVIDIA VRAM number
        # — show it, since that's what actually drives the recommendation.
        # On unified-memory Apple Silicon the probe is unknown and we fall
        # back to the RAM hint.
        if getattr(vram, "known", False) and vram.total_gb > 0:
            ctx_hint = f"GPU {vram.gpu_name or 'NVIDIA'} · VRAM {vram.total_gb:.0f} GB"
        elif ram_gb > 0:
            ctx_hint = f"RAM {ram_gb:.0f} GB · larger = more memory"
        else:
            ctx_hint = "larger ctx = more KV cache RAM"
        self._ctx_hint_label = self._make_label(
            ctx_hint, x=_INNER_L + 172, y=y + 3, w=240, h=16,
            size=9, color=NSColor.tertiaryLabelColor())
        cv.addSubview_(self._ctx_hint_label)
        y -= 4

        # NOTE: download_source picker used to live here. It moved into the
        # Model Manager window so it sits next to the Download buttons it
        # actually controls. The "auto" default already chains
        # HF → Mirror → ModelScope, so users behind firewalls don't lose
        # anything by the picker being one click further away.

        # ── Log Section ──
        log_hdr_y = y - 22
        cv.addSubview_(self._make_label("Service Log", x=_INNER_L, y=log_hdr_y, w=120, h=20,
                                         size=13, bold=True))
        cv.addSubview_(self._make_button("Open File", x=WIN_W - _PAD - 80, y=log_hdr_y - 2,
                                          w=80, h=22, action="onOpenLog:"))
        cv.addSubview_(self._make_button("Clear", x=WIN_W - _PAD - 80 - 64, y=log_hdr_y - 2,
                                          w=60, h=22, action="onClearLog:"))

        log_top = log_hdr_y - 8
        log_h = log_top - 16
        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(_PAD, 16, _CARD_W, log_h))
        scroll.setHasVerticalScroller_(True)
        scroll.setAutohidesScrollers_(True)
        scroll.setBorderType_(1)
        scroll.setWantsLayer_(True)
        scroll.layer().setCornerRadius_(8.0)

        self._log_textview = NSTextView.alloc().initWithFrame_(
            NSMakeRect(0, 0, _CARD_W - 16, log_h))
        self._log_textview.setEditable_(False)
        self._log_textview.setSelectable_(True)
        self._log_textview.setFont_(NSFont.monospacedSystemFontOfSize_weight_(11, 0.0))
        self._log_textview.setTextColor_(NSColor.secondaryLabelColor())
        self._log_textview.setBackgroundColor_(NSColor.textBackgroundColor())
        self._log_textview.setAutoresizingMask_(1)
        self._log_textview.textContainer().setWidthTracksTextView_(True)
        scroll.setDocumentView_(self._log_textview)
        cv.addSubview_(scroll)
        self._scroll_view = scroll

        self._window.makeKeyAndOrderFront_(None)

    # ── Model Manager Window ─────────────────────────────────

    def _build_model_manager(self):
        if self._model_manager_window:
            self._refresh_model_manager_content()
            self._model_manager_window.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
            return

        mw_w, mw_h = 560, 580
        style = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable)
        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(300, 200, mw_w, mw_h), style, NSBackingStoreBuffered, False)
        win.setTitle_("Comni — Model Manager")
        win.setMinSize_((500, 480))
        win.setReleasedWhenClosed_(False)
        win.setDelegate_(self)
        win.center()
        win.setBackgroundColor_(NSColor.underPageBackgroundColor())
        self._model_manager_window = win
        self._downloader = None
        self._dl_progress_label = None
        self._dl_progress_bar = None

        self._refresh_model_manager_content()
        win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    def _refresh_model_manager_content(self):
        win = self._model_manager_window
        if not win:
            return
        cv = win.contentView()
        for sub in list(cv.subviews()):
            sub.removeFromSuperview()

        mw_w = int(win.frame().size.width)
        mw_h = int(win.frame().size.height)
        pad = 20
        y = mw_h - 30

        # ── Available Models (from registry) ──
        y -= 24
        cv.addSubview_(self._make_label("Available Models", x=pad, y=y, w=300, h=22,
                                         size=16, bold=True))
        y -= 8

        # ── Download source picker ──
        # Moved here from the main window: this selector only affects the
        # Download buttons in the cards below, so co-locating them keeps
        # scope obvious. Persisted via ~/.comni/config.json so the choice
        # carries over to the next session and to the Windows build.
        y -= 26
        cv.addSubview_(self._make_label(
            "Download from", x=pad, y=y + 2, w=110, h=20,
            size=11, color=NSColor.secondaryLabelColor()))
        from AppKit import NSPopUpButton
        self._download_source_popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(pad + 110, y, 200, 22), False)
        for _key, label in DOWNLOAD_SOURCE_CHOICES:
            self._download_source_popup.addItemWithTitle_(label)
        try:
            from server.model_hub import load_comni_config
            cur_src = load_comni_config().get("download_source", "auto")
            keys = [k for (k, _) in DOWNLOAD_SOURCE_CHOICES]
            self._download_source_popup.selectItemAtIndex_(
                keys.index(cur_src) if cur_src in keys else 0)
        except Exception:
            self._download_source_popup.selectItemAtIndex_(0)
        self._download_source_popup.setTarget_(self)
        self._download_source_popup.setAction_(b"onDownloadSourceChanged:")
        self._download_source_popup.setFont_(NSFont.systemFontOfSize_(11))
        cv.addSubview_(self._download_source_popup)
        cv.addSubview_(self._make_label(
            "auto: HF → Mirror → ModelScope",
            x=pad + 320, y=y + 3, w=240, h=16,
            size=9, color=NSColor.tertiaryLabelColor()))
        y -= 6

        try:
            from server.model_hub import list_available_models, _MODELS_HOME as mh
            registry_models = list_available_models()
        except Exception:
            registry_models = []

        models = scan_models()
        active_dir = get_active_model_dir()
        installed_dirs = {Path(m["path"]).name for m in models}

        for spec in registry_models:
            card_h = 104
            y -= card_h + 8
            card = self._make_card(x=pad, y=y, w=mw_w - 2 * pad, h=card_h)
            cv.addSubview_(card)

            cx = pad + 14
            cw = mw_w - 2 * pad - 28

            is_installed = spec.get("dir_name", "") in installed_dirs
            installed_model = None
            if is_installed:
                for m in models:
                    if Path(m["path"]).name == spec["dir_name"]:
                        installed_model = m
                        break
            is_complete = installed_model and installed_model.get("valid", False)

            # Row 1: name + status badge
            name_str = spec.get("display_name", spec["id"])
            if is_complete and installed_model["path"] == active_dir:
                name_str += "  ✓ Active"
            elif is_complete:
                name_str += "  ✓ Installed"
            elif is_installed:
                name_str += "  ⚠ Incomplete"
            badge_color = (NSColor.controlAccentColor() if is_complete
                           else NSColor.systemOrangeColor() if is_installed
                           else NSColor.labelColor())
            cv.addSubview_(self._make_label(
                name_str, x=cx, y=y + card_h - 14 - 18, w=cw, h=18,
                size=13, bold=True, color=badge_color))

            # Row 2: description
            cv.addSubview_(self._make_label(
                spec.get("description", ""), x=cx, y=y + card_h - 38 - 14, w=cw, h=14,
                size=11, color=NSColor.secondaryLabelColor()))

            # Row 3: component status (if installed)
            if installed_model:
                comp_parts = []
                comp_parts.append("✓ Audio" if installed_model["has_audio"] else "✗ Audio")
                comp_parts.append("✓ TTS" if installed_model["has_tts"] else "✗ TTS")
                comp_parts.append("✓ Vision" if installed_model["has_vision"] else "✗ Vision")
                if installed_model.get("has_vision_ane"):
                    comp_parts.append("✓ ANE")
                cv.addSubview_(self._make_label(
                    "  ".join(comp_parts), x=cx, y=y + card_h - 58 - 14, w=cw - 200, h=14,
                    size=11, color=NSColor.secondaryLabelColor()))

            # Row 4: quant selector + action buttons
            btn_row_y = y + 10
            tag_val = hash(spec["id"]) & 0x7FFFFFFF
            needs_download = not is_installed or not is_complete

            if needs_download:
                from AppKit import NSPopUpButton
                popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
                    NSMakeRect(cx, btn_row_y, 140, 24), False)
                for vi, v in enumerate(spec.get("llm_variants", [])):
                    size_gb = round(v.get("size", 0) / (1024**3), 1)
                    title = f"{v['quant']}  ({size_gb} GB)"
                    if v.get("recommended"):
                        title += " ★"
                    popup.addItemWithTitle_(title)
                # Pre-select: if partially installed, match the existing LLM variant
                recommended_idx = next(
                    (i for i, v in enumerate(spec.get("llm_variants", []))
                     if v.get("recommended")), 0)
                if installed_model and installed_model.get("llm"):
                    existing_llm = installed_model["llm"]
                    for vi, v in enumerate(spec.get("llm_variants", [])):
                        if v["file"] == existing_llm:
                            recommended_idx = vi
                            break
                popup.selectItemAtIndex_(recommended_idx)
                popup.setTag_(tag_val)
                cv.addSubview_(popup)
                if not hasattr(self, "_quant_popups"):
                    self._quant_popups = {}
                self._quant_popups[spec["id"]] = popup

                btn_label = "Resume" if is_installed else "Download"
                dl_btn = self._make_button(btn_label, x=cx + 150, y=btn_row_y,
                                            w=100, h=24, action="onDownloadModel:")
                dl_btn.setTag_(tag_val)
                cv.addSubview_(dl_btn)
            else:
                cv.addSubview_(self._make_button("Verify", x=cx + cw - 80, y=btn_row_y,
                                                  w=80, h=24, action="onVerifyModel:"))

            # HF repo link
            cv.addSubview_(self._make_label(
                f"HF: {spec.get('hf_repo', '')}",
                x=cx, y=y + 4, w=cw, h=12, size=10,
                color=NSColor.tertiaryLabelColor()))

        # ── Download progress area ──
        y -= 8
        self._dl_progress_label = self._make_label(
            "", x=pad, y=y - 16, w=mw_w - 2 * pad, h=16, size=11,
            color=NSColor.secondaryLabelColor())
        cv.addSubview_(self._dl_progress_label)
        y -= 22
        self._dl_progress_bar = NSProgressIndicator.alloc().initWithFrame_(
            NSMakeRect(pad, y - 6, mw_w - 2 * pad, 6))
        self._dl_progress_bar.setStyle_(0)
        self._dl_progress_bar.setIndeterminate_(False)
        self._dl_progress_bar.setMinValue_(0.0)
        self._dl_progress_bar.setMaxValue_(100.0)
        self._dl_progress_bar.setDoubleValue_(0.0)
        self._dl_progress_bar.setHidden_(True)
        cv.addSubview_(self._dl_progress_bar)
        y -= 16

        # ── Installed Models (local) ──
        y -= 24
        cv.addSubview_(self._make_label("Local Models", x=pad, y=y, w=300, h=22,
                                         size=14, bold=True))
        y -= 8

        # Only show non-registry models (manually imported)
        extra_models = [m for m in models if Path(m["path"]).name not in
                        {s["dir_name"] for s in registry_models}]
        for m in extra_models:
            card_h = 70
            y -= card_h + 8
            card = self._make_card(x=pad, y=y, w=mw_w - 2 * pad, h=card_h)
            cv.addSubview_(card)
            cx = pad + 14
            cw = mw_w - 2 * pad - 28
            cv.addSubview_(self._make_label(
                m["name"], x=cx, y=y + card_h - 14 - 16, w=cw, h=16,
                size=12, bold=True))
            comp_parts = []
            comp_parts.append("✓ Audio" if m["has_audio"] else "✗ Audio")
            comp_parts.append("✓ TTS" if m["has_tts"] else "✗ TTS")
            comp_parts.append("✓ Vision" if m["has_vision"] else "✗ Vision")
            if m.get("has_vision_ane"):
                comp_parts.append("✓ ANE")
            cv.addSubview_(self._make_label(
                "  ".join(comp_parts), x=cx, y=y + card_h - 36 - 14, w=cw, h=14,
                size=11, color=NSColor.secondaryLabelColor()))
            loc = m["real_path"] if m["is_symlink"] else m["path"]
            loc_label = self._make_label(
                ("→ " if m["is_symlink"] else "") + loc,
                x=cx, y=y + 6, w=cw, h=12, size=10,
                color=NSColor.tertiaryLabelColor())
            loc_label.cell().setLineBreakMode_(NSLineBreakByTruncatingMiddle)
            cv.addSubview_(loc_label)

        # ── Bottom buttons ──
        btn_y = 50
        cv.addSubview_(self._make_button("Import Model Folder…", x=pad, y=btn_y,
                                          w=160, h=30, action="onImportModel:"))
        cv.addSubview_(self._make_button("Open in Finder", x=pad + 170, y=btn_y,
                                          w=120, h=30, action="onOpenModelsFolder:"))
        cv.addSubview_(self._make_button("Remove Selected", x=pad + 300, y=btn_y,
                                          w=130, h=30, action="onRemoveModel:"))

        cv.addSubview_(self._make_label(
            f"Model home:  {_MODELS_HOME}",
            x=pad, y=16, w=mw_w - 2 * pad, h=16, size=10,
            color=NSColor.tertiaryLabelColor()))

    # ── UI Primitives ────────────────────────────────────────

    def _make_card(self, x, y, w, h):
        box = NSBox.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
        box.setBoxType_(4)
        box.setTitlePosition_(0)
        box.setCornerRadius_(10.0)
        box.setFillColor_(NSColor.controlBackgroundColor())
        box.setBorderColor_(NSColor.separatorColor())
        box.setBorderWidth_(0.5)
        box.setContentViewMargins_((0, 0))
        return box

    def _make_label(self, text, x, y, w, h, size=13, bold=False,
                    color=None, align=NSTextAlignmentLeft):
        lbl = NSTextField.labelWithString_(text)
        lbl.setFrame_(NSMakeRect(x, y, w, h))
        font = NSFont.boldSystemFontOfSize_(size) if bold else NSFont.systemFontOfSize_(size)
        lbl.setFont_(font)
        lbl.setAlignment_(align)
        if color:
            lbl.setTextColor_(color)
        lbl.setBezeled_(False)
        lbl.setDrawsBackground_(False)
        lbl.setEditable_(False)
        lbl.setSelectable_(False)
        return lbl

    def _make_button(self, title, x, y, w, h, action):
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
        btn.setTitle_(title)
        btn.setBezelStyle_(NSBezelStyleRounded)
        btn.setTarget_(self)
        btn.setAction_(action)
        return btn

    def _make_port_field(self, value, x, y, w):
        tf = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, 20))
        tf.setStringValue_(str(int(value)))
        tf.setEditable_(True)
        tf.setBezeled_(True)
        tf.setDrawsBackground_(True)
        tf.setFont_(NSFont.monospacedSystemFontOfSize_weight_(11, 0.0))
        return tf

    # ── Log ──────────────────────────────────────────────────

    def _append_log(self, text):
        self._log_buffer.append(text)

    def _flushLogBuffer(self):
        if not self._log_buffer:
            return
        lines = self._log_buffer[:]
        self._log_buffer.clear()
        text = "".join(lines)
        storage = self._log_textview.textStorage()
        storage.beginEditing()
        storage.mutableString().appendString_(text)
        storage.endEditing()
        if storage.length() > 100000:
            storage.deleteCharactersInRange_((0, storage.length() - 100000))
        self._log_textview.scrollRangeToVisible_((storage.length(), 0))

    def _start_log_tail(self):
        self._log_running = True
        self._log_tail_thread = threading.Thread(target=self._tail_log_file, daemon=True)
        self._log_tail_thread.start()

    def _stop_log_tail(self):
        self._log_running = False

    def _tail_log_file(self):
        while self._log_running and not _LOG_PATH.exists():
            time.sleep(0.2)
        if not self._log_running:
            return
        try:
            with open(_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                while self._log_running:
                    line = f.readline()
                    if line:
                        self._append_log(line)
                    else:
                        time.sleep(0.1)
        except Exception as e:
            self._append_log(f"[Log reader error: {e}]\n")

    # ── State ────────────────────────────────────────────────

    def _update_ui(self):
        label_text, (r, g, b) = STATUS_DISPLAY[self._state]
        color = NSColor.colorWithRed_green_blue_alpha_(r, g, b, 1.0)
        self._status_label.setStringValue_(f"●  {label_text}")
        self._status_label.setTextColor_(color)

        is_running = self._state == ServiceState.RUNNING
        is_stopped = self._state in (ServiceState.STOPPED, ServiceState.ERROR)
        is_starting = self._state == ServiceState.STARTING

        self._start_btn.setEnabled_(is_stopped)
        self._stop_btn.setEnabled_(is_running)
        self._open_btn.setHidden_(not is_running)

        if is_starting:
            self._progress_bar.setHidden_(False)
            self._progress_bar.startAnimation_(None)
            self._progress_text.setHidden_(False)
        else:
            self._progress_bar.stopAnimation_(None)
            self._progress_bar.setHidden_(True)
            self._progress_text.setHidden_(True)

        model_dir = get_active_model_dir()
        self._model_name_label.setStringValue_(get_model_display_name(model_dir))
        self._component_label.setStringValue_(get_component_status_text(model_dir))

        icons = {"stopped": "◻ ", "starting": "◼ ", "running": "▶ ",
                 "stopping": "◼ ", "error": "✕ "}
        self._status_item.setTitle_(f"{icons.get(self._state, '')}Comni")
        self._menu_status.setTitle_(f"Status: {label_text}")

    def _set_state(self, state):
        self._state = state
        self.performSelectorOnMainThread_withObject_waitUntilDone_("_doUpdateUI", None, False)

    def _doUpdateUI(self):
        self._update_ui()

    def _set_progress_text(self, text):
        self._pending_progress_text = text
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "_doSetProgressText", None, False)

    def _doSetProgressText(self):
        if hasattr(self, "_pending_progress_text"):
            self._progress_text.setStringValue_(self._pending_progress_text)

    # ── Actions ──────────────────────────────────────────────

    @objc.typedSelector(b'v@:@')
    def onStart_(self, sender):
        try:
            logger.info("onStart_")
            model_dir = get_active_model_dir()
            if not model_dir:
                self._do_import_model_and_start()
                return

            gw, wk = self._read_ports_from_ui()
            gw, wk, port_notes = resolve_gateway_worker_ports(gw, wk)
            self._port = gw
            self._worker_port = wk
            self._gw_field.setStringValue_(str(gw))
            self._wk_field.setStringValue_(str(wk))
            self._url_label.setStringValue_(f"https://localhost:{self._port}")
            save_config(model_dir, self._port, self._worker_port)
            for line in port_notes:
                self._append_log(line)

            if not find_llama_server():
                self._append_log(
                    f"\nError: llama-server not found.\n"
                    f"  cd {_REPO_ROOT}\n"
                    "  cmake -B build && cmake --build build --target llama-server -j\n\n")
                self._set_state(ServiceState.ERROR)
                return

            check = check_model_dir(model_dir)
            if not check["valid"]:
                self._append_log("\nModel issues:\n")
                for m in check["missing"]:
                    self._append_log(f"  - {m}\n")
                self._append_log("Continuing anyway…\n\n")

            self._set_state(ServiceState.STARTING)
            self._append_log(f"\n{'=' * 50}\n")
            self._append_log(f"Starting services…  Model: {get_model_display_name(model_dir)}\n")
            self._append_log(f"{'=' * 50}\n\n")
            threading.Thread(target=self._do_start, daemon=True).start()
        except Exception:
            logger.exception("onStart_ failed")
            self._append_log(f"\nUnexpected error — see {_APP_LOG_PATH}\n")

    def _do_import_model_and_start(self):
        panel = NSOpenPanel.openPanel()
        panel.setCanChooseDirectories_(True)
        panel.setCanChooseFiles_(False)
        panel.setAllowsMultipleSelection_(False)
        panel.setPrompt_("Import Model")
        panel.setMessage_("Select a GGUF model directory")
        if panel.runModal() != 1:
            return
        src = str(panel.URLs()[0].path())
        path, err = import_model_link(src)
        if err:
            self._append_log(f"\nImport error: {err}\n")
            return
        self._append_log(f"\nModel imported: {src}\n")
        self._update_ui()
        self.onStart_(None)

    def _do_start(self):
        try:
            logger.info("_do_start worker=%s gateway=%s", self._worker_port, self._port)
            if _LOG_PATH.exists():
                _LOG_PATH.write_text("")
            self._log_file = open(_LOG_PATH, "a", encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(_SERVER_DIR)
            # Prevent system HTTP proxy (e.g. Clash/V2Ray on 127.0.0.1:7890) from
            # intercepting loopback IPC between gateway / worker / llama-server.
            # macOS ExceptionsList doesn't reliably bypass 'localhost' host name
            # in Python urllib/httpx; force all children to disable proxies entirely.
            for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                       "http_proxy", "https_proxy", "all_proxy"):
                env.pop(_k, None)
            env["NO_PROXY"] = "*"
            env["no_proxy"] = "*"

            # ---------------------------------------------------------------
            # 防止系统级 HTTP 代理劫持 loopback。
            # 常见场景：用户开着 Clash/V2Ray，系统代理指向 127.0.0.1:7890。
            # 子进程里 httpx / urllib 默认会读环境变量 HTTP_PROXY/HTTPS_PROXY，
            # 于是把本该直连 worker 的 http://127.0.0.1:22700/ 发到 Clash
            # 被回 502。显式清空所有 proxy 相关变量，并强制 NO_PROXY=*。
            # ---------------------------------------------------------------
            for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                       "http_proxy", "https_proxy", "all_proxy"):
                env.pop(_k, None)
            env["NO_PROXY"] = "*"
            env["no_proxy"] = "*"

            self._set_progress_text("Loading models…")
            self._start_log_tail()

            self._worker_proc = subprocess.Popen(
                [sys.executable, str(_SERVER_DIR / "worker.py"),
                 "--port", str(self._worker_port), "--gpu-id", "0", "--worker-index", "0"],
                env=env, cwd=str(_SERVER_DIR),
                stdout=self._log_file, stderr=subprocess.STDOUT)

            if not self._wait_health(f"http://127.0.0.1:{self._worker_port}", 300):
                if self._worker_proc and self._worker_proc.poll() is not None:
                    self._append_log(f"\nWorker exited with code {self._worker_proc.returncode}\n")
                else:
                    self._append_log("\nWorker startup timeout (300s)\n")
                self._set_state(ServiceState.ERROR)
                return

            self._append_log("\nWorker ready!\n")
            self._set_progress_text("Starting gateway…")

            # Gateway 默认用 HTTPS（自签名证书）。
            # HTTP 模式下浏览器会禁掉麦克风/摄像头（非 secure context），
            # 移动端通过局域网 IP 访问时录音/视频完全不可用，所以必须 HTTPS。
            self._gateway_proc = subprocess.Popen(
                [sys.executable, str(_SERVER_DIR / "gateway.py"),
                 "--port", str(self._port),
                 # 保持 HTTPS: 菜单栏 app 场景需要手机浏览器走 secure context,
                 # 否则麦克风/摄像头被浏览器禁用. (注: upstream 曾一度改为 --http,
                 # 但与下面 "https://{host}:{port}" 的启动日志和 mobile 使用场景
                 # 不兼容, 这里恢复 HTTPS.)
                 "--workers", f"127.0.0.1:{self._worker_port}"],
                env=env, cwd=str(_SERVER_DIR),
                stdout=self._log_file, stderr=subprocess.STDOUT)

            time.sleep(3)
            if self._gateway_proc.poll() is not None:
                self._append_log("\nGateway exited unexpectedly\n")
                self._set_state(ServiceState.ERROR)
                return

            self._set_state(ServiceState.RUNNING)
            lan_ip = self._get_lan_ip()
            lan_line = (f"  On your phone (same Wi-Fi): https://{lan_ip}:{self._port}\n"
                        if lan_ip else "")
            self._append_log(
                f"\n{'=' * 50}\n"
                f"Server running at https://localhost:{self._port}\n"
                f"{lan_line}"
                f"{'=' * 50}\n\n"
                "Modes: Turn-based · Omni Duplex · Audio Duplex · Half-Duplex · Mobile\n"
                "Note: 自签名证书，浏览器首次打开会提示不安全，点『高级 → 继续访问』即可。\n"
                "Click 'Open Web UI' or visit the URL above.\n")
        except Exception as e:
            logger.exception("_do_start failed")
            self._set_state(ServiceState.ERROR)
            self._append_log(f"\nError: {e}\n")
            import traceback
            self._append_log(traceback.format_exc())

    def _wait_health(self, url, timeout=300):
        # 注意：不能直接用 urllib.request.urlopen()。macOS 上即便
        # ExceptionsList 含 localhost / 127.0.0.1，Python 的 urllib 仍然会
        # 读 $HTTP_PROXY 把请求走代理。这里用 ProxyHandler({}) 强制不走
        # 任何代理，和 curl 行为对齐。
        import urllib.request
        # Build an opener that ignores system HTTP proxy, so loopback probes
        # aren't intercepted by local HTTP proxies (Clash/V2Ray etc.).
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(timeout // 2):
            try:
                resp = opener.open(f"{url}/health", timeout=3)
                if resp.status == 200:
                    data = json.loads(resp.read())
                    if data.get("model_loaded", True):
                        return True
            except Exception:
                pass
            if self._worker_proc and self._worker_proc.poll() is not None:
                return False
            time.sleep(2)
        return False

    @objc.python_method
    def _is_private_ipv4(self, ip: str) -> bool:
        """只接受常见家庭/办公局域网段，排除 VPN 假 IP (如 198.18.x Clash FakeIP)
        和 CGNAT / link-local / loopback。"""
        try:
            p = [int(x) for x in ip.split(".")]
            if len(p) != 4 or any(x < 0 or x > 255 for x in p):
                return False
            a, b = p[0], p[1]
            if a == 10:
                return True
            if a == 172 and 16 <= b <= 31:
                return True
            if a == 192 and b == 168:
                return True
            return False
        except Exception:
            return False

    @objc.python_method
    def _get_lan_ip(self):
        """尽量拿一个手机能连到的局域网 IP。探测顺序：
        1. networksetup 找出 Wi-Fi / Ethernet 的 BSD 接口（如 en0），取其 IPv4
        2. 兜底：ifconfig 扫所有非 VPN/虚拟接口
        3. 最差：UDP connect 8.8.8.8 拿默认出网 IP

        开 VPN（Clash / V2Ray / ExpressVPN 等）时，connect(8.8.8.8) 会被 VPN
        劫持路由，拿到 utun* 上的假 IP（如 Clash FakeIP 段 198.18.0.0/15），
        手机在同一 Wi-Fi 下根本连不到，所以不能用作首选方案。
        """
        import subprocess, re, socket

        # --- 1) networksetup 枚举硬件端口，取 Wi-Fi / Ethernet ---
        preferred_ifs = []
        try:
            out = subprocess.check_output(
                ["networksetup", "-listallhardwareports"],
                text=True, stderr=subprocess.DEVNULL, timeout=2)
            port = dev = None
            for ln in out.splitlines() + [""]:
                if ln.startswith("Hardware Port:"):
                    port = ln.split(":", 1)[1].strip()
                elif ln.startswith("Device:"):
                    dev = ln.split(":", 1)[1].strip()
                elif not ln.strip():
                    # 块结束，提交
                    if port and dev and any(k in port for k in
                            ("Wi-Fi", "AirPort", "Ethernet", "Thunderbolt")):
                        preferred_ifs.append(dev)
                    port = dev = None
        except Exception:
            pass

        for ifn in preferred_ifs:
            try:
                ip = subprocess.check_output(
                    ["ipconfig", "getifaddr", ifn],
                    text=True, stderr=subprocess.DEVNULL, timeout=1).strip()
                if ip and self._is_private_ipv4(ip):
                    return ip
            except Exception:
                continue

        # --- 2) ifconfig 兜底，排除虚拟 / VPN 接口 ---
        BAD_PREFIX = (
            "utun", "ipsec", "ppp", "tap", "tun",
            "bridge", "awdl", "llw", "lo", "gif", "stf", "anpi", "ap",
        )
        try:
            out = subprocess.check_output(
                ["ifconfig"], text=True, stderr=subprocess.DEVNULL, timeout=2)
            current_if = None
            for ln in out.splitlines():
                if re.match(r"^[a-zA-Z]", ln):
                    current_if = ln.split(":", 1)[0]
                    continue
                m = re.search(r"\binet (\d+\.\d+\.\d+\.\d+)\b", ln)
                if not m or not current_if:
                    continue
                if current_if.startswith(BAD_PREFIX):
                    continue
                ip = m.group(1)
                if self._is_private_ipv4(ip):
                    return ip
        except Exception:
            pass

        # --- 3) 最差兜底：UDP connect ---
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.3)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            if ip and self._is_private_ipv4(ip):
                return ip
        except Exception:
            pass
        return None

    @objc.typedSelector(b'v@:@')
    def onStop_(self, sender):
        try:
            logger.info("onStop_")
            self._set_state(ServiceState.STOPPING)
            self._append_log("\nStopping services…\n")
            threading.Thread(target=self._do_stop, daemon=True).start()
        except Exception:
            logger.exception("onStop_ failed")

    def _do_stop(self):
        logger.info("_do_stop")
        self._stop_log_tail()
        for name, proc in [("Gateway", self._gateway_proc), ("Worker", self._worker_proc)]:
            if proc and proc.poll() is None:
                self._append_log(f"  Stopping {name} (pid={proc.pid})…\n")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self._worker_proc = None
        self._gateway_proc = None
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        self._set_state(ServiceState.STOPPED)
        self._append_log("Server stopped.\n\n")

    @objc.typedSelector(b'v@:@')
    def onOpenBrowser_(self, sender):
        try:
            logger.info("onOpenBrowser_")
            webbrowser.open(f"https://localhost:{self._port}")
        except Exception:
            logger.exception("onOpenBrowser_ failed")

    @objc.typedSelector(b'v@:@')
    def onVisionBackendChanged_(self, sender):
        try:
            idx = sender.indexOfSelectedItem()
            vb = {0: "auto", 1: "metal", 2: "coreml"}.get(idx, "auto")
            from server.model_hub import load_comni_config, save_comni_config
            cfg = load_comni_config()
            cfg["vision_backend"] = vb
            save_comni_config(cfg)
            self._append_log(f"Vision backend → {vb} (restart service to apply)\n")
        except Exception:
            logger.exception("onVisionBackendChanged_ failed")

    @objc.typedSelector(b'v@:@')
    def onContextSizeChanged_(self, sender):
        try:
            idx = sender.indexOfSelectedItem()
            if idx < 0 or idx >= len(CTX_SIZE_CHOICES):
                return
            cs = CTX_SIZE_CHOICES[idx]
            from server.model_hub import load_comni_config, save_comni_config
            cfg = load_comni_config()
            cfg["ctx_size"] = cs
            save_comni_config(cfg)
            self._append_log(
                f"Context size → {cs} ({cs // 1024}K) — restart service to apply\n")
        except Exception:
            logger.exception("onContextSizeChanged_ failed")

    @objc.typedSelector(b'v@:@')
    def onDownloadSourceChanged_(self, sender):
        try:
            idx = sender.indexOfSelectedItem()
            if idx < 0 or idx >= len(DOWNLOAD_SOURCE_CHOICES):
                return
            key, label = DOWNLOAD_SOURCE_CHOICES[idx]
            from server.model_hub import load_comni_config, save_comni_config
            cfg = load_comni_config()
            cfg["download_source"] = key
            save_comni_config(cfg)
            self._append_log(
                f"Download source → {label} (applies to next download)\n")
        except Exception:
            logger.exception("onDownloadSourceChanged_ failed")

    @objc.typedSelector(b'v@:@')
    def onCopyURL_(self, sender):
        try:
            url = f"https://localhost:{self._port}"
            subprocess.run(["pbcopy"], input=url.encode(), check=True)
            self._append_log(f"URL copied: {url}\n")
        except Exception:
            logger.exception("onCopyURL_ failed")

    @objc.python_method
    def _mobile_url(self):
        """移动端 URL。IP 会动态重新探测，保证切换 Wi-Fi 后也正确。"""
        lan_ip = self._get_lan_ip()
        if not lan_ip:
            return None
        return f"https://{lan_ip}:{self._port}/mobile/"

    @objc.typedSelector(b'v@:@')
    def onShowMobileQR_(self, sender):
        """弹出二维码窗口，手机扫码即可打开移动端。"""
        try:
            logger.info("onShowMobileQR_")
            url = self._mobile_url()
            if not url:
                self._alert("No LAN IP detected",
                            "未探测到可用的局域网 IP，请确认 Mac 已连接 Wi-Fi。")
                return
            # 刷新一下 UI 上的 Mobile URL，防止 IP 漂移后对不上
            try:
                if hasattr(self, "_mobile_url_label") and self._mobile_url_label is not None:
                    self._mobile_url_label.setStringValue_(url)
            except Exception:
                pass
            self._present_qr_window(url)
            self._append_log(f"Mobile QR shown for: {url}\n")
        except Exception:
            logger.exception("onShowMobileQR_ failed")

    @objc.python_method
    def _alert(self, title: str, info: str):
        try:
            a = NSAlert.alloc().init()
            a.setAlertStyle_(NSAlertStyleInformational)
            a.setMessageText_(title)
            a.setInformativeText_(info)
            a.addButtonWithTitle_("OK")
            a.runModal()
        except Exception:
            logger.exception("_alert failed")

    @objc.python_method
    def _make_qr_image(self, text: str, px: int = 320):
        """用系统自带 CIQRCodeGenerator 生成 NSImage（无外部依赖）。"""
        from Foundation import NSData
        from Quartz import (
            CIImage, CIFilter, CIContext, CIVector,
        )
        data = NSData.dataWithBytes_length_(text.encode("utf-8"), len(text.encode("utf-8")))
        f = CIFilter.filterWithName_("CIQRCodeGenerator")
        f.setValue_forKey_(data, "inputMessage")
        # 高纠错 H：被 Safari/微信扫描时更宽容
        f.setValue_forKey_("H", "inputCorrectionLevel")
        ci = f.outputImage()
        if ci is None:
            return None
        ext = ci.extent()
        # 放大到 px × px（最近邻采样，保持方块边缘锐利）
        scale = float(px) / max(ext.size.width, 1.0)
        from Quartz import CGAffineTransformMakeScale
        ci = ci.imageByApplyingTransform_(CGAffineTransformMakeScale(scale, scale))
        ctx = CIContext.contextWithOptions_(None)
        cg = ctx.createCGImage_fromRect_(ci, ci.extent())
        if cg is None:
            return None
        from AppKit import NSBitmapImageRep
        from Foundation import NSMakeSize
        rep = NSBitmapImageRep.alloc().initWithCGImage_(cg)
        img = NSImage.alloc().initWithSize_(NSMakeSize(px, px))
        img.addRepresentation_(rep)
        return img

    @objc.python_method
    def _present_qr_window(self, url: str):
        """独立窗口展示二维码 + URL + Copy 按钮。再次调用会复用现有窗口。"""
        from AppKit import NSImageView, NSImageScaleProportionallyUpOrDown
        qr_w, qr_h = 340, 440
        if getattr(self, "_qr_window", None) is None:
            rect = NSMakeRect(0, 0, qr_w, qr_h)
            mask = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable)
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, mask, NSBackingStoreBuffered, False)
            win.setTitle_("Mobile — Scan to Open")
            win.setReleasedWhenClosed_(False)
            cv = win.contentView()

            # QR 图片区域
            iv = NSImageView.alloc().initWithFrame_(NSMakeRect(10, 110, 320, 320))
            try:
                iv.setImageScaling_(NSImageScaleProportionallyUpOrDown)
            except Exception:
                pass
            cv.addSubview_(iv)
            self._qr_image_view = iv

            # URL 文本
            lbl = self._make_label("", x=10, y=70, w=320, h=32,
                                    size=11, color=NSColor.secondaryLabelColor())
            try:
                lbl.cell().setWraps_(True)
                lbl.cell().setScrollable_(False)
                lbl.setAlignment_(NSTextAlignmentCenter)
            except Exception:
                pass
            cv.addSubview_(lbl)
            self._qr_url_label = lbl

            tip = self._make_label(
                "Scan with your phone (same Wi-Fi). 首次会提示证书不安全，点『继续』即可。",
                x=10, y=38, w=320, h=30, size=10,
                color=NSColor.tertiaryLabelColor())
            try:
                tip.cell().setWraps_(True)
                tip.setAlignment_(NSTextAlignmentCenter)
            except Exception:
                pass
            cv.addSubview_(tip)

            cv.addSubview_(self._make_button(
                "Copy URL", x=120, y=8, w=100, h=24, action="onCopyMobileURL:"))

            win.center()
            self._qr_window = win

        img = self._make_qr_image(url, px=320)
        if img is not None:
            self._qr_image_view.setImage_(img)
        self._qr_url_label.setStringValue_(url)
        self._qr_current_url = url
        self._qr_window.makeKeyAndOrderFront_(None)
        # 把窗口带到前台
        try:
            NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass

    @objc.typedSelector(b'v@:@')
    def onCopyMobileURL_(self, sender):
        try:
            url = getattr(self, "_qr_current_url", None) or self._mobile_url() or ""
            if url:
                subprocess.run(["pbcopy"], input=url.encode(), check=True)
                self._append_log(f"Mobile URL copied: {url}\n")
        except Exception:
            logger.exception("onCopyMobileURL_ failed")

    # ── About Window ─────────────────────────────────────────
    #
    # 单独的 "About Comni" 窗口，集中展示项目相关链接：
    #   - 本 app 的 cpp 引擎源码（llama.cpp-omni）
    #   - 模型仓库（MiniCPM-o on GitHub / HF / ModelScope）
    #   - 上游引擎 llama.cpp（致谢）
    #
    # 入口点：菜单栏下拉里的 "About Comni…"。
    # 链接通过 setTag_(idx) + self._about_link_urls 列表查表打开，
    # 避免每条链接写一个独立的 selector。

    def _present_about_window(self):
        """About 窗口：项目链接、模型仓库链接、致谢。复用同一窗口。"""
        aw_w, aw_h = 380, 420
        if getattr(self, "_about_window", None) is None:
            rect = NSMakeRect(0, 0, aw_w, aw_h)
            mask = (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable)
            win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
                rect, mask, NSBackingStoreBuffered, False)
            win.setTitle_("About Comni")
            win.setReleasedWhenClosed_(False)
            cv = win.contentView()

            # tag → URL 表。顺序就是按钮 setTag_ 的 index，
            # onOpenAboutLink_ 用 sender.tag() 从这里查。
            self._about_link_urls = [
                "https://github.com/tc-mb/llama.cpp-omni",
                "https://github.com/OpenBMB/MiniCPM-o",
                "https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf",
                "https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf",
                "https://github.com/ggml-org/llama.cpp",
            ]

            y = aw_h - 36

            # ── Header: 名字 / slogan / 版本号 ──
            cv.addSubview_(self._make_label(
                "Comni", x=0, y=y, w=aw_w, h=28,
                size=22, bold=True, align=NSTextAlignmentCenter))
            y -= 26
            cv.addSubview_(self._make_label(
                "Omni inference in C/C++", x=0, y=y, w=aw_w, h=18,
                size=11.5, color=NSColor.secondaryLabelColor(),
                align=NSTextAlignmentCenter))
            y -= 16
            cv.addSubview_(self._make_label(
                _format_version_tag(), x=0, y=y, w=aw_w, h=14,
                size=10, color=NSColor.tertiaryLabelColor(),
                align=NSTextAlignmentCenter))
            y -= 18

            # ── 上分割线 ──
            sep = NSBox.alloc().initWithFrame_(NSMakeRect(20, y, aw_w - 40, 1))
            sep.setBoxType_(2)  # NSBoxSeparator
            cv.addSubview_(sep)
            y -= 18

            # 行布局尺寸
            BTN_W, BTN_H = 130, 22
            ROW_H = 26
            SEC_GAP = 6

            def _section(title, yy):
                cv.addSubview_(self._make_label(
                    title, x=20, y=yy, w=aw_w - 40, h=18,
                    size=12, bold=True, color=NSColor.secondaryLabelColor()))
                return yy - 22

            def _link_row(text, btn_title, tag, yy):
                cv.addSubview_(self._make_label(
                    text, x=28, y=yy + 2, w=aw_w - 28 - BTN_W - 24, h=20, size=12))
                btn = self._make_button(
                    btn_title, x=aw_w - 20 - BTN_W, y=yy, w=BTN_W, h=BTN_H,
                    action="onOpenAboutLink:")
                btn.setTag_(tag)
                cv.addSubview_(btn)
                return yy - ROW_H

            # ── 本项目 ──
            y = _section("This app", y)
            y = _link_row("llama.cpp-omni", "GitHub →", 0, y)
            y -= SEC_GAP

            # ── 模型 ──
            y = _section("Model — MiniCPM-o 4.5", y)
            y = _link_row("Project page", "GitHub →", 1, y)
            y = _link_row("Model weights (GGUF)", "Hugging Face →", 2, y)
            y = _link_row("Model weights (GGUF)", "ModelScope →", 3, y)
            y -= SEC_GAP

            # ── 上游引擎致谢 ──
            y = _section("Built on", y)
            y = _link_row("llama.cpp (upstream)", "GitHub →", 4, y)

            # ── 底部 footer ──
            sep2 = NSBox.alloc().initWithFrame_(NSMakeRect(20, 30, aw_w - 40, 1))
            sep2.setBoxType_(2)
            cv.addSubview_(sep2)
            cv.addSubview_(self._make_label(
                "MIT License",
                x=0, y=10, w=aw_w, h=14,
                size=10, color=NSColor.tertiaryLabelColor(),
                align=NSTextAlignmentCenter))

            win.center()
            self._about_window = win

        self._about_window.makeKeyAndOrderFront_(None)
        try:
            NSApp.activateIgnoringOtherApps_(True)
        except Exception:
            pass

    @objc.typedSelector(b'v@:@')
    def onOpenAboutLink_(self, sender):
        try:
            idx = sender.tag()
            urls = getattr(self, "_about_link_urls", [])
            if 0 <= idx < len(urls):
                webbrowser.open(urls[idx])
        except Exception:
            logger.exception("onOpenAboutLink_ failed")

    @objc.typedSelector(b'v@:@')
    def onOpenLog_(self, sender):
        try:
            if _LOG_PATH.exists():
                subprocess.Popen(["open", str(_LOG_PATH)])
            else:
                self._append_log("No log file yet.\n")
        except Exception:
            logger.exception("onOpenLog_ failed")

    @objc.typedSelector(b'v@:@')
    def onClearLog_(self, sender):
        try:
            storage = self._log_textview.textStorage()
            storage.beginEditing()
            storage.mutableString().setString_("")
            storage.endEditing()
        except Exception:
            logger.exception("onClearLog_ failed")

    # ── Model Actions ────────────────────────────────────────

    @objc.typedSelector(b'v@:@')
    def onManageModels_(self, sender):
        try:
            logger.info("onManageModels_")
            self._build_model_manager()
        except Exception:
            logger.exception("onManageModels_ failed")

    @objc.typedSelector(b'v@:@')
    def onImportModel_(self, sender):
        try:
            logger.info("onImportModel_")
            panel = NSOpenPanel.openPanel()
            panel.setCanChooseDirectories_(True)
            panel.setCanChooseFiles_(False)
            panel.setAllowsMultipleSelection_(False)
            panel.setPrompt_("Import")
            panel.setMessage_("Select a GGUF model directory")
            if panel.runModal() == 1:
                src = str(panel.URLs()[0].path())
                path, err = import_model_link(src)
                if err:
                    logger.warning("Import error: %s", err)
                    self._append_log(f"Import error: {err}\n")
                else:
                    logger.info("Model imported: %s → %s", src, path)
                    self._append_log(f"Model imported: {src}\n")
                    check = check_model_dir(path)
                    if check["valid"] and check.get("llm"):
                        save_config(path, self._port, self._worker_port)
                    self._update_ui()
                    self._refresh_model_manager_content()
        except Exception:
            logger.exception("onImportModel_ failed")

    @objc.typedSelector(b'v@:@')
    def onOpenModelsFolder_(self, sender):
        try:
            _MODELS_HOME.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["open", str(_MODELS_HOME)])
        except Exception:
            logger.exception("onOpenModelsFolder_ failed")

    @objc.typedSelector(b'v@:@')
    def onRemoveModel_(self, sender):
        try:
            logger.info("onRemoveModel_")
            models = scan_models()
            if not models:
                return
            target = models[-1] if len(models) == 1 else models[-1]
            p = Path(target["path"])
            if p.is_symlink():
                p.unlink()
                logger.info("Removed symlink: %s", target["name"])
                self._append_log(f"Removed link: {target['name']}\n")
            else:
                alert = NSAlert.alloc().init()
                alert.setMessageText_("Remove model?")
                alert.setInformativeText_(
                    f"'{target['name']}' is not a symlink.\n"
                    "Only the link in ~/.comni/models/ will be removed.\n"
                    "The original files will NOT be deleted.")
                alert.addButtonWithTitle_("Remove")
                alert.addButtonWithTitle_("Cancel")
                alert.setAlertStyle_(NSAlertStyleWarning)
                if alert.runModal() == 1000:
                    import shutil
                    shutil.rmtree(str(p), ignore_errors=True)
                    logger.info("Removed directory: %s", target["name"])
                    self._append_log(f"Removed: {target['name']}\n")
                else:
                    return
            self._update_ui()
            self._refresh_model_manager_content()
        except Exception:
            logger.exception("onRemoveModel_ failed")

    # ── Download / Verify Actions ────────────────────────────

    @objc.typedSelector(b'v@:@')
    def onDownloadModel_(self, sender):
        try:
            from server.model_hub import (
                list_available_models, ModelDownloader, DownloadProgress,
                get_hf_mirror, save_comni_config, load_comni_config,
            )
            logger.info("onDownloadModel_")
            if self._downloader and self._downloader._thread and self._downloader._thread.is_alive():
                self._append_log("A download is already in progress.\n")
                return

            tag = sender.tag() if sender else 0
            spec = None
            for s in list_available_models():
                if (hash(s["id"]) & 0x7FFFFFFF) == tag:
                    spec = s
                    break
            if not spec:
                self._append_log("Model spec not found.\n")
                return

            quant_popups = getattr(self, "_quant_popups", {})
            popup = quant_popups.get(spec["id"])
            quant_idx = popup.indexOfSelectedItem() if popup else 0
            variants = spec.get("llm_variants", [])
            if quant_idx < len(variants):
                quant = variants[quant_idx]["quant"]
            else:
                quant = variants[0]["quant"]

            size_gb = sum(
                e.get("size", 0) for e in spec.get("required_files", [])
            ) + sum(
                e.get("size", 0) for e in spec.get("optional_files", [])
            )
            for v in variants:
                if v["quant"] == quant:
                    size_gb += v.get("size", 0)
                    break
            size_gb = round(size_gb / (1024**3), 1)

            self._append_log(
                f"\nDownloading {spec['display_name']} ({quant})  ~{size_gb} GB\n")

            if self._dl_progress_bar:
                self._dl_progress_bar.setHidden_(False)
                self._dl_progress_bar.setDoubleValue_(0.0)
            if self._dl_progress_label:
                self._dl_progress_label.setStringValue_("Preparing download…")

            def on_progress(prog: DownloadProgress):
                self._pending_dl_progress = prog
                self.performSelectorOnMainThread_withObject_waitUntilDone_(
                    "_doUpdateDLProgress", None, False)

            self._downloader = ModelDownloader(spec, quant)
            self._downloader.start(progress_cb=on_progress)

        except Exception:
            logger.exception("onDownloadModel_ failed")
            self._append_log("Download failed — see app log.\n")

    @staticmethod
    def _fmt_speed(bps: float) -> str:
        if bps >= 1024 * 1024:
            return f"{bps / (1024*1024):.1f} MB/s"
        if bps >= 1024:
            return f"{bps / 1024:.0f} KB/s"
        return f"{bps:.0f} B/s"

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        if n >= 1024**3:
            return f"{n / 1024**3:.1f} GB"
        if n >= 1024**2:
            return f"{n / 1024**2:.0f} MB"
        return f"{n / 1024:.0f} KB"

    def _doUpdateDLProgress(self):
        prog = getattr(self, "_pending_dl_progress", None)
        if not prog:
            return
        if self._dl_progress_label:
            if prog.status == "downloading":
                pct = 0
                if prog.bytes_total > 0:
                    pct = min(int(prog.bytes_done * 100 / prog.bytes_total), 99)
                speed_str = self._fmt_speed(prog.speed_bps) if prog.speed_bps > 0 else "…"
                done_str = self._fmt_bytes(prog.bytes_done)
                total_str = self._fmt_bytes(prog.bytes_total) if prog.bytes_total > 0 else "?"
                workers = prog.active_workers if prog.active_workers > 0 else 1
                self._dl_progress_label.setStringValue_(
                    f"{prog.file_index}/{prog.total_files} done  "
                    f"{done_str}/{total_str}  {pct}%  {speed_str}"
                    f"  ×{workers}" if workers > 1 else
                    f"{prog.file_index}/{prog.total_files} done  "
                    f"{done_str}/{total_str}  {pct}%  {speed_str}")
                if self._dl_progress_bar:
                    self._dl_progress_bar.setDoubleValue_(float(pct))
            elif prog.status == "verifying":
                self._dl_progress_label.setStringValue_("Verifying files…")
            elif prog.status == "done":
                self._dl_progress_label.setStringValue_("Download complete ✓")
                if self._dl_progress_bar:
                    self._dl_progress_bar.setDoubleValue_(100.0)
                self._append_log("Download and verification complete!\n")
                self._update_ui()
                self._refresh_model_manager_content()
            elif prog.status == "error":
                self._dl_progress_label.setStringValue_(f"Error: {prog.error}")
                self._append_log(f"Download error: {prog.error}\n")
                if self._dl_progress_bar:
                    self._dl_progress_bar.setHidden_(True)

    @objc.typedSelector(b'v@:@')
    def onVerifyModel_(self, sender):
        try:
            from server.model_hub import verify_model, match_spec_by_dir
            logger.info("onVerifyModel_")
            model_dir = get_active_model_dir()
            if not model_dir:
                self._append_log("No active model to verify.\n")
                return
            vr = verify_model(model_dir)
            if vr.complete:
                self._append_log(
                    f"Verification OK: {len(vr.verified)} files verified.\n")
            else:
                self._append_log(f"Verification FAILED:\n")
                for m in vr.missing:
                    self._append_log(f"  Missing: {m}\n")
                for m in vr.size_mismatch:
                    self._append_log(f"  Size mismatch: {m}\n")
        except Exception:
            logger.exception("onVerifyModel_ failed")

    @objc.typedSelector(b'v@:@')
    def onCancelDownload_(self, sender):
        if self._downloader:
            self._downloader.cancel()
            self._append_log("Download cancelled.\n")
            if self._dl_progress_bar:
                self._dl_progress_bar.setHidden_(True)
            if self._dl_progress_label:
                self._dl_progress_label.setStringValue_("Cancelled")

    # ── Menu Bar Actions ─────────────────────────────────────

    @objc.typedSelector(b'v@:@')
    def menuStart_(self, sender): self.onStart_(sender)
    @objc.typedSelector(b'v@:@')
    def menuStop_(self, sender): self.onStop_(sender)
    @objc.typedSelector(b'v@:@')
    def menuOpenBrowser_(self, sender): self.onOpenBrowser_(sender)
    @objc.typedSelector(b'v@:@')
    def menuShowWindow_(self, sender):
        self._window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
    @objc.typedSelector(b'v@:@')
    def menuAbout_(self, sender):
        try:
            self._present_about_window()
        except Exception:
            logger.exception("menuAbout_ failed")
    @objc.typedSelector(b'v@:@')
    def menuQuit_(self, sender):
        logger.info("menuQuit_ — shutting down")
        try:
            if self._state == ServiceState.RUNNING:
                self._do_stop()
        except Exception:
            logger.exception("cleanup failed during quit")
        NSApp.terminate_(None)

    # ── Window Delegate ──────────────────────────────────────

    def windowShouldClose_(self, window):
        if window == self._model_manager_window:
            window.orderOut_(None)
            return False
        window.orderOut_(None)
        return False

    def applicationShouldTerminateAfterLastWindowClosed_(self, app):
        return False

    def applicationShouldHandleReopen_hasVisibleWindows_(self, app, flag):
        if not flag:
            self._window.makeKeyAndOrderFront_(None)
        return True


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = AppDelegate.alloc().init()
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
