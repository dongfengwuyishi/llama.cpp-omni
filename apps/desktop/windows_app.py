#!/usr/bin/env python3
"""Comni — Windows Desktop App

基于 PySide6 的原生 Windows 应用，功能与 macOS 版 menubar_app.py 对齐：
    - 主窗口：服务状态、模型卡片、服务 URL、端口设置、实时日志
    - 系统托盘图标（右键菜单可快速 Start/Stop/打开 Web UI/退出）
    - 模型管理窗口（导入/删除/校验/HF 下载）
    - 后端自动检测：有 NVIDIA GPU 走 CUDA，否则走 CPU
    - App 数据目录：%APPDATA%\\Comni\\

该脚本既可以用系统 Python 直接运行（开发模式），
也可以被 PyInstaller 打包成 Comni.exe（发布模式）。
"""

from __future__ import annotations

import os
import sys
import json
import time
import socket
import signal
import shutil
import ctypes
import logging
import logging.handlers
import subprocess
import threading
import webbrowser
import traceback
from pathlib import Path
from typing import Optional, List


# ------------------------------------------------------------
# Early crash log — written even if logging module fails to init.
# Any uncaught exception during import/boot goes here.
# ------------------------------------------------------------
def _early_log(msg: str) -> None:
    try:
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        p = Path(appdata) / "Comni"
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "comni_boot.log", "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


# ------------------------------------------------------------
# PyInstaller windowed-mode safeguard:
#   When console=False, sys.stdout / sys.stderr are None. The logging
#   module's StreamHandler then raises AttributeError on first write,
#   and since logger init is at module top-level the whole process
#   exits silently before any line of our code runs. Replacing None
#   with a devnull sink is the standard fix.
# ------------------------------------------------------------
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

_early_log(f"boot — exe={sys.executable}  frozen={getattr(sys, 'frozen', False)}  "
           f"_MEIPASS={getattr(sys, '_MEIPASS', '(none)')}")


# ------------------------------------------------------------
# Path resolution — dev mode vs. PyInstaller frozen mode
# ------------------------------------------------------------
# When frozen by PyInstaller, we ship:
#
#   Comni/
#     Comni.exe
#     _internal/                        <- PyInstaller runtime
#     resources/
#       apps/
#         server/, assets/, frontend/, desktop/
#       build/bin/Release/llama-server.exe
#       python-embed/python.exe         <- embedded Python for running services
#
def _resolve_app_root() -> tuple[Path, Path, bool]:
    """Return (apps_root, repo_root, frozen?).

    PyInstaller onedir layout:
      Comni.exe                          <- sys.executable
      _internal/                         <- sys._MEIPASS (PyInstaller runtime)
        resources/
          apps/{server, assets, frontend, desktop}
          build/bin/Release/llama-server.exe
    """
    frozen = bool(getattr(sys, "frozen", False))
    if frozen:
        exe_dir = Path(sys.executable).resolve().parent
        # PyInstaller exposes the bundle dir via sys._MEIPASS (onedir: _internal/)
        meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
        # Search order: _MEIPASS/resources → exe_dir/resources → exe_dir
        for base in (meipass / "resources", exe_dir / "resources", exe_dir):
            if (base / "apps").is_dir():
                return base / "apps", base, True
        # Last resort — something isn't laid out right, point at _MEIPASS anyway
        return meipass / "apps", meipass, True

    here = Path(__file__).resolve().parent
    apps_root = here.parent
    repo_root = apps_root.parent
    return apps_root, repo_root, False


try:
    _APPS_ROOT, _REPO_ROOT, _FROZEN = _resolve_app_root()
except Exception as _e:
    _early_log(f"_resolve_app_root failed: {_e}\n{traceback.format_exc()}")
    raise
_SERVER_DIR = _APPS_ROOT / "server"
_DESKTOP_DIR = _APPS_ROOT / "desktop"
_CONFIG_PATH = _SERVER_DIR / "config.json"
_CONFIG_EXAMPLE = _SERVER_DIR / "config.example.json"

# Make `from server.model_hub import ...` work
if str(_APPS_ROOT) not in sys.path:
    sys.path.insert(0, str(_APPS_ROOT))

# ------------------------------------------------------------
# Windows-only: user directories
# ------------------------------------------------------------
_APPDATA = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
_APP_SUPPORT = _APPDATA / "Comni"
_APP_SUPPORT.mkdir(parents=True, exist_ok=True)

_LOG_PATH = _APP_SUPPORT / "comni_service.log"
_APP_LOG_PATH = _APP_SUPPORT / "comni_app.log"

_COMNI_HOME = Path.home() / ".comni"
_MODELS_HOME = _COMNI_HOME / "models"

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Ports
# ------------------------------------------------------------
DEFAULT_WORKER_BASE_PORT = 22700
DEFAULT_GATEWAY_PORT = 8006
LEGACY_WORKER_PORT = 22400


# ------------------------------------------------------------
# PySide6 imports — done after early path setup
# ------------------------------------------------------------
try:
    from PySide6.QtCore import (
        Qt, QTimer, QThread, Signal, Slot, QSize,
    )
    from PySide6.QtGui import (
        QIcon, QTextCursor, QFont, QPixmap, QColor, QPainter, QBrush,
    )
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QFrame, QPlainTextEdit,
        QLineEdit, QFileDialog, QMessageBox, QSystemTrayIcon, QMenu,
        QProgressBar, QComboBox, QDialog, QScrollArea, QStyleFactory,
    )
except Exception as e:
    _early_log(f"PySide6 import FAILED: {e}\n{traceback.format_exc()}")
    try:
        sys.stderr.write(
            "ERROR: PySide6 is required.\n"
            "  pip install PySide6\n"
            f"({e})\n")
    except Exception:
        pass
    sys.exit(2)


# ============================================================
# Helpers — mirror menubar_app.py's top-level helpers
# ============================================================

def find_llama_server() -> Optional[str]:
    """Find llama-server.exe (bundled first, then dev build dirs)."""
    exe_dir = Path(sys.executable).resolve().parent
    candidates = [
        # Bundled layout
        _REPO_ROOT / "build" / "bin" / "Release" / "llama-server.exe",
        _REPO_ROOT / "build" / "bin" / "llama-server.exe",
        _REPO_ROOT / "bin" / "llama-server.exe",
        # Next to the exe
        exe_dir / "llama-server.exe",
        exe_dir / "resources" / "build" / "bin" / "Release" / "llama-server.exe",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return None


def detect_gpu_backend() -> str:
    """Return 'cuda' if NVIDIA GPU is available, else 'cpu'."""
    nv = shutil.which("nvidia-smi")
    if not nv:
        return "cpu"
    try:
        out = subprocess.check_output(
            [nv, "--query-gpu=name", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        if out.strip():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def get_gpu_name() -> str:
    nv = shutil.which("nvidia-smi")
    if not nv:
        return ""
    try:
        out = subprocess.check_output(
            [nv, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
            creationflags=0x08000000,
        ).strip()
        return out.splitlines()[0] if out else ""
    except Exception:
        return ""


def get_system_info() -> str:
    parts: list[str] = []
    try:
        import platform
        cpu = platform.processor() or platform.machine()
        if cpu:
            parts.append(cpu)
    except Exception:
        pass
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
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
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ms = MEMORYSTATUSEX()
        ms.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            parts.append(f"{round(ms.ullTotalPhys / (1024**3))} GB")
    except Exception:
        pass
    gpu = get_gpu_name()
    if gpu:
        parts.append(gpu)
    return "  ·  ".join(parts) if parts else "Windows"


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def next_free_port(start: int, max_tries: int = 128) -> tuple[int, str]:
    orig = int(start)
    for i in range(max_tries):
        p = orig + i
        if p > 65535:
            break
        if not is_port_in_use(p):
            note = f"Port {orig} busy → using {p}\n" if p != orig else ""
            return p, note
    return orig, f"Warning: could not find free port from {orig}\n"


def resolve_gateway_worker_ports(preferred_gw: int, preferred_wk: int):
    notes: list[str] = []
    gw, wk = int(preferred_gw), int(preferred_wk)
    if gw == LEGACY_WORKER_PORT:
        gw = DEFAULT_GATEWAY_PORT
    if wk == LEGACY_WORKER_PORT:
        wk = DEFAULT_WORKER_BASE_PORT
    ng, n1 = next_free_port(gw)
    if n1:
        notes.append(n1)
    gw = ng
    nw, n2 = next_free_port(wk)
    if n2:
        notes.append(n2)
    wk = nw
    if gw == wk:
        nw, n3 = next_free_port(gw + 1)
        if n3:
            notes.append(n3)
        wk = nw
    return gw, wk, notes


def _parse_port(s, fallback: int) -> int:
    try:
        p = int(str(s).strip())
        if 1024 <= p <= 65535:
            return p
    except (TypeError, ValueError):
        pass
    return fallback


# ------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------

def get_model_dir_from_config() -> Optional[str]:
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f).get("cpp_backend", {}).get("model_dir")
        except Exception:
            pass
    return None


def _is_junction(p: Path) -> bool:
    """Detect NTFS directory junction (treated like a directory symlink)."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        GetFileAttributesW = kernel32.GetFileAttributesW
        GetFileAttributesW.restype = ctypes.c_uint32
        FILE_ATTRIBUTE_REPARSE_POINT = 0x0400
        INVALID_FILE_ATTRIBUTES = 0xFFFFFFFF
        attrs = GetFileAttributesW(str(p))
        return (attrs != INVALID_FILE_ATTRIBUTES
                and bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT))
    except Exception:
        return False


def _is_junction_or_symlink(p: Path) -> bool:
    try:
        return p.is_symlink() or _is_junction(p)
    except Exception:
        return False


def check_model_dir(model_dir: str) -> dict:
    """Mirror macOS check_model_dir, minus ANE/CoreML awareness."""
    result = {"valid": True, "missing": [], "llm": None,
              "has_audio": False, "has_tts": False, "has_vision": False,
              "has_vision_ane": False}
    model_path = Path(model_dir)
    if not model_path.exists():
        result["valid"] = False
        result["missing"].append("Directory not found")
        return result

    try:
        from server.model_hub import verify_model  # type: ignore
        vr = verify_model(model_dir)
        result["llm"] = vr.llm
        result["has_audio"] = vr.has_audio
        result["has_tts"] = vr.has_tts
        result["has_vision"] = vr.has_vision
        result["has_vision_ane"] = getattr(vr, "has_vision_ane", False)
        result["missing"] = list(vr.missing) + list(vr.size_mismatch)
        result["valid"] = vr.complete
        return result
    except Exception:
        pass

    # Fallback scan
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
    return result


def scan_models() -> List[dict]:
    if not _MODELS_HOME.exists():
        return []
    results = []
    for d in sorted(_MODELS_HOME.iterdir()):
        if not (d.is_dir() or _is_junction_or_symlink(d)):
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
            "is_symlink": d.is_symlink() or _is_junction(d),
            "real_path": str(d.resolve()),
            "size_gb": size_gb,
            **check,
        })
    return results


def import_model_link(source_dir: str) -> tuple[Optional[str], Optional[str]]:
    """Link a model directory into ~/.comni/models/.

    Windows strategy:
      1. Try os.symlink (requires Developer Mode or admin).
      2. Fallback: create an NTFS directory junction with `mklink /J` (no admin).
      3. On failure: return human-readable error.
    """
    _MODELS_HOME.mkdir(parents=True, exist_ok=True)
    src = Path(source_dir).resolve()
    if not src.is_dir():
        return None, f"Source is not a directory: {src}"
    link = _MODELS_HOME / src.name
    if link.exists() or _is_junction_or_symlink(link):
        try:
            if link.resolve() == src:
                return str(link), None
        except Exception:
            pass
        return None, f"'{src.name}' already exists in models home"

    # Try symlink first
    try:
        os.symlink(str(src), str(link), target_is_directory=True)
        return str(link), None
    except (OSError, NotImplementedError) as e:
        logger.debug("os.symlink failed (%s) — trying junction", e)

    # Fallback: mklink /J (directory junction)
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(src)],
            check=True, capture_output=True, text=True,
            creationflags=0x08000000)
        return str(link), None
    except subprocess.CalledProcessError as e:
        return None, (
            "Failed to create symlink or junction.\n"
            f"  {(e.stderr or '').strip() or e}\n\n"
            "Tip: enable Windows 'Developer Mode' in\n"
            "     Settings → Privacy & Security → For developers.")


def get_active_model_dir() -> Optional[str]:
    for m in scan_models():
        if m["valid"]:
            return m["path"]
    cfg = get_model_dir_from_config()
    if cfg and Path(cfg).exists():
        return cfg
    return None


def _prettify_model_family(raw: str) -> str:
    import re
    s = re.sub(r'[-_]gguf$', '', raw, flags=re.IGNORECASE)
    s = re.sub(r'-(\d+)_(\d+)_(\d+)(?=-|$)', r' \1.\2.\3', s)
    s = re.sub(r'-(\d+)_(\d+)(?=-|$)', r' \1.\2', s)
    return s.strip()


def get_model_display_name(model_dir: Optional[str] = None) -> str:
    if not model_dir:
        model_dir = get_active_model_dir()
    if not model_dir:
        return "No model"

    try:
        from server.model_hub import match_spec_by_dir  # type: ignore
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
                return f"{pretty}  ·  {q}" if pretty else q
        return _prettify_model_family(base)
    return _prettify_model_family(Path(model_dir).name)


def get_component_status_text(model_dir: Optional[str] = None) -> str:
    if not model_dir:
        model_dir = get_active_model_dir()
    if not model_dir:
        return "No model installed"
    c = check_model_dir(model_dir)
    parts = [
        ("✓ LLM"    if c.get("llm")    else "✗ LLM"),
        ("✓ Audio"  if c["has_audio"]  else "✗ Audio"),
        ("✓ TTS"    if c["has_tts"]    else "✗ TTS"),
        ("✓ Vision" if c["has_vision"] else "✗ Vision"),
    ]
    return "  ".join(parts)


def save_config(model_dir: str, gateway_port: int = DEFAULT_GATEWAY_PORT,
                worker_base_port: int = DEFAULT_WORKER_BASE_PORT):
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
    elif _CONFIG_EXAMPLE.exists():
        with open(_CONFIG_EXAMPLE, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {"backend": "cpp", "model": {"model_path": "unused-for-cpp-backend"}}
    config["backend"] = "cpp"
    config.setdefault("cpp_backend", {})
    config["cpp_backend"]["model_dir"] = model_dir
    # Always overwrite llamacpp_root — never trust the value baked into a
    # shipped config.json, since the packaging machine's path will not exist
    # on the user's machine (e.g. D:\tc_mb\... vs G:\Comni\...).
    config["cpp_backend"]["llamacpp_root"] = str(_REPO_ROOT)
    # Windows: no CoreML/ANE
    config["cpp_backend"]["vision_backend"] = "auto"
    config.setdefault("service", {})
    config["service"]["gateway_port"] = gateway_port
    config["service"]["worker_base_port"] = worker_base_port
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


# ============================================================
# Service states
# ============================================================

class ServiceState:
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


STATUS_DISPLAY = {
    ServiceState.STOPPED:  ("Stopped",   "#8c8c8c"),
    ServiceState.STARTING: ("Starting…", "#f3b223"),
    ServiceState.RUNNING:  ("Running",   "#35c759"),
    ServiceState.STOPPING: ("Stopping…", "#f3b223"),
    ServiceState.ERROR:    ("Error",     "#e53935"),
}


# ============================================================
# Background thread controlling worker.py + gateway.py
# ============================================================

class ServiceController(QThread):

    state_changed = Signal(str)
    log_line = Signal(str)
    progress_text = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker_proc: Optional[subprocess.Popen] = None
        self._gateway_proc: Optional[subprocess.Popen] = None
        self._log_file = None
        self._mode = ""
        self._gw_port = DEFAULT_GATEWAY_PORT
        self._wk_port = DEFAULT_WORKER_BASE_PORT
        self._log_tail_thread: Optional[threading.Thread] = None
        self._log_tail_running = False

    # public API
    def start_service(self, gw_port: int, wk_port: int):
        self._mode = "start"
        self._gw_port = gw_port
        self._wk_port = wk_port
        self.start()

    def stop_service(self):
        self._mode = "stop"
        self.start()

    def is_running(self) -> bool:
        return (self._worker_proc is not None and self._worker_proc.poll() is None
                and self._gateway_proc is not None and self._gateway_proc.poll() is None)

    def run(self):
        try:
            if self._mode == "start":
                self._do_start()
            elif self._mode == "stop":
                self._do_stop()
        except Exception as e:
            logger.exception("ServiceController.run failed")
            self.log_line.emit(f"\nUnexpected error: {e}\n")
            self.state_changed.emit(ServiceState.ERROR)

    def _popen(self, cmd: list[str], env: dict) -> subprocess.Popen:
        CREATE_NO_WINDOW = 0x08000000
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        return subprocess.Popen(
            cmd, env=env, cwd=str(_SERVER_DIR),
            stdout=self._log_file, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )

    def _do_start(self):
        logger.info("start worker=%s gateway=%s", self._wk_port, self._gw_port)
        if _LOG_PATH.exists():
            try:
                _LOG_PATH.write_text("", encoding="utf-8")
            except Exception:
                pass
        self._log_file = open(_LOG_PATH, "a", encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(_SERVER_DIR)
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        backend = detect_gpu_backend()
        self.log_line.emit(f"GPU backend: {backend}\n")

        self.progress_text.emit("Loading models…")
        self._start_log_tail()

        python_exe = self._resolve_python()
        if not python_exe:
            self.log_line.emit(
                "\nError: No Python interpreter found. "
                "Install Python 3.10+ and put it on PATH, or set "
                "the COMNI_PYTHON env var to its full path.\n")
            self.state_changed.emit(ServiceState.ERROR)
            return
        self.log_line.emit(f"Using Python: {python_exe}\n")

        worker_cmd = [
            python_exe, str(_SERVER_DIR / "worker.py"),
            "--port", str(self._wk_port),
            "--gpu-id", "0",
            "--worker-index", "0",
        ]
        self.log_line.emit(f"Starting worker on port {self._wk_port}…\n")
        try:
            self._worker_proc = self._popen(worker_cmd, env)
        except FileNotFoundError:
            self.log_line.emit(f"\nError: cannot execute {python_exe}\n")
            self.state_changed.emit(ServiceState.ERROR)
            return

        if not self._wait_health(f"http://localhost:{self._wk_port}", timeout=300):
            if self._worker_proc and self._worker_proc.poll() is not None:
                self.log_line.emit(
                    f"\nWorker exited with code {self._worker_proc.returncode}\n")
            else:
                self.log_line.emit("\nWorker startup timeout (300s)\n")
            self.state_changed.emit(ServiceState.ERROR)
            return

        self.log_line.emit("\nWorker ready!\n")
        self.progress_text.emit("Starting gateway…")

        gateway_cmd = [
            python_exe, str(_SERVER_DIR / "gateway.py"),
            "--port", str(self._gw_port),
            "--workers", f"localhost:{self._wk_port}",
            "--http",
        ]
        self._gateway_proc = self._popen(gateway_cmd, env)
        time.sleep(3)
        if self._gateway_proc.poll() is not None:
            self.log_line.emit("\nGateway exited unexpectedly\n")
            self.state_changed.emit(ServiceState.ERROR)
            return

        self.state_changed.emit(ServiceState.RUNNING)
        self.log_line.emit(
            f"\n{'=' * 50}\n"
            f"Server running at http://localhost:{self._gw_port}\n"
            f"{'=' * 50}\n\n"
            "Modes: Turn-based · Omni Duplex · Audio Duplex · Half-Duplex\n"
            "Click 'Open Web UI' or visit the URL above.\n")

    def _do_stop(self):
        logger.info("stop")
        self._stop_log_tail()
        for name, proc in [("Gateway", self._gateway_proc),
                           ("Worker", self._worker_proc)]:
            if proc and proc.poll() is None:
                self.log_line.emit(f"  Stopping {name} (pid={proc.pid})…\n")
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                            capture_output=True, check=False,
                            creationflags=0x08000000)
                        proc.wait(timeout=5)
                except Exception as e:
                    self.log_line.emit(f"  {name} kill failed: {e}\n")
        self._worker_proc = None
        self._gateway_proc = None
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        self.state_changed.emit(ServiceState.STOPPED)
        self.log_line.emit("Server stopped.\n\n")

    def _wait_health(self, url: str, timeout: int = 300) -> bool:
        import urllib.request
        for _ in range(timeout // 2):
            try:
                resp = urllib.request.urlopen(f"{url}/health", timeout=3)
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

    def _resolve_python(self) -> Optional[str]:
        """Find a Python interpreter that can run worker.py / gateway.py.

        When frozen, sys.executable is Comni.exe (not a real Python). We try:
          1. $COMNI_PYTHON env var (full path or bare name)
          2. <exe_dir>/python-embed/python.exe (bundled embedded Python)
          3. conda base (<USERPROFILE>/miniconda3/python.exe, Anaconda3/python.exe)
          4. `python` and `py` on PATH
        """
        candidates: list[str] = []
        env_py = os.environ.get("COMNI_PYTHON")
        if env_py:
            candidates.append(env_py)
        if _FROZEN:
            exe_dir = Path(sys.executable).parent
            candidates.append(str(exe_dir / "python-embed" / "python.exe"))
            candidates.append(str(_REPO_ROOT / "python-embed" / "python.exe"))
            candidates.append(str(exe_dir / "resources" / "python-embed" / "python.exe"))
            candidates.append(str(exe_dir / "python.exe"))
        else:
            candidates.append(sys.executable)
        for p in (
            Path.home() / "miniconda3" / "python.exe",
            Path.home() / "anaconda3" / "python.exe",
            Path.home() / "Anaconda3" / "python.exe",
            Path("C:/ProgramData/miniconda3/python.exe"),
            Path("C:/ProgramData/Anaconda3/python.exe"),
        ):
            candidates.append(str(p))
        candidates.append(shutil.which("python") or "")
        candidates.append(shutil.which("py") or "")
        for c in candidates:
            if c and Path(c).is_file():
                return c
        return shutil.which("py") or shutil.which("python")

    # log tail
    def _start_log_tail(self):
        self._log_tail_running = True
        self._log_tail_thread = threading.Thread(
            target=self._tail_log_file, daemon=True)
        self._log_tail_thread.start()

    def _stop_log_tail(self):
        self._log_tail_running = False

    def _tail_log_file(self):
        while self._log_tail_running and not _LOG_PATH.exists():
            time.sleep(0.2)
        if not self._log_tail_running:
            return
        try:
            with open(_LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                while self._log_tail_running:
                    line = f.readline()
                    if line:
                        self.log_line.emit(line)
                    else:
                        time.sleep(0.1)
        except Exception as e:
            self.log_line.emit(f"[Log reader error: {e}]\n")


# ============================================================
# Main window
# ============================================================

class MainWindow(QMainWindow):

    APP_TITLE = "Comni"

    def __init__(self, tray_available: bool = True):
        super().__init__()
        self._tray_available = tray_available
        self.setWindowTitle(self.APP_TITLE)
        self.resize(580, 780)
        self.setMinimumSize(QSize(480, 600))

        self._state = ServiceState.STOPPED
        self._controller = ServiceController(self)
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.log_line.connect(self._append_log)
        self._controller.progress_text.connect(self._on_progress_text)

        gw, wk = self._load_ports_from_config()
        self._gw_port = gw
        self._wk_port = wk

        self._build_ui()
        if self._tray_available:
            self._build_tray()
        self._update_ui()
        self._run_first_launch_check()

    def _load_ports_from_config(self) -> tuple[int, int]:
        gw, wk = DEFAULT_GATEWAY_PORT, DEFAULT_WORKER_BASE_PORT
        if _CONFIG_PATH.exists():
            try:
                with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                gw = cfg.get("service", {}).get("gateway_port", gw)
                wk = cfg.get("service", {}).get("worker_base_port", wk)
            except Exception:
                pass
        if wk == LEGACY_WORKER_PORT:
            wk = DEFAULT_WORKER_BASE_PORT
        if gw == LEGACY_WORKER_PORT:
            gw = DEFAULT_GATEWAY_PORT
        return gw, wk

    # ---------- UI ----------

    def _card(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.NoFrame)
        f.setStyleSheet(
            "QFrame { background-color: #ffffff; border: 1px solid #e0e0e0;"
            " border-radius: 10px; }")
        return f

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(10)

        # Title
        title = QLabel("Comni")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        root.addWidget(title)
        subtitle = QLabel("Multimodal AI  —  Local Inference")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(subtitle)

        # Status card
        sc = self._card()
        sc_l = QVBoxLayout(sc)
        sc_l.setContentsMargins(16, 14, 16, 14)
        sc_l.setSpacing(6)

        st_row = QHBoxLayout()
        self._status_label = QLabel("●  Stopped")
        self._status_label.setStyleSheet("font-size: 18px; font-weight: 600;")
        st_row.addWidget(self._status_label)
        st_row.addStretch(1)
        self._open_btn = QPushButton("Open Web UI  →")
        self._open_btn.clicked.connect(self.on_open_browser)
        self._open_btn.setVisible(False)
        st_row.addWidget(self._open_btn)
        sc_l.addLayout(st_row)

        self._sys_label = QLabel(get_system_info())
        self._sys_label.setStyleSheet("color: #666; font-size: 11px;")
        self._sys_label.setWordWrap(True)
        sc_l.addWidget(self._sys_label)

        srv_bin = find_llama_server()
        self._server_label = QLabel(
            "llama-server  ✓" if srv_bin else "llama-server  ✗  Not found")
        self._server_label.setStyleSheet(
            "color: #666; font-size: 11px;" if srv_bin
            else "color: #d33; font-size: 11px;")
        sc_l.addWidget(self._server_label)

        self._progress_label = QLabel("")
        self._progress_label.setStyleSheet("color: #666; font-size: 11px;")
        sc_l.addWidget(self._progress_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setVisible(False)
        sc_l.addWidget(self._progress_bar)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  Start Server")
        self._start_btn.setDefault(True)
        self._start_btn.setMinimumHeight(34)
        self._start_btn.clicked.connect(self.on_start)
        btn_row.addWidget(self._start_btn, 2)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setMinimumHeight(34)
        self._stop_btn.clicked.connect(self.on_stop)
        btn_row.addWidget(self._stop_btn, 1)
        sc_l.addLayout(btn_row)
        root.addWidget(sc)

        # Model card
        mc = self._card()
        mc_l = QVBoxLayout(mc)
        mc_l.setContentsMargins(16, 12, 16, 12)
        mc_l.setSpacing(6)

        mrow = QHBoxLayout()
        self._model_name_label = QLabel(get_model_display_name())
        self._model_name_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        mrow.addWidget(self._model_name_label, 1)
        manage_btn = QPushButton("Manage Models")
        manage_btn.clicked.connect(self.on_manage_models)
        mrow.addWidget(manage_btn)
        mc_l.addLayout(mrow)

        self._comp_label = QLabel(get_component_status_text())
        self._comp_label.setStyleSheet("color: #666; font-size: 11px;")
        mc_l.addWidget(self._comp_label)
        root.addWidget(mc)

        # URL card
        uc = self._card()
        uc_l = QHBoxLayout(uc)
        uc_l.setContentsMargins(16, 10, 16, 10)
        uc_l.addWidget(QLabel("URL"))
        self._url_label = QLabel(f"http://localhost:{self._gw_port}")
        self._url_label.setStyleSheet("color: #555;")
        uc_l.addWidget(self._url_label, 1)
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self.on_copy_url)
        uc_l.addWidget(copy_btn)
        root.addWidget(uc)

        # Ports
        prow = QHBoxLayout()
        prow.setSpacing(6)
        prow.addWidget(self._muted_label("Ports"))
        prow.addWidget(self._muted_label("Gateway"))
        self._gw_edit = QLineEdit(str(self._gw_port))
        self._gw_edit.setFixedWidth(72)
        prow.addWidget(self._gw_edit)
        prow.addWidget(self._muted_label("Worker"))
        self._wk_edit = QLineEdit(str(self._wk_port))
        self._wk_edit.setFixedWidth(72)
        prow.addWidget(self._wk_edit)
        prow.addWidget(self._muted_label("auto-resolves if busy"))
        prow.addStretch(1)
        root.addLayout(prow)

        # Log
        log_hdr = QHBoxLayout()
        lh = QLabel("Service Log")
        lh.setStyleSheet("font-size: 13px; font-weight: 600;")
        log_hdr.addWidget(lh)
        log_hdr.addStretch(1)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.on_clear_log)
        log_hdr.addWidget(clear_btn)
        open_log_btn = QPushButton("Open File")
        open_log_btn.clicked.connect(self.on_open_log)
        log_hdr.addWidget(open_log_btn)
        root.addLayout(log_hdr)

        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setMaximumBlockCount(4000)
        self._log_view.setStyleSheet(
            "QPlainTextEdit { background:#fbfbfb; border:1px solid #e0e0e0;"
            " border-radius:8px; color:#444; font-family: Consolas, monospace;"
            " font-size: 11px; }")
        root.addWidget(self._log_view, 1)

    def _muted_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#999; font-size: 11px;")
        return lbl

    # ---------- tray ----------

    def _build_tray(self):
        icon = self.windowIcon()
        if icon.isNull():
            icon = _make_fallback_icon()
            self.setWindowIcon(icon)
        self._tray = QSystemTrayIcon(icon, self)
        self._tray.setToolTip("Comni — Stopped")
        menu = QMenu()
        self._tray_status_action = menu.addAction("Status: Stopped")
        self._tray_status_action.setEnabled(False)
        menu.addSeparator()
        act_start = menu.addAction("Start Server")
        act_start.triggered.connect(self.on_start)
        act_stop = menu.addAction("Stop Server")
        act_stop.triggered.connect(self.on_stop)
        menu.addSeparator()
        act_open = menu.addAction("Open Web UI")
        act_open.triggered.connect(self.on_open_browser)
        act_show = menu.addAction("Show Window")
        act_show.triggered.connect(self._restore_window)
        menu.addSeparator()
        act_quit = menu.addAction("Quit")
        act_quit.triggered.connect(self._really_quit)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason):
        if reason in (QSystemTrayIcon.DoubleClick, QSystemTrayIcon.Trigger):
            self._restore_window()

    def _restore_window(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _really_quit(self):
        if self._controller.is_running():
            ret = QMessageBox.question(
                self, "Quit Comni",
                "Service is running. Stop and quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret != QMessageBox.Yes:
                return
            self._controller.stop_service()
            self._controller.wait(8000)
        self._tray.hide()
        QApplication.instance().quit()

    # ---------- first launch ----------

    def _run_first_launch_check(self):
        issues = []
        if not find_llama_server():
            issues.append(
                "llama-server.exe not found.\n"
                "  Build Release:  cmake -B build -DCMAKE_BUILD_TYPE=Release "
                "&& cmake --build build --config Release --target llama-server\n"
                "  Or drop llama-server.exe next to Comni.exe.")
        model_dir = get_active_model_dir()
        if not model_dir:
            issues.append(
                "No model installed. Click 'Manage Models' to download or "
                "import a GGUF folder.")
        if issues:
            self._append_log("=== Startup Check ===\n")
            for i in issues:
                self._append_log(f"  {i}\n")
            self._append_log("=====================\n")

    # ---------- slots ----------

    @Slot()
    def on_start(self):
        try:
            model_dir = get_active_model_dir()
            if not model_dir:
                ret = QMessageBox.information(
                    self, "No model",
                    "No model installed. Import one now?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if ret == QMessageBox.Yes and self._do_import_model():
                    QTimer.singleShot(100, self.on_start)
                return

            gw = _parse_port(self._gw_edit.text(), self._gw_port)
            wk = _parse_port(self._wk_edit.text(), self._wk_port)
            gw, wk, notes = resolve_gateway_worker_ports(gw, wk)
            self._gw_port, self._wk_port = gw, wk
            self._gw_edit.setText(str(gw))
            self._wk_edit.setText(str(wk))
            self._url_label.setText(f"http://localhost:{gw}")
            save_config(model_dir, gw, wk)
            for n in notes:
                self._append_log(n)

            if not find_llama_server():
                self._append_log("\nError: llama-server.exe not found.\n")
                self._on_state_changed(ServiceState.ERROR)
                return

            self._on_state_changed(ServiceState.STARTING)
            self._append_log(f"\n{'=' * 50}\n")
            self._append_log(
                f"Starting services…  Model: {get_model_display_name(model_dir)}\n")
            self._append_log(f"{'=' * 50}\n\n")
            self._controller.start_service(gw, wk)
        except Exception as e:
            logger.exception("on_start failed")
            self._append_log(f"\nError: {e}\n")

    @Slot()
    def on_stop(self):
        self._on_state_changed(ServiceState.STOPPING)
        self._append_log("\nStopping services…\n")
        self._controller.stop_service()

    @Slot()
    def on_open_browser(self):
        webbrowser.open(f"http://localhost:{self._gw_port}")

    @Slot()
    def on_copy_url(self):
        url = f"http://localhost:{self._gw_port}"
        QApplication.clipboard().setText(url)
        self._append_log(f"URL copied: {url}\n")

    @Slot()
    def on_open_log(self):
        if _LOG_PATH.exists():
            try:
                os.startfile(str(_LOG_PATH))  # noqa: B606
            except Exception as e:
                self._append_log(f"Could not open log file: {e}\n")
        else:
            self._append_log("No log file yet.\n")

    @Slot()
    def on_clear_log(self):
        self._log_view.clear()

    @Slot()
    def on_manage_models(self):
        dlg = ModelManagerDialog(self)
        dlg.exec()
        self._update_ui()

    def _do_import_model(self) -> bool:
        src = QFileDialog.getExistingDirectory(
            self, "Select a GGUF model directory", str(Path.home()))
        if not src:
            return False
        path, err = import_model_link(src)
        if err:
            QMessageBox.warning(self, "Import failed", err)
            return False
        self._append_log(f"Model imported: {src}\n")
        check = check_model_dir(path)
        if check["valid"] and check.get("llm"):
            save_config(path, self._gw_port, self._wk_port)
        self._update_ui()
        return True

    @Slot(str)
    def _on_state_changed(self, state: str):
        self._state = state
        self._update_ui()

    def _update_ui(self):
        text, color = STATUS_DISPLAY[self._state]
        self._status_label.setText(f"●  {text}")
        self._status_label.setStyleSheet(
            f"font-size: 18px; font-weight: 600; color: {color};")

        is_running = self._state == ServiceState.RUNNING
        is_stopped = self._state in (ServiceState.STOPPED, ServiceState.ERROR)
        is_starting = self._state == ServiceState.STARTING

        self._start_btn.setEnabled(is_stopped)
        self._stop_btn.setEnabled(is_running)
        self._open_btn.setVisible(is_running)

        self._progress_bar.setVisible(is_starting)
        self._progress_label.setVisible(is_starting)

        model_dir = get_active_model_dir()
        self._model_name_label.setText(get_model_display_name(model_dir))
        self._comp_label.setText(get_component_status_text(model_dir))

        if hasattr(self, "_tray"):
            self._tray_status_action.setText(f"Status: {text}")
            self._tray.setToolTip(f"Comni — {text}")

    @Slot(str)
    def _on_progress_text(self, t: str):
        self._progress_label.setText(t)

    @Slot(str)
    def _append_log(self, text: str):
        if not text:
            return
        self._log_view.moveCursor(QTextCursor.End)
        self._log_view.insertPlainText(text)
        self._log_view.moveCursor(QTextCursor.End)

    def closeEvent(self, event):
        # If we have a tray icon, hide to tray instead of quitting.
        # Otherwise fall through and let Qt quit the app.
        if hasattr(self, "_tray") and self._tray and self._tray.isVisible():
            if not getattr(self, "_close_hint_shown", False):
                QMessageBox.information(
                    self, "Comni",
                    "Comni keeps running in the system tray.\n"
                    "Right-click the tray icon to quit.")
                self._close_hint_shown = True
            self.hide()
            event.ignore()
            return
        # No tray — accept the close; QApplication.quitOnLastWindowClosed
        # (set to True at startup in this case) will terminate the app.
        event.accept()


# ============================================================
# Model manager dialog
# ============================================================

class ModelManagerDialog(QDialog):

    def __init__(self, parent: MainWindow):
        super().__init__(parent)
        self.setWindowTitle("Comni — Model Manager")
        self.resize(640, 580)
        self._main = parent
        self._downloader = None
        self._quant_combos: dict[str, QComboBox] = {}
        self._latest_progress = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        hdr = QLabel("Available Models")
        hdr.setStyleSheet("font-size: 15px; font-weight: 600;")
        root.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._list_container = QWidget()
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(8)
        scroll.setWidget(self._list_container)
        root.addWidget(scroll, 1)

        self._dl_label = QLabel("")
        self._dl_label.setStyleSheet("color:#555; font-size: 11px;")
        root.addWidget(self._dl_label)
        self._dl_bar = QProgressBar()
        self._dl_bar.setRange(0, 100)
        self._dl_bar.setValue(0)
        self._dl_bar.setVisible(False)
        root.addWidget(self._dl_bar)

        btns = QHBoxLayout()
        self._import_btn = QPushButton("Import Model Folder…")
        self._import_btn.clicked.connect(self._on_import)
        btns.addWidget(self._import_btn)
        self._open_dir_btn = QPushButton("Open in Explorer")
        self._open_dir_btn.clicked.connect(self._on_open_folder)
        btns.addWidget(self._open_dir_btn)
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._on_remove)
        btns.addWidget(self._remove_btn)
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        root.addLayout(btns)

        info = QLabel(f"Model home:  {_MODELS_HOME}")
        info.setStyleSheet("color:#999; font-size:10px;")
        info.setWordWrap(True)
        root.addWidget(info)

        self._dl_timer = QTimer(self)
        self._dl_timer.setInterval(200)
        self._dl_timer.timeout.connect(self._poll_downloader)

        self._refresh()

    def _refresh(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self._quant_combos.clear()

        try:
            from server.model_hub import list_available_models  # type: ignore
            registry_models = list_available_models()
        except Exception:
            registry_models = []

        models = scan_models()
        active_dir = get_active_model_dir()
        installed_dirs = {Path(m["path"]).name for m in models}

        for spec in registry_models:
            self._list_layout.addWidget(
                self._make_registry_card(spec, models, installed_dirs, active_dir))

        extras = [m for m in models
                  if Path(m["path"]).name not in {s.get("dir_name", "") for s in registry_models}]
        if extras:
            sep = QLabel("Locally Imported")
            sep.setStyleSheet("font-size: 13px; font-weight: 600; margin-top: 6px;")
            self._list_layout.addWidget(sep)
            for m in extras:
                self._list_layout.addWidget(self._make_local_card(m, active_dir))

        self._list_layout.addStretch(1)

    def _make_registry_card(self, spec, models, installed_dirs, active_dir) -> QWidget:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e5e5e5; border-radius:10px; }")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(4)

        is_installed = spec.get("dir_name", "") in installed_dirs
        installed_model = None
        if is_installed:
            for m in models:
                if Path(m["path"]).name == spec["dir_name"]:
                    installed_model = m
                    break
        is_complete = installed_model and installed_model.get("valid", False)

        name_str = spec.get("display_name", spec.get("id", ""))
        if is_complete and installed_model["path"] == active_dir:
            name_str += "  ✓ Active"
            color = "#1f7fe0"
        elif is_complete:
            name_str += "  ✓ Installed"
            color = "#1f7fe0"
        elif is_installed:
            name_str += "  ⚠ Incomplete"
            color = "#f3a536"
        else:
            color = "#222"
        name = QLabel(name_str)
        name.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {color};")
        lay.addWidget(name)

        desc = QLabel(spec.get("description", ""))
        desc.setStyleSheet("color:#666; font-size: 11px;")
        desc.setWordWrap(True)
        lay.addWidget(desc)

        if installed_model:
            comp = [
                ("✓ Audio" if installed_model["has_audio"] else "✗ Audio"),
                ("✓ TTS" if installed_model["has_tts"] else "✗ TTS"),
                ("✓ Vision" if installed_model["has_vision"] else "✗ Vision"),
            ]
            comp_lbl = QLabel("  ".join(comp))
            comp_lbl.setStyleSheet("color:#666; font-size: 11px;")
            lay.addWidget(comp_lbl)

        row = QHBoxLayout()
        needs_download = not is_installed or not is_complete
        if needs_download:
            combo = QComboBox()
            for v in spec.get("llm_variants", []):
                size_gb = round(v.get("size", 0) / (1024**3), 1)
                title = f"{v['quant']}  ({size_gb} GB)"
                if v.get("recommended"):
                    title += " ★"
                combo.addItem(title)
            rec_idx = 0
            for i, v in enumerate(spec.get("llm_variants", [])):
                if v.get("recommended"):
                    rec_idx = i
                    break
            if installed_model and installed_model.get("llm"):
                for i, v in enumerate(spec.get("llm_variants", [])):
                    if v["file"] == installed_model["llm"]:
                        rec_idx = i
                        break
            combo.setCurrentIndex(rec_idx)
            combo.setFixedWidth(170)
            self._quant_combos[spec["id"]] = combo
            row.addWidget(combo)
            btn_label = "Resume" if is_installed else "Download"
            dl = QPushButton(btn_label)
            dl.clicked.connect(lambda _, s=spec: self._on_download(s))
            row.addWidget(dl)
        else:
            vbtn = QPushButton("Verify")
            vbtn.clicked.connect(lambda _, s=spec: self._on_verify(s))
            row.addWidget(vbtn)

        row.addStretch(1)
        hf = QLabel(f"HF: {spec.get('hf_repo', '')}")
        hf.setStyleSheet("color:#aaa; font-size: 10px;")
        row.addWidget(hf)
        lay.addLayout(row)
        return card

    def _make_local_card(self, m, active_dir) -> QWidget:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e5e5e5; border-radius:10px; }")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        name = QLabel(m["name"] + ("  ✓ Active" if m["path"] == active_dir else ""))
        name.setStyleSheet("font-size: 12px; font-weight: 600;")
        lay.addWidget(name)
        comp = [
            ("✓ Audio" if m["has_audio"] else "✗ Audio"),
            ("✓ TTS" if m["has_tts"] else "✗ TTS"),
            ("✓ Vision" if m["has_vision"] else "✗ Vision"),
        ]
        c = QLabel("  ".join(comp))
        c.setStyleSheet("color:#666; font-size: 11px;")
        lay.addWidget(c)
        loc = QLabel(("→ " if m["is_symlink"] else "") + m["real_path"])
        loc.setStyleSheet("color:#aaa; font-size: 10px;")
        loc.setWordWrap(True)
        lay.addWidget(loc)
        return card

    def _on_import(self):
        if self._main._do_import_model():
            self._refresh()

    def _on_open_folder(self):
        _MODELS_HOME.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(_MODELS_HOME)])

    def _on_remove(self):
        models = scan_models()
        if not models:
            QMessageBox.information(self, "No models", "No installed models.")
            return
        target = models[-1]
        p = Path(target["path"])
        ret = QMessageBox.question(
            self, "Remove",
            f"Remove '{target['name']}'?\n\n"
            "Only the link in ~/.comni/models/ will be removed.\n"
            "Original files will NOT be deleted (unless this is a full copy).",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        try:
            if p.is_symlink() or _is_junction(p):
                try:
                    p.unlink()
                except Exception:
                    os.rmdir(str(p))
            else:
                shutil.rmtree(str(p), ignore_errors=True)
            QMessageBox.information(self, "Removed", f"Removed: {target['name']}")
        except Exception as e:
            QMessageBox.warning(self, "Remove failed", str(e))
        self._refresh()

    def _on_verify(self, spec):
        try:
            from server.model_hub import verify_model  # type: ignore
        except Exception as e:
            QMessageBox.warning(self, "Verify", f"model_hub unavailable: {e}")
            return
        model_dir = None
        for m in scan_models():
            if Path(m["path"]).name == spec.get("dir_name"):
                model_dir = m["path"]
                break
        if not model_dir:
            QMessageBox.warning(self, "Verify", "Model not installed.")
            return
        vr = verify_model(model_dir)
        if vr.complete:
            QMessageBox.information(
                self, "Verify OK", f"{len(vr.verified)} files verified.")
        else:
            lines = [f"Missing: {m}" for m in vr.missing]
            lines += [f"Size mismatch: {m}" for m in vr.size_mismatch]
            QMessageBox.warning(self, "Verify FAILED", "\n".join(lines) or "Unknown")

    def _on_download(self, spec):
        try:
            from server.model_hub import ModelDownloader  # type: ignore
        except Exception as e:
            QMessageBox.warning(self, "Download", f"model_hub unavailable: {e}")
            return
        if self._downloader and getattr(self._downloader, "_thread", None) \
                and self._downloader._thread.is_alive():
            QMessageBox.information(
                self, "Download", "A download is already in progress.")
            return

        combo = self._quant_combos.get(spec["id"])
        quant_idx = combo.currentIndex() if combo else 0
        variants = spec.get("llm_variants", [])
        quant = (variants[quant_idx]["quant"] if quant_idx < len(variants)
                 else (variants[0]["quant"] if variants else ""))
        self._dl_bar.setVisible(True)
        self._dl_bar.setValue(0)
        self._dl_label.setText(f"Starting {spec['display_name']} ({quant})…")
        self._latest_progress = None
        self._downloader = ModelDownloader(spec, quant)
        self._downloader.start(progress_cb=self._on_progress)
        self._dl_timer.start()

    def _on_progress(self, prog):
        self._latest_progress = prog

    def _poll_downloader(self):
        prog = self._latest_progress
        if not prog:
            return
        if prog.status == "downloading":
            pct = 0
            if prog.bytes_total > 0:
                pct = min(int(prog.bytes_done * 100 / prog.bytes_total), 99)
            speed = _fmt_speed(prog.speed_bps) if prog.speed_bps > 0 else "…"
            done_str = _fmt_bytes(prog.bytes_done)
            total_str = _fmt_bytes(prog.bytes_total) if prog.bytes_total > 0 else "?"
            self._dl_label.setText(
                f"{prog.file_index}/{prog.total_files} done   "
                f"{done_str}/{total_str}   {pct}%   {speed}")
            self._dl_bar.setValue(pct)
        elif prog.status == "verifying":
            self._dl_label.setText("Verifying files…")
        elif prog.status == "done":
            self._dl_label.setText("Download complete ✓")
            self._dl_bar.setValue(100)
            self._dl_timer.stop()
            self._refresh()
        elif prog.status == "error":
            self._dl_label.setText(f"Error: {prog.error}")
            self._dl_bar.setVisible(False)
            self._dl_timer.stop()


def _fmt_speed(bps: float) -> str:
    if bps >= 1024 * 1024:
        return f"{bps / (1024 * 1024):.1f} MB/s"
    if bps >= 1024:
        return f"{bps / 1024:.0f} KB/s"
    return f"{bps:.0f} B/s"


def _fmt_bytes(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} MB"
    return f"{n / 1024:.0f} KB"


# ============================================================
# Fallback icon (used when no Comni.ico is present)
# ============================================================

def _make_fallback_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(QColor("#1f7fe0")))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor("white"))
    font = QFont("Segoe UI", 24, QFont.Bold)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignCenter, "C")
    p.end()
    return QIcon(pix)


def _load_icon() -> QIcon:
    for candidate in [
        _DESKTOP_DIR / "packaging" / "windows" / "Comni.ico",
        Path(sys.executable).parent / "Comni.ico",
        _APPS_ROOT / "assets" / "Comni.ico",
    ]:
        if candidate.is_file():
            return QIcon(str(candidate))
    return _make_fallback_icon()


# ============================================================
# main
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("Comni starting — frozen=%s  apps_root=%s  repo_root=%s",
                _FROZEN, _APPS_ROOT, _REPO_ROOT)
    logger.info("Python: %s", sys.version.split()[0])
    logger.info("Executable: %s", sys.executable)
    logger.info("Log path: %s", _APP_LOG_PATH)

    # Global crash handler — any unhandled exception goes to the app log
    def _excepthook(exc_type, exc, tb):
        import traceback
        logger.error("UNHANDLED: %s\n%s",
                     exc, "".join(traceback.format_exception(exc_type, exc, tb)))
    sys.excepthook = _excepthook

    # AppUserModelID: makes taskbar icon match tray icon
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "com.comni.app")
    except Exception:
        pass

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    try:
        app = QApplication(sys.argv)
        logger.info("QApplication created")
        app.setApplicationName("Comni")
        app.setOrganizationName("Comni")

        try:
            app.setStyle(QStyleFactory.create("Fusion"))
        except Exception:
            pass

        icon = _load_icon()
        app.setWindowIcon(icon)

        # Tray behaviour:
        #   - If tray is available → keep running after the window closes
        #     (close = hide to tray). Matches macOS menu-bar app UX.
        #   - If tray is NOT available (rare: no explorer.exe, non-interactive
        #     session, …) → quit when the window closes, otherwise the
        #     process would become invisible and unkillable.
        tray_ok = QSystemTrayIcon.isSystemTrayAvailable()
        app.setQuitOnLastWindowClosed(not tray_ok)
        logger.info("Tray available: %s (quitOnLastWindowClosed=%s)",
                    tray_ok, not tray_ok)

        logger.info("Constructing MainWindow…")
        win = MainWindow(tray_available=tray_ok)
        win.setWindowIcon(icon)
        win.show()
        win.raise_()
        win.activateWindow()
        logger.info("MainWindow shown — entering event loop")

        signal.signal(signal.SIGINT, signal.SIG_DFL)
        rc = app.exec()
        logger.info("Event loop exited with rc=%s", rc)
        sys.exit(rc)
    except Exception:
        logger.exception("Fatal error in main()")
        raise


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as _e:
        _early_log(f"main() raised: {_e}\n{traceback.format_exc()}")
        raise
