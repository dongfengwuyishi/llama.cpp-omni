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


# ------------------------------------------------------------
# Build metadata helpers — mirror of the macOS implementation in
# menubar_app.py. Both platforms read apps/build_info.json that is
# written by the packaging step (build_dmg.sh on macOS, the Windows
# spec/installer on win32). In dev / source-tree mode the file is
# absent, in which case we fall back to "dev".
# About 对话框、菜单栏标题等都需要 _format_version_tag() —— 之前
# windows_app.py 只调了它没定义，点 About 直接 NameError 弹不出来。
# ------------------------------------------------------------
def _load_build_info() -> dict:
    path = _APPS_ROOT / "build_info.json"
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _format_version_tag() -> str:
    """Short human-readable version, e.g. 'v1.0.16' or 'dev'."""
    bi = _load_build_info()
    v = bi.get("version")
    if isinstance(v, str) and v:
        return f"v{v}" if not v.startswith("v") else v
    return "dev"

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

# Models directory is resolved through comni_paths so user overrides
# (COMNI_MODELS env var, or models_home in ~/.comni/config.json) take effect.
# C: drive on Windows is often the smallest; users must be able to redirect
# multi-GB GGUF files elsewhere — see "Change Models Folder…" in the GUI.
from server.comni_paths import (
    comni_home as _comni_home_fn,
    models_home as _models_home_fn,
    set_models_home as _set_models_home_fn,
    models_home_source as _models_home_source_fn,
    free_bytes as _free_bytes_fn,
)


def _comni_home() -> Path:
    return _comni_home_fn()


def _models_home() -> Path:
    return _models_home_fn()

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


# ------------------------------------------------------------
# LAN IP 探测:避免命中 Clash / V2Ray TUN 虚拟网卡的坑。
#
# 经验教训 (你的机器实测):
#   - socket.connect(("8.8.8.8", 80)) 拿到的 source IP 会被 TUN 代理劫持,
#     返回 198.18.0.1 (Clash Meta TUN 默认段,IANA 保留给 benchmark 的
#     198.18.0.0/15),手机扫这个 URL 根本连不上。
#   - 所以绝不能信"默认路由源 IP",必须自己枚举所有网卡,
#     排掉虚拟/代理网卡,按私有段优先级挑真实家用网卡。
# ------------------------------------------------------------

# 必须排掉的 IP 段(虚拟 / 代理 / link-local / loopback / VirtualBox)
def _is_bogus_ip(ip: str) -> bool:
    if not ip:
        return True
    if ip.startswith(("127.", "0.")):                  # loopback / unspecified
        return True
    if ip.startswith("169.254."):                      # link-local (APIPA)
        return True
    if ip.startswith(("198.18.", "198.19.")):          # IANA benchmark / Clash Meta TUN
        return True
    if ip.startswith("192.168.56."):                   # VirtualBox host-only 默认段
        return True
    if ip.startswith("192.0.0.") or ip.startswith("192.0.2."):  # RFC 6890 TEST-NET
        return True
    return False


# 网卡 friendly name 命中以下关键词时,直接当虚拟网卡排除
_VIRTUAL_IFACE_HINTS = (
    "clash", "mihomo", "tun", "tap", "wireguard", "openvpn",
    "vpn", "vethernet", "virtualbox", "vmware", "hyper-v",
    "loopback", "docker", "wsl", "shadowsocks", "v2ray", "sing-box",
)


def _is_virtual_iface_name(name: Optional[str]) -> bool:
    if not name:
        return False
    lo = name.lower()
    return any(k in lo for k in _VIRTUAL_IFACE_HINTS)


def _rank_private_ip(ip: str) -> int:
    """数字越小优先级越高,代表越像家用局域网。"""
    if ip.startswith("192.168."):
        return 0                                         # 家用 Wi-Fi / LAN 最常见
    if ip.startswith("10."):
        return 1                                         # 企业 / 公寓 LAN
    if ip.startswith("172."):
        try:
            second = int(ip.split(".", 2)[1])
            if 16 <= second <= 31:
                return 2                                 # RFC1918 / 但 Docker/WSL2 也常见
        except ValueError:
            pass
    return 9                                             # 其他(公网 / 未归类)


def _enum_adapters_win32() -> List[tuple]:
    """调 Win32 GetAdaptersAddresses 枚举所有启用中的 IPv4 网卡。
    返回 [(ip, friendly_name), ...]。失败返回 []。"""
    try:
        import ctypes
        from ctypes import wintypes

        AF_INET = 2
        AF_UNSPEC = 0
        GAA_FLAG_SKIP_ANYCAST = 0x0002
        GAA_FLAG_SKIP_MULTICAST = 0x0004
        GAA_FLAG_SKIP_DNS_SERVER = 0x0008

        ULONG = wintypes.ULONG
        PVOID = ctypes.c_void_p

        class SOCKADDR(ctypes.Structure):
            _fields_ = [
                ("sa_family", ctypes.c_ushort),
                ("sa_data", ctypes.c_ubyte * 14),
            ]

        class SOCKET_ADDRESS(ctypes.Structure):
            _fields_ = [
                ("lpSockaddr", ctypes.POINTER(SOCKADDR)),
                ("iSockaddrLength", ctypes.c_int),
            ]

        class IP_ADAPTER_UNICAST_ADDRESS(ctypes.Structure):
            pass
        IP_ADAPTER_UNICAST_ADDRESS._fields_ = [
            ("Length", ULONG),
            ("Flags", wintypes.DWORD),
            ("Next", ctypes.POINTER(IP_ADAPTER_UNICAST_ADDRESS)),
            ("Address", SOCKET_ADDRESS),
            # 其余字段我们不用,塞一个 cushion 就好
            ("_tail", ctypes.c_ubyte * 128),
        ]

        class IP_ADAPTER_ADDRESSES(ctypes.Structure):
            pass
        # 这个结构体在不同 Windows 版本字段会增减,
        # 我们用 union 只读前面固定的几个字段,后面 padding。
        IP_ADAPTER_ADDRESSES._fields_ = [
            ("Length", ULONG),
            ("IfIndex", wintypes.DWORD),
            ("Next", ctypes.POINTER(IP_ADAPTER_ADDRESSES)),
            ("AdapterName", ctypes.c_char_p),
            ("FirstUnicastAddress", ctypes.POINTER(IP_ADAPTER_UNICAST_ADDRESS)),
            ("FirstAnycastAddress", PVOID),
            ("FirstMulticastAddress", PVOID),
            ("FirstDnsServerAddress", PVOID),
            ("DnsSuffix", ctypes.c_wchar_p),
            ("Description", ctypes.c_wchar_p),
            ("FriendlyName", ctypes.c_wchar_p),
            ("_tail", ctypes.c_ubyte * 512),
        ]

        iphlpapi = ctypes.WinDLL("iphlpapi")
        GetAdaptersAddresses = iphlpapi.GetAdaptersAddresses
        GetAdaptersAddresses.argtypes = [
            ULONG, ULONG, PVOID, PVOID, ctypes.POINTER(ULONG)]
        GetAdaptersAddresses.restype = ULONG

        buf_size = ULONG(15 * 1024)                     # 15KB 足够装几十张网卡
        buf = ctypes.create_string_buffer(buf_size.value)
        flags = (GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST
                 | GAA_FLAG_SKIP_DNS_SERVER)
        ret = GetAdaptersAddresses(AF_INET, flags, None,
                                   ctypes.cast(buf, PVOID), ctypes.byref(buf_size))
        if ret == 111:                                   # ERROR_BUFFER_OVERFLOW
            buf = ctypes.create_string_buffer(buf_size.value)
            ret = GetAdaptersAddresses(AF_INET, flags, None,
                                       ctypes.cast(buf, PVOID),
                                       ctypes.byref(buf_size))
        if ret != 0:
            return []

        results: List[tuple] = []
        ptr = ctypes.cast(buf, ctypes.POINTER(IP_ADAPTER_ADDRESSES))
        while ptr:
            ad = ptr.contents
            name = ad.FriendlyName or ad.Description or ""
            ua = ad.FirstUnicastAddress
            while ua:
                sock = ua.contents.Address.lpSockaddr
                if sock and sock.contents.sa_family == AF_INET:
                    raw = bytes(sock.contents.sa_data[2:6])
                    ip = ".".join(str(b) for b in raw)
                    results.append((ip, name))
                ua = ua.contents.Next if bool(ua.contents.Next) else None
            ptr = ad.Next if bool(ad.Next) else None
        return results
    except Exception as e:
        logger.debug("_enum_adapters_win32 failed: %s", e)
        return []


def _get_lan_ip() -> Optional[str]:
    """返回一个手机能连到的家用/办公局域网 IPv4。

    策略(按优先级):
      1. Win32 GetAdaptersAddresses 枚举所有启用的物理网卡 IPv4,
         排除虚拟网卡(TUN/VPN/Docker/vEthernet/VirtualBox),
         按 192.168.* > 10.* > 172.16-31.* 排序。
      2. socket.getaddrinfo(hostname) 兜底(无法拿到 friendly name,
         只能靠 IP 段启发过滤)。
      3. UDP connect 8.8.8.8 拿默认路由源 IP 兜底(会被 Clash TUN 劫持,
         所以只在前两步都空的时候才用)。
    """
    candidates: List[tuple] = []                         # [(rank, ip), ...]

    # --- 1) Win32 枚举(最准,有 interface friendly name 可排虚拟网卡) ---
    for ip, name in _enum_adapters_win32():
        if _is_bogus_ip(ip):
            continue
        if _is_virtual_iface_name(name):
            logger.debug("skip virtual iface: %s -> %s", name, ip)
            continue
        candidates.append((_rank_private_ip(ip), ip))

    # --- 2) hostname resolve 兜底(拿不到 iface name,仅靠 IP 段过滤) ---
    if not candidates:
        try:
            hostname = socket.gethostname()
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = info[4][0]
                if _is_bogus_ip(ip):
                    continue
                candidates.append((_rank_private_ip(ip), ip))
        except Exception:
            pass

    # --- 3) 最后兜底:UDP connect(会被 TUN 劫持) ---
    if not candidates:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.settimeout(0.3)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
            if not _is_bogus_ip(ip):
                candidates.append((_rank_private_ip(ip), ip))
        except Exception:
            pass

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _make_qr_pixmap(text: str, size_px: int = 200):
    """生成二维码 QPixmap,失败时返回 None(不抛出,让 UI 降级为纯文本)。"""
    try:
        import io
        import qrcode           # PyInstaller 会把它自动 pick up
        from PySide6.QtGui import QPixmap
    except Exception as e:
        logger.warning("qrcode / PIL / Qt import failed: %s", e)
        return None
    try:
        # box_size/border 影响像素密度,ERROR_CORRECT_M 容错更好被扫描
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(text)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        pix = QPixmap()
        if not pix.loadFromData(buf.getvalue(), "PNG"):
            return None
        # 按目标尺寸缩放,保持方块锐利用 FastTransformation (不模糊)
        from PySide6.QtCore import Qt
        return pix.scaled(size_px, size_px,
                          Qt.KeepAspectRatio, Qt.FastTransformation)
    except Exception as e:
        logger.warning("QR generation failed for %r: %s", text, e)
        return None


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
    home = _models_home()
    if not home.exists():
        return []
    results = []
    for d in sorted(home.iterdir()):
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
    home = _models_home()
    home.mkdir(parents=True, exist_ok=True)
    src = Path(source_dir).resolve()
    if not src.is_dir():
        return None, f"Source is not a directory: {src}"
    link = home / src.name
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


# Context window choices exposed in the main window. Larger ctx → bigger KV
# cache → more VRAM/RAM. We deliberately cap the picker at 8K on the desktop
# build:
#   * MiniCPM-o is shipped for end-side use — the typical user runs on a
#     6-12 GB GPU or 16 GB RAM, where 16K/32K KV cache fights with the
#     vision/audio encoders for memory and degrades to CPU offload.
#   * Multi-turn voice/video chats almost never need more than ~8K of live
#     context once the C++ sliding-window prune kicks in.
#   * Anything bigger should be a server build, not a desktop app.
# Power users on workstations can still hand-edit ctx_size in
# ~/.comni/config.json — the picker just hides the foot-gun by default.
CTX_SIZE_CHOICES = (4096, 8192)
CTX_SIZE_MAX = max(CTX_SIZE_CHOICES)


def _coerce_ctx_size(cs: object) -> int | None:
    """Map any incoming ctx_size value onto an allowed choice.

    Returns None for values we can't parse at all (caller decides whether to
    fall back to a recommended default). Values larger than CTX_SIZE_MAX are
    clamped down — this is what migrates 16K/32K configs left over from
    earlier Comni builds, without forcing the user to re-pick.
    """
    if not isinstance(cs, int):
        return None
    if cs in CTX_SIZE_CHOICES:
        return cs
    if cs > CTX_SIZE_MAX:
        return CTX_SIZE_MAX
    # Unusual small values (e.g. 2048) — snap to the nearest allowed choice.
    return min(CTX_SIZE_CHOICES, key=lambda c: abs(c - cs))

# Download source choices exposed in the main window — same values the
# macOS app writes to ~/.comni/config.json's ``download_source`` so the
# user's preference survives moving between macOS and Windows builds.
DOWNLOAD_SOURCE_CHOICES = (
    ("auto", "Auto (recommended)"),
    ("hf", "HuggingFace"),
    ("hf_mirror", "HF Mirror (hf-mirror.com)"),
    ("modelscope", "ModelScope (魔搭, China)"),
)


def _detect_system_ram_gb() -> float:
    """Total physical RAM in GB on Windows; 0.0 if unknown.

    Used to pick a sensible default ctx_size on first launch. Falls back
    silently rather than raising — a wrong ctx_size is recoverable, but a
    crashed launcher is not.
    """
    try:
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
        return 0.0


_VRAM_PROBE_CACHE = None  # populated lazily by _detect_vram(); see below


def _detect_vram():
    """Cached VRAM probe — nvidia-smi takes ~50ms, multiple callers
    (popup init, _ensure_ctx_size_default, the GPU label) all want the
    same answer, so we run it once per process.

    Returns a vram_probe.ProbeResult or a stub with .known=False if the
    probe module itself failed to import (e.g. very old packaging).
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

    The signal we actually care about is **GPU VRAM**, not system RAM:
    on a 12 GB consumer NVIDIA card (5070-class) the full omni stack
    (LLM + KV @ 8K + vision + audio + TTS LLM + Token2Wav) overflows
    VRAM, CUDA evicts the vocoder to host memory, and Token2Wav's RTF
    explodes from ~0.17 to ~3.0 — that's the "8K cards, 4K smooth"
    regression we hit in testing.

    Decision tree:
      * NVIDIA VRAM probe succeeded:
          - <14 GB total VRAM  → 4K  (5070/4060 Ti class — 8K guarantees
                                       vocoder PCIE swap)
          - ≥14 GB total VRAM  → 8K  (4080+/workstation class)
      * Probe failed (macOS, AMD, broken driver):
          - ram_gb <8 GB       → 4K  (laptops / mini-PCs)
          - otherwise          → 8K  (Apple Silicon unified memory,
                                       desktops with healthy RAM)

    Note we deliberately never recommend >8K on the desktop build — see
    the rationale next to CTX_SIZE_CHOICES.
    """
    try:
        from server.vram_probe import recommend_ctx_size  # type: ignore
        v = _detect_vram()
        return recommend_ctx_size(vram_total_gb=v.total_gb, ram_gb=ram_gb)
    except Exception:
        if 0 < ram_gb < 8:
            return 4096
        return 8192


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

    try:
        from server.model_hub import load_comni_config, save_comni_config
        user_cfg = load_comni_config()
        raw_cs = user_cfg.get("ctx_size")
        coerced = _coerce_ctx_size(raw_cs)
        if coerced is not None:
            config["cpp_backend"]["ctx_size"] = coerced
            # Migrate the user's persisted choice in-place when we had to
            # clamp it (e.g. 16384 → 8192). Otherwise the dropdown re-opens
            # showing a stale value and the GUI ↔ runtime drift forever.
            if raw_cs != coerced:
                user_cfg["ctx_size"] = coerced
                try:
                    save_comni_config(user_cfg)
                    logger.info(
                        "Migrated ctx_size in ~/.comni/config.json: %r -> %d",
                        raw_cs, coerced)
                except Exception:
                    logger.exception("ctx_size migration save failed")
    except Exception:
        pass

    config.setdefault("service", {})
    config["service"]["gateway_port"] = gateway_port
    config["service"]["worker_base_port"] = worker_base_port
    with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


# ============================================================
# Windows Job Object helpers (kill-on-close) — 彻底解决孤儿子进程残留
# ============================================================
# 即使 Comni.exe 被强杀 / 崩溃 / 正常关闭, OS 都会关闭本进程持有的所有
# HANDLE; 对 kill-on-close Job 而言, HANDLE 被关意味着 Job 里所有进程会
# 被 OS 强制 TerminateProcess 掉, 无一漏网. 配合 cpp_backend 里 worker →
# llama-server 的那层 Job, 形成两级闭环.
# ------------------------------------------------------------

def _create_kill_on_close_job_win():
    """Return a Job Object HANDLE with KILL_ON_JOB_CLOSE, or None if not Win."""
    if not sys.platform.startswith("win"):
        return None
    try:
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                        ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong),
                        ("e", ctypes.c_ulonglong), ("f", ctypes.c_ulonglong)]

        class _BASIC(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit",     ctypes.c_int64),
                ("LimitFlags",              wintypes.DWORD),
                ("MinimumWorkingSetSize",   ctypes.c_size_t),
                ("MaximumWorkingSetSize",   ctypes.c_size_t),
                ("ActiveProcessLimit",      wintypes.DWORD),
                ("Affinity",                ctypes.c_size_t),
                ("PriorityClass",           wintypes.DWORD),
                ("SchedulingClass",         wintypes.DWORD),
            ]

        class _EXT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _BASIC),
                ("IoInfo",                _IO_COUNTERS),
                ("ProcessMemoryLimit",    ctypes.c_size_t),
                ("JobMemoryLimit",        ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed",     ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]

        hJob = kernel32.CreateJobObjectW(None, None)
        if not hJob:
            return None
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(
            hJob, JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info),
        ):
            kernel32.CloseHandle(hJob)
            return None
        return hJob
    except Exception:
        return None


def _assign_process_to_job_win(hJob, pid: int) -> bool:
    if hJob is None or not sys.platform.startswith("win"):
        return False
    try:
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        hProc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not hProc:
            return False
        try:
            return bool(kernel32.AssignProcessToJobObject(hJob, hProc))
        finally:
            kernel32.CloseHandle(hProc)
    except Exception:
        return False


# ============================================================
# Single-instance & orphan cleanup
#
# Why this exists: users naturally double-click Comni.exe, click X, then
# double-click again before the model has finished loading / unloading.
# Without these helpers each launch starts a fresh Comni.exe process while
# the previous one is still inside a multi-GB GGUF mmap or VRAM upload —
# they pile up, ports collide, and after a minute or two five tray icons
# and five MainWindows pop up at once. Reproduced from a user report on
# 2026-04-26: "开了好几次, 过了一两分钟之后, 响应了, 弹了5个 omni".
# ============================================================

_SINGLE_INSTANCE_MUTEX_NAME = "Comni-SingleInstance-v1"
_MAIN_WINDOW_TITLE = "Comni — MiniCPM-o 4.5"  # used to find existing window


def _acquire_single_instance_lock():
    """Try to take the single-instance mutex.

    Returns:
        - mutex_handle (truthy) if THIS process is the first instance.
          Caller must keep the handle alive (just stash on a global / Qt
          object); OS releases it on process death.
        - None if another instance already holds it. Caller should
          activate that instance's window and exit.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ERROR_ALREADY_EXISTS = 183
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CreateMutexW.argtypes = [
            ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR,
        ]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        # Use the local namespace (no "Global\\" prefix) — single-instance
        # is per-Windows-session, not per-machine. Two different users
        # logged in via fast user switching each get their own Comni.
        handle = kernel32.CreateMutexW(None, True, _SINGLE_INSTANCE_MUTEX_NAME)
        last_err = ctypes.get_last_error()
        if not handle:
            logger.warning("CreateMutexW failed (err=%d)", last_err)
            return None
        if last_err == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            return None
        return handle
    except Exception:
        logger.exception("single-instance mutex setup failed")
        return None


def _activate_existing_comni_window() -> bool:
    """Find the running Comni's main window and bring it to the foreground.

    Used when our process detects another instance via the single-instance
    mutex. Restores from minimized / hidden-to-tray state.
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        from ctypes import wintypes
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetWindowTextW.argtypes = [
            wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.SetForegroundWindow.argtypes = [wintypes.HWND]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        SW_RESTORE = 9

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]

        found: list[int] = []

        def cb(hwnd, _lparam):
            n = user32.GetWindowTextLengthW(hwnd)
            if n <= 0:
                return True
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            # Match by prefix so version suffix changes don't break us.
            if buf.value.startswith("Comni"):
                found.append(int(hwnd))
                return False  # stop enumeration
            return True

        user32.EnumWindows(EnumWindowsProc(cb), 0)
        if not found:
            return False
        hwnd = found[0]
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetForegroundWindow(hwnd)
        return True
    except Exception:
        logger.exception("activate existing window failed")
        return False


def _process_path(pid: int) -> Optional[Path]:
    """Return the full image path of `pid`, or None on error / no perms.

    Uses QueryFullProcessImageNameW which works for cross-architecture
    (32-bit caller → 64-bit target) and only needs PROCESS_QUERY_LIMITED_INFORMATION,
    so we can inspect processes started by other elevated apps too.
    """
    if not sys.platform.startswith("win"):
        return None
    try:
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE, wintypes.DWORD,
            wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        h = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return None
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            if not kernel32.QueryFullProcessImageNameW(
                    h, 0, buf, ctypes.byref(size)):
                return None
            return Path(buf.value)
        finally:
            kernel32.CloseHandle(h)
    except Exception:
        return None


def _enumerate_pids() -> list[int]:
    """Return all PIDs visible to the current process."""
    if not sys.platform.startswith("win"):
        return []
    try:
        from ctypes import wintypes
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.EnumProcesses.restype = wintypes.BOOL
        psapi.EnumProcesses.argtypes = [
            ctypes.POINTER(wintypes.DWORD), wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        # 4096 PIDs is way more than any sane Windows session has.
        arr = (wintypes.DWORD * 4096)()
        cb_needed = wintypes.DWORD(0)
        if not psapi.EnumProcesses(arr, ctypes.sizeof(arr),
                                   ctypes.byref(cb_needed)):
            return []
        n = cb_needed.value // ctypes.sizeof(wintypes.DWORD)
        return [int(arr[i]) for i in range(n) if arr[i] != 0]
    except Exception:
        return []


def _kill_orphan_subprocesses() -> int:
    """Kill leftover Comni-owned subprocesses (worker / gateway / llama-server).

    "Owned" means: the process's executable lives inside our install
    directory tree (`_REPO_ROOT`). This is a strict whitelist — we never
    touch a python.exe / llama-server.exe that's part of an unrelated
    project, even if it happens to be running.

    Returns the number of processes killed. Safe to call multiple times.
    """
    if not sys.platform.startswith("win"):
        return 0

    self_pid = os.getpid()
    install_root = _REPO_ROOT.resolve()
    targets: list[tuple[int, Path]] = []

    for pid in _enumerate_pids():
        if pid == self_pid:
            continue
        path = _process_path(pid)
        if path is None:
            continue
        try:
            path_resolved = path.resolve()
            path_resolved.relative_to(install_root)
        except (OSError, ValueError):
            continue
        # Only kill known child binaries — not Comni.exe itself (the
        # other instance handling is via the mutex).
        name = path_resolved.name.lower()
        if name in ("python.exe", "pythonw.exe", "llama-server.exe"):
            targets.append((pid, path_resolved))

    if not targets:
        return 0

    logger.warning(
        "found %d orphan subprocess(es) from a previous Comni run; killing",
        len(targets))
    for pid, path in targets:
        logger.warning("  pid=%d  %s", pid, path)
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True, check=False,
                creationflags=0x08000000)
        except Exception as e:
            logger.warning("    taskkill pid=%d failed: %s", pid, e)
    # Give the OS a moment to release ports.
    time.sleep(0.5)
    return len(targets)


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
        # Windows Job Object (kill-on-close).
        # 所有通过 _popen 启动的子进程 (worker / gateway) 都会 assign 到这里,
        # 这样即使 Comni.exe 崩溃 / 被 taskkill, OS 也会自动把 Job 里所有
        # 进程 (以及它们的子孙) 一并杀掉. 再配合 cpp_backend 里 worker →
        # llama-server 那一层 Job, 形成闭环:
        #   Comni.exe 退出 任何方式  → OS 杀 worker+gateway
        #   worker 退出                → OS 杀 llama-server
        self._proc_job = _create_kill_on_close_job_win()

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
        proc = subprocess.Popen(
            cmd, env=env, cwd=str(_SERVER_DIR),
            stdout=self._log_file, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
        )
        # 把子进程 assign 到 kill-on-close Job. 保证 Comni.exe 无论如何退出,
        # worker / gateway 及其所有子孙进程都会被 OS 自动杀掉.
        if self._proc_job is not None:
            try:
                _assign_process_to_job_win(self._proc_job, proc.pid)
            except Exception as e:
                logger.warning(f"assign_process_to_job pid={proc.pid}: {e}")
        return proc

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
        # Prevent host-level HTTP proxy (Clash / V2Ray / corporate proxy) from
        # hijacking loopback IPC between gateway / worker / llama-server.
        # Python's proxy_bypass doesn't always treat 'localhost' as an
        # exception on non-English Windows / macOS locales.
        for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                   "http_proxy", "https_proxy", "all_proxy"):
            env.pop(_k, None)
        env["NO_PROXY"] = "*"
        env["no_proxy"] = "*"
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

        if not self._wait_health(f"http://127.0.0.1:{self._wk_port}", timeout=300):
            if self._worker_proc and self._worker_proc.poll() is not None:
                self.log_line.emit(
                    f"\nWorker exited with code {self._worker_proc.returncode}\n")
            else:
                self.log_line.emit("\nWorker startup timeout (300s)\n")
            self.state_changed.emit(ServiceState.ERROR)
            return

        self.log_line.emit("\nWorker ready!\n")
        self.progress_text.emit("Starting gateway…")

        # 默认走 HTTPS (自签证书在 apps/certs/),
        # 手机浏览器要求 secure context 才能启用麦克风/摄像头/WebRTC,
        # 本地访问也不成问题 — 首次进入浏览器会弹不安全警告,
        # 点「高级 → 继续访问」即可。
        #
        # 启动前先确保证书覆盖当前 LAN IP — 老证书没有 SAN,
        # Android Chrome 会拒绝授予 getUserMedia 权限。
        try:
            from server.cert_gen import ensure_cert_for_lan  # type: ignore
            certs_dir = _APPS_ROOT / "certs"
            cert_path, key_path, lan_ips = ensure_cert_for_lan(certs_dir)
            if lan_ips:
                self.log_line.emit(
                    f"  TLS cert covers: localhost, 127.0.0.1, "
                    f"{', '.join(lan_ips)}\n")
        except Exception as e:
            logger.warning("cert ensure failed, falling back to existing cert: %s", e)
            self.log_line.emit(
                f"  Warning: could not refresh TLS cert ({e}). "
                "Camera/mic on phone may not work.\n")

        gateway_cmd = [
            python_exe, str(_SERVER_DIR / "gateway.py"),
            "--port", str(self._gw_port),
            "--workers", f"127.0.0.1:{self._wk_port}",
        ]
        self._gateway_proc = self._popen(gateway_cmd, env)
        time.sleep(3)
        if self._gateway_proc.poll() is not None:
            self.log_line.emit("\nGateway exited unexpectedly\n")
            self.state_changed.emit(ServiceState.ERROR)
            return

        self.state_changed.emit(ServiceState.RUNNING)
        lan_ip = _get_lan_ip()
        lan_line = (f"  On your phone (same Wi-Fi): https://{lan_ip}:{self._gw_port}\n"
                    if lan_ip else "")
        self.log_line.emit(
            f"\n{'=' * 50}\n"
            f"Server running at https://localhost:{self._gw_port}\n"
            f"{lan_line}"
            f"{'=' * 50}\n\n"
            "Modes: Turn-based · Omni Duplex · Audio Duplex · Half-Duplex\n"
            "Note: 自签名证书,浏览器首次打开会提示不安全,点「高级 → 继续访问」即可。\n"
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
        # Bypass any host-level HTTP proxy: loopback probes must not be
        # intercepted by local HTTP proxies (Clash/V2Ray etc.).
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
        self.resize(560, 720)
        self.setMinimumSize(QSize(500, 620))

        self._state = ServiceState.STOPPED
        self._controller = ServiceController(self)
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.log_line.connect(self._append_log)
        self._controller.progress_text.connect(self._on_progress_text)

        gw, wk = self._load_ports_from_config()
        self._gw_port = gw
        self._wk_port = wk

        self._lan_ip: Optional[str] = None  # populated when service runs
        self._qr_dialog: Optional["QRDialog"] = None

        self._ensure_ctx_size_default()
        self._build_ui()
        if self._tray_available:
            self._build_tray()
        self._update_ui()
        self._run_first_launch_check()

    def _ensure_ctx_size_default(self):
        """Stamp a RAM-aware ctx_size into ~/.comni/config.json on first run.

        Also migrates legacy values (16K / 32K from earlier Comni builds) down
        to the current allowed range — those builds let users pick larger
        values, but the desktop picker now caps at 8K (see CTX_SIZE_CHOICES).
        Without the clamp, the dropdown silently falls back to the
        recommended default and the user's runtime would mysteriously shrink
        from "32K I picked last week" → "8K". Migrating in-place keeps the
        change visible and one-time only.
        """
        try:
            from server.model_hub import load_comni_config, save_comni_config
            cfg = load_comni_config()
            raw_cs = cfg.get("ctx_size")
            coerced = _coerce_ctx_size(raw_cs)
            if coerced is not None and raw_cs == coerced:
                return  # already a legal value, nothing to do
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
        # 标题行：[64px spacer] [stretch] [Title] [stretch] [About 64px]
        # 左侧加一个跟 About 按钮等宽的占位，否则 stretch 不对称、"Comni"
        # 会被挤到偏左 ~32px。这里宁可多写一行也别让用户感觉标题歪。
        _ABOUT_BTN_W = 64
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addSpacing(_ABOUT_BTN_W)
        title_row.addStretch(1)
        title = QLabel("Comni")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600;")
        title_row.addWidget(title)
        title_row.addStretch(1)
        about_btn = QPushButton("About")
        about_btn.setFixedSize(_ABOUT_BTN_W, 22)
        about_btn.setStyleSheet("QPushButton { font-size: 11px; }")
        about_btn.clicked.connect(self.on_show_about)
        title_row.addWidget(about_btn)
        root.addLayout(title_row)
        subtitle = QLabel("Omni inference in C/C++")
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
        self._status_label.setStyleSheet("font-size: 20px; font-weight: 600;")
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
        btn_row.setSpacing(10)
        self._start_btn = QPushButton("▶  Start Server")
        self._start_btn.setDefault(True)
        self._start_btn.setMinimumHeight(36)
        self._start_btn.setStyleSheet("QPushButton { font-size: 13px; }")
        self._start_btn.clicked.connect(self.on_start)
        btn_row.addWidget(self._start_btn, 2)
        self._stop_btn = QPushButton("■  Stop")
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.setStyleSheet("QPushButton { font-size: 13px; }")
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

        # Service card (单卡两行,对齐 macOS 菜单栏版布局)
        #   Row1: [Desktop]   https://localhost:{port}          [Copy]
        #   Row2: [Mobile ]   https://{lan_ip}:{port} or hint   [ QR ]
        svc = self._card()
        svc_l = QVBoxLayout(svc)
        svc_l.setContentsMargins(16, 12, 16, 12)
        svc_l.setSpacing(10)

        # ─ Desktop row ─
        d_row = QHBoxLayout()
        d_row.setSpacing(8)
        d_lbl = QLabel("Desktop")
        d_lbl.setFixedWidth(58)
        d_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #333;")
        d_row.addWidget(d_lbl)
        self._url_label = QLabel(f"https://localhost:{self._gw_port}")
        self._url_label.setStyleSheet("color: #666; font-size: 12px;")
        self._url_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        d_row.addWidget(self._url_label, 1)
        copy_btn = QPushButton("Copy")
        copy_btn.setFixedWidth(72)
        copy_btn.clicked.connect(self.on_copy_url)
        d_row.addWidget(copy_btn)
        svc_l.addLayout(d_row)

        # ─ Mobile row ─
        m_row = QHBoxLayout()
        m_row.setSpacing(8)
        m_lbl = QLabel("Mobile")
        m_lbl.setFixedWidth(58)
        m_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #333;")
        m_row.addWidget(m_lbl)
        self._mobile_url_label = QLabel("(service not running)")
        self._mobile_url_label.setStyleSheet("color: #aaa; font-size: 12px;")
        self._mobile_url_label.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        m_row.addWidget(self._mobile_url_label, 1)
        self._mobile_qr_btn = QPushButton("QR")
        self._mobile_qr_btn.setFixedWidth(72)
        self._mobile_qr_btn.setEnabled(False)
        self._mobile_qr_btn.clicked.connect(self.on_show_mobile_qr)
        m_row.addWidget(self._mobile_qr_btn)
        svc_l.addLayout(m_row)

        root.addWidget(svc)

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

        # ─ Context size picker (parallel with Ports row, mirrors macOS layout) ─
        ctx_row = QHBoxLayout()
        ctx_row.setSpacing(6)
        ctx_row.addWidget(self._muted_label("Context"))
        self._ctx_combo = QComboBox()
        for cs in CTX_SIZE_CHOICES:
            self._ctx_combo.addItem(f"{cs // 1024}K  ({cs})", cs)
        try:
            from server.model_hub import load_comni_config
            raw_cs = load_comni_config().get("ctx_size")
            cur_cs = _coerce_ctx_size(raw_cs)
            if cur_cs is None:
                cur_cs = _recommend_ctx_size(_detect_system_ram_gb())
            self._ctx_combo.setCurrentIndex(CTX_SIZE_CHOICES.index(cur_cs))
        except Exception:
            # Fallback to 8K when in doubt — _recommend_ctx_size's default.
            self._ctx_combo.setCurrentIndex(CTX_SIZE_CHOICES.index(8192))
        self._ctx_combo.setFixedWidth(120)
        self._ctx_combo.currentIndexChanged.connect(self._on_ctx_size_changed)
        ctx_row.addWidget(self._ctx_combo)

        _ram_gb = _detect_system_ram_gb()
        _vram = _detect_vram()
        # Surface the detected GPU/VRAM so users on small cards understand
        # *why* the default is 4K. No GPU text on macOS / AMD / probe-failed
        # boxes — falling back to the RAM hint preserves the old UX.
        if getattr(_vram, "known", False) and _vram.total_gb > 0:
            ctx_hint = (f"GPU {_vram.gpu_name or 'NVIDIA'} · "
                        f"VRAM {_vram.total_gb:.0f} GB")
        elif _ram_gb > 0:
            ctx_hint = f"RAM {_ram_gb:.0f} GB · larger = more memory"
        else:
            ctx_hint = "larger ctx = more KV cache RAM"
        ctx_row.addWidget(self._muted_label(ctx_hint))
        ctx_row.addStretch(1)
        root.addLayout(ctx_row)

        # NOTE: download_source picker used to live here. It moved into the
        # Model Manager dialog so it sits next to the Download buttons it
        # actually controls. The default ("auto" → HF → Mirror → ModelScope)
        # already covers users behind network restrictions, so hiding the
        # dropdown from the main window de-clutters the runtime panel
        # without losing functionality.

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
        act_about = menu.addAction("About Comni…")
        act_about.triggered.connect(self.on_show_about)
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
        # If the service is up (even mid-load), we must drain children
        # synchronously. Letting Qt exit while llama-server is still
        # in a multi-GB GGUF mmap leaves zombies that hold ports / VRAM
        # for ~30s and cause the next launch to misbehave.
        if self._controller.is_running():
            ret = QMessageBox.question(
                self, "Quit Comni",
                "Service is running. Stop and quit?\n\n"
                "Note: if a model is still loading, this can take "
                "up to ~30 seconds while it unmaps from VRAM.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if ret != QMessageBox.Yes:
                return

            # Visual feedback so users don't think the app froze and
            # start clicking Comni.exe again (the very thing that
            # triggered this bug in the first place).
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._set_quitting_ui()
            try:
                self._controller.stop_service()
                # 60s is enough for an Apple-Silicon-tier model unload
                # *and* the worker/gateway clean shutdown. Previously
                # 8s — empirically too short under load.
                if not self._controller.wait(60_000):
                    logger.warning(
                        "controller.wait() timed out, forcing kill")
            finally:
                QApplication.restoreOverrideCursor()

        # Belt-and-suspenders: even if the controller cleaned up its
        # own children, Job-Object kill-on-close fires when *this*
        # process exits. But if Job creation failed silently for any
        # reason (rare; AV blocking ctypes), explicitly nuke any
        # leftovers right before quit so the next launch is clean.
        try:
            _kill_orphan_subprocesses()
        except Exception:
            pass

        if hasattr(self, "_tray") and self._tray:
            self._tray.hide()
        QApplication.instance().quit()

    def _set_quitting_ui(self) -> None:
        """Disable controls and show a 'Stopping…' banner while quitting."""
        try:
            for btn_name in ("_start_btn", "_stop_btn", "_open_btn",
                             "_models_btn", "_qr_btn"):
                btn = getattr(self, btn_name, None)
                if btn is not None:
                    btn.setEnabled(False)
            if hasattr(self, "_status_label"):
                self._status_label.setText("Stopping…")
        except Exception:
            pass

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
            self._url_label.setText(f"https://localhost:{gw}")
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
        webbrowser.open(f"https://localhost:{self._gw_port}")

    @Slot()
    def on_show_about(self):
        """About 对话框：项目链接 / 模型仓库 / 上游引擎致谢。

        使用 QMessageBox + RichText，再把内部 label 的 openExternalLinks
        打开，让 <a> 链接默认走系统浏览器（QMessageBox 默认不开）。
        """
        body = (
            "<div align='center'>"
            "<h2 style='margin:0'>Comni</h2>"
            "<p style='color:#666; margin:4px 0'>Omni inference in C/C++</p>"
            f"<p style='color:#999; margin:0; font-size:11px'>"
            f"{_format_version_tag()}</p>"
            "</div>"
            "<hr>"
            "<p><b>This app</b><br>"
            "<a href='https://github.com/tc-mb/llama.cpp-omni'>"
            "llama.cpp-omni — GitHub</a></p>"
            "<p><b>Model — MiniCPM-o 4.5</b><br>"
            "<a href='https://github.com/OpenBMB/MiniCPM-o'>"
            "Project page (GitHub)</a><br>"
            "<a href='https://huggingface.co/openbmb/MiniCPM-o-4_5-gguf'>"
            "Weights on Hugging Face</a><br>"
            "<a href='https://modelscope.cn/models/OpenBMB/MiniCPM-o-4_5-gguf'>"
            "Weights on ModelScope</a></p>"
            "<p><b>Built on</b><br>"
            "<a href='https://github.com/ggml-org/llama.cpp'>"
            "llama.cpp (upstream)</a></p>"
            "<p style='color:#999; font-size:11px'>MIT License</p>"
        )
        box = QMessageBox(self)
        box.setWindowTitle("About Comni")
        box.setTextFormat(Qt.RichText)
        box.setText(body)
        try:
            ic = self.windowIcon()
            if not ic.isNull():
                box.setIconPixmap(ic.pixmap(64, 64))
        except Exception:
            pass
        box.setStandardButtons(QMessageBox.Ok)
        # 默认 QMessageBox 不开链接 —— 把内部 label 的 openExternalLinks 打开
        try:
            for lbl in box.findChildren(QLabel):
                lbl.setOpenExternalLinks(True)
        except Exception:
            pass
        box.exec()

    @Slot()
    def on_copy_url(self):
        url = f"https://localhost:{self._gw_port}"
        QApplication.clipboard().setText(url)
        self._append_log(f"URL copied: {url}\n")

    @Slot()
    def on_show_mobile_qr(self):
        """弹出二维码窗口(对齐 macOS 菜单栏版的 QR panel 行为)。"""
        if not self._lan_ip:
            return
        url = f"https://{self._lan_ip}:{self._gw_port}"
        if getattr(self, "_qr_dialog", None) is None:
            self._qr_dialog = QRDialog(self)
        self._qr_dialog.set_url(url)
        self._qr_dialog.show()
        self._qr_dialog.raise_()
        self._qr_dialog.activateWindow()

    def _refresh_mobile_card(self, running: bool) -> None:
        """服务变为 RUNNING 时探测 LAN IP 并更新 Mobile 行;
        STOPPED 时恢复灰色占位。"""
        if not running:
            self._lan_ip = None
            self._mobile_url_label.setText("(service not running)")
            self._mobile_url_label.setStyleSheet(
                "color: #aaa; font-size: 12px;")
            self._mobile_qr_btn.setEnabled(False)
            if getattr(self, "_qr_dialog", None) is not None:
                try:
                    self._qr_dialog.close()
                except Exception:
                    pass
            return

        lan_ip = _get_lan_ip()
        self._lan_ip = lan_ip
        if not lan_ip:
            self._mobile_url_label.setText("(no LAN IP detected)")
            self._mobile_url_label.setStyleSheet(
                "color: #c0392b; font-size: 12px;")
            self._mobile_qr_btn.setEnabled(False)
            return

        url = f"https://{lan_ip}:{self._gw_port}"
        self._mobile_url_label.setText(url)
        self._mobile_url_label.setStyleSheet(
            "color: #666; font-size: 12px;")
        self._mobile_qr_btn.setEnabled(True)
        # 如果 QR 弹窗已打开,同步刷新其 URL (切 Wi-Fi 后场景)
        if getattr(self, "_qr_dialog", None) is not None \
                and self._qr_dialog.isVisible():
            self._qr_dialog.set_url(url)

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

    @Slot(int)
    def _on_ctx_size_changed(self, idx: int):
        if idx < 0 or idx >= len(CTX_SIZE_CHOICES):
            return
        cs = CTX_SIZE_CHOICES[idx]
        try:
            from server.model_hub import load_comni_config, save_comni_config
            cfg = load_comni_config()
            cfg["ctx_size"] = cs
            save_comni_config(cfg)
            self._append_log(
                f"Context size → {cs} ({cs // 1024}K) — restart service to apply\n")
        except Exception as e:
            self._append_log(f"Save ctx_size failed: {e}\n")

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
        prev = self._state
        self._state = state
        # 进入/离开 RUNNING 时更新移动端 URL + 二维码
        if state == ServiceState.RUNNING and prev != ServiceState.RUNNING:
            self._refresh_mobile_card(running=True)
        elif state != ServiceState.RUNNING and prev == ServiceState.RUNNING:
            self._refresh_mobile_card(running=False)
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
        # No tray — drain children synchronously, otherwise Qt happily
        # exits while llama-server is still in mmap and the next launch
        # collides on ports.
        event.ignore()
        self._really_quit()


# ============================================================
# Mobile QR popup dialog
#
# 对齐 macOS 菜单栏版的 `_present_qr_window` 体验:
# 点「QR」按钮后弹出一个独立小窗口,展示 320x320 二维码 +
# URL 文本 + Copy 按钮。多次点击复用同一个窗口。
# ============================================================

class QRDialog(QDialog):

    _QR_PX = 320
    _WIN_W = 360
    _WIN_H = 460

    def __init__(self, parent: "MainWindow"):
        super().__init__(parent)
        self._parent_window = parent
        self._url: str = ""
        self.setWindowTitle("Mobile — Scan to Open")
        self.setModal(False)
        self.setFixedSize(self._WIN_W, self._WIN_H)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        self._qr_label = QLabel()
        self._qr_label.setFixedSize(self._QR_PX, self._QR_PX)
        self._qr_label.setAlignment(Qt.AlignCenter)
        self._qr_label.setStyleSheet(
            "QLabel { background:#fafafa; border:1px solid #e0e0e0;"
            " border-radius:6px; color:#aaa; font-size:11px; }")
        self._qr_label.setText("(generating…)")
        qr_wrap = QHBoxLayout()
        qr_wrap.addStretch(1)
        qr_wrap.addWidget(self._qr_label)
        qr_wrap.addStretch(1)
        root.addLayout(qr_wrap)

        self._url_display = QLabel("")
        self._url_display.setAlignment(Qt.AlignCenter)
        self._url_display.setStyleSheet("color: #555; font-size: 12px;")
        self._url_display.setWordWrap(True)
        self._url_display.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self._url_display)

        tip = QLabel(
            "Scan with your phone (same Wi-Fi).\n"
            "First visit: browser shows a self-signed cert warning\u2014\n"
            "tap \u300cAdvanced \u2192 Proceed\u300d to continue.")
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet("color: #999; font-size: 10px;")
        tip.setWordWrap(True)
        root.addWidget(tip)

        root.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        copy_btn = QPushButton("Copy URL")
        copy_btn.setFixedWidth(110)
        copy_btn.clicked.connect(self._on_copy)
        btn_row.addWidget(copy_btn)
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(self.hide)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

    def set_url(self, url: str) -> None:
        self._url = url
        self._url_display.setText(url)
        pix = _make_qr_pixmap(url, size_px=self._QR_PX)
        if pix is not None:
            self._qr_label.setPixmap(pix)
            self._qr_label.setText("")
        else:
            self._qr_label.clear()
            self._qr_label.setText("QR unavailable")

    @Slot()
    def _on_copy(self) -> None:
        if not self._url:
            return
        QApplication.clipboard().setText(self._url)
        try:
            self._parent_window._append_log(f"Mobile URL copied: {self._url}\n")
        except Exception:
            pass


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

        # ─ Download source (auto = HF → Mirror → ModelScope) ─
        # Lives here, not in the main window: the picker only affects the
        # Download buttons below, so co-locating them keeps the scope
        # obvious. Persisted to ~/.comni/config.json so the choice survives
        # across sessions and across macOS/Windows builds.
        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        src_label = QLabel("Download from")
        src_label.setStyleSheet("color:#666; font-size: 11px;")
        src_row.addWidget(src_label)
        self._dl_source_combo = QComboBox()
        for key, label in DOWNLOAD_SOURCE_CHOICES:
            self._dl_source_combo.addItem(label, key)
        try:
            from server.model_hub import load_comni_config
            cur_src = load_comni_config().get("download_source", "auto")
            keys = [k for (k, _) in DOWNLOAD_SOURCE_CHOICES]
            self._dl_source_combo.setCurrentIndex(
                keys.index(cur_src) if cur_src in keys else 0)
        except Exception:
            self._dl_source_combo.setCurrentIndex(0)
        self._dl_source_combo.setFixedWidth(220)
        self._dl_source_combo.currentIndexChanged.connect(
            self._on_download_source_changed)
        src_row.addWidget(self._dl_source_combo)
        src_hint = QLabel("auto: HF → Mirror → ModelScope")
        src_hint.setStyleSheet("color:#999; font-size: 10px;")
        src_row.addWidget(src_hint)
        src_row.addStretch(1)
        root.addLayout(src_row)

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
        self._change_dir_btn = QPushButton("Change Folder…")
        self._change_dir_btn.setToolTip(
            "Move the models directory to another drive (e.g. D:\\)\n"
            "Useful when C: is running low on space.")
        self._change_dir_btn.clicked.connect(self._on_change_models_home)
        btns.addWidget(self._change_dir_btn)
        self._remove_btn = QPushButton("Remove Selected")
        self._remove_btn.clicked.connect(self._on_remove)
        btns.addWidget(self._remove_btn)
        btns.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btns.addWidget(close_btn)
        root.addLayout(btns)

        self._home_info = QLabel()
        self._home_info.setStyleSheet("color:#999; font-size:10px;")
        self._home_info.setWordWrap(True)
        self._home_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._update_home_info_label()
        root.addWidget(self._home_info)

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

    def _on_download_source_changed(self, idx: int):
        if idx < 0 or idx >= len(DOWNLOAD_SOURCE_CHOICES):
            return
        key, label = DOWNLOAD_SOURCE_CHOICES[idx]
        try:
            from server.model_hub import load_comni_config, save_comni_config
            cfg = load_comni_config()
            cfg["download_source"] = key
            save_comni_config(cfg)
            # Mirror to the main window's log so the user sees feedback even
            # though the picker now lives in this modal — otherwise changing
            # the source feels silent.
            self._main._append_log(
                f"Download source → {label} (applies to next download)\n")
        except Exception as e:
            self._main._append_log(f"Save download_source failed: {e}\n")

    def _on_import(self):
        if self._main._do_import_model():
            self._refresh()

    def _on_open_folder(self):
        home = _models_home()
        home.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(home)])

    def _update_home_info_label(self) -> None:
        home = _models_home()
        src = _models_home_source_fn()
        src_label = {
            "env":     " (set via COMNI_MODELS env var)",
            "config":  "",
            "default": " (default)",
        }.get(src, "")
        free = _free_bytes_fn(home)
        free_str = f" — {free / 1024**3:.1f} GB free" if free is not None else ""
        self._home_info.setText(f"Models folder: {home}{src_label}{free_str}")

    def _on_change_models_home(self):
        from PySide6.QtWidgets import QFileDialog
        current = _models_home()
        new_dir = QFileDialog.getExistingDirectory(
            self, "Choose a folder for Comni models", str(current.parent if current.exists() else Path.home()))
        if not new_dir:
            return
        new_path = Path(new_dir).resolve()
        # Don't let users pick a folder inside the install bundle (would
        # be wiped on uninstall) or the same folder.
        if new_path == current:
            return
        try:
            new_path.relative_to(Path(sys.executable).resolve().parent)
            QMessageBox.warning(
                self, "Invalid location",
                "Don't choose a folder inside the Comni install directory — "
                "uninstalling the app would delete your models. Pick a folder "
                "on a data drive (e.g. D:\\Models\\Comni).")
            return
        except ValueError:
            pass

        # Suggest moving existing data over.
        existing = list(current.iterdir()) if current.exists() else []
        move_them = False
        if existing:
            ret = QMessageBox.question(
                self, "Move existing models?",
                f"You have {len(existing)} item(s) at:\n{current}\n\n"
                f"Move them to the new folder?\n{new_path}\n\n"
                "Yes = move (recommended)\n"
                "No  = leave at the current location, only future downloads "
                "go to the new folder.",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes)
            if ret == QMessageBox.Cancel:
                return
            move_them = (ret == QMessageBox.Yes)

        try:
            new_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot create folder:\n{new_path}\n\n{e}")
            return

        if move_them:
            try:
                for child in existing:
                    target = new_path / child.name
                    if target.exists():
                        QMessageBox.warning(
                            self, "Already exists",
                            f"'{child.name}' already exists in target. Skipping.")
                        continue
                    shutil.move(str(child), str(target))
            except Exception as e:
                QMessageBox.critical(
                    self, "Move failed",
                    f"Some items couldn't be moved:\n{e}\n\n"
                    "The setting was NOT changed. Please move them manually "
                    "and try again.")
                return

        _set_models_home_fn(new_path)
        self._update_home_info_label()
        self._refresh()
        QMessageBox.information(
            self, "Done",
            f"Models folder is now:\n{new_path}\n\n"
            "Future downloads will go here.")

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
# Icon loading
#
# In frozen mode the spec ships Comni.ico to _internal/resources/
# (i.e. <_MEIPASS>/resources/Comni.ico), and Comni.png next to it as a
# fallback. We hunt for them in this order:
#   1. <_MEIPASS>/resources/Comni.ico        ← frozen, primary
#   2. <_MEIPASS>/resources/Comni.png        ← frozen, PNG fallback
#   3. <exe>/Comni.ico                       ← user dropped one next to exe
#   4. dev-tree packaging/windows/Comni.ico  ← running from source
#   5. dev-tree packaging/macos/Comni.png    ← running from source, PNG
#   6. assets/Comni.ico                      ← legacy
# Only if all of those miss do we draw the placeholder square. The
# placeholder used to be a blue rounded box with a white "C" — it is
# the well-known "blue square" users see when ico shipping is broken.
# We now draw a neutral grey square so a regression is obvious instead
# of being silently mistaken for a brand color.
# ============================================================

def _make_fallback_icon() -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QBrush(QColor("#9aa0a6")))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(4, 4, 56, 56, 12, 12)
    p.setPen(QColor("white"))
    font = QFont("Segoe UI", 24, QFont.Bold)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignCenter, "?")
    p.end()
    return QIcon(pix)


def _load_icon() -> QIcon:
    meipass = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "_MEIPASS", "") else None
    exe_dir = Path(sys.executable).resolve().parent

    candidates: list[Path] = []
    if meipass is not None:
        candidates += [
            meipass / "resources" / "Comni.ico",
            meipass / "resources" / "Comni.png",
            meipass / "Comni.ico",
        ]
    candidates += [
        exe_dir / "Comni.ico",
        _DESKTOP_DIR / "packaging" / "windows" / "Comni.ico",
        _DESKTOP_DIR / "packaging" / "macos" / "Comni.png",
        _APPS_ROOT / "assets" / "Comni.ico",
    ]

    for candidate in candidates:
        try:
            if candidate.is_file():
                icon = QIcon(str(candidate))
                if not icon.isNull():
                    logger.info("loaded app icon from %s", candidate)
                    return icon
        except Exception:
            continue

    logger.warning("no app icon found, using grey placeholder. searched: %s",
                   [str(c) for c in candidates])
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

    # ── Single-instance guard ──────────────────────────────────────────
    # Must run BEFORE QApplication is constructed so we don't pay the
    # PySide6 startup cost just to immediately quit. If another Comni is
    # already running, surface its window and exit silently — no error
    # popup (the user just double-clicked the icon, that's not an error).
    _SINGLE_INSTANCE_HANDLE = _acquire_single_instance_lock()
    if _SINGLE_INSTANCE_HANDLE is None and sys.platform.startswith("win"):
        logger.info("another Comni instance is running; activating it")
        # Best-effort: bring the existing window to front. If we can't
        # find it (e.g. the previous process is hung mid-startup), still
        # exit — coming in as a second GUI would only make things worse.
        _activate_existing_comni_window()
        sys.exit(0)
    logger.info("acquired single-instance mutex (handle=%s)",
                _SINGLE_INSTANCE_HANDLE)

    # Clean up zombies left by a previous abrupt exit. Strict whitelist:
    # only kills python.exe / llama-server.exe whose image lives under
    # our install directory. Real-world trigger: user closes Comni while
    # llama-server is still loading a GGUF, the kernel takes ~30s to
    # actually unmap the file, and the next launch fails to bind
    # gateway / worker / cpp_server ports.
    try:
        n_killed = _kill_orphan_subprocesses()
        if n_killed:
            logger.info("cleaned up %d orphan subprocess(es)", n_killed)
    except Exception:
        logger.exception("orphan cleanup failed (non-fatal)")

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
