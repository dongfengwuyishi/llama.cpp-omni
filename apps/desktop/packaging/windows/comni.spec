# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Comni Windows Desktop App
#
# Build with:
#   pyinstaller apps/desktop/packaging/windows/comni.spec --noconfirm --clean
#
# Outputs to:
#   dist/Comni/
#     Comni.exe
#     _internal/...
#     resources/
#       apps/
#       build/bin/Release/llama-server.exe  (if found)
#
# The spec is deliberately NOT bundling server/worker's heavy deps
# (fastapi, uvicorn, onnxruntime, librosa, huggingface_hub, …) inside
# Comni.exe. Those run in a separate Python process started by
# windows_app.py (from miniconda / a system Python). This keeps Comni.exe
# fast to start and small (< 200MB).

import os
import sys
import shutil
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

# The spec file is executed from the repo root when PyInstaller runs
# `pyinstaller apps/desktop/packaging/windows/comni.spec`.
REPO_ROOT = Path(os.getcwd()).resolve()
APPS_ROOT = REPO_ROOT / "apps"
DESKTOP_DIR = APPS_ROOT / "desktop"
PACK_DIR = DESKTOP_DIR / "packaging" / "windows"

MAIN_SCRIPT = str(DESKTOP_DIR / "windows_app.py")

# ------------------------------------------------------------
# Icon
# ------------------------------------------------------------
ICON_ICO = PACK_DIR / "Comni.ico"
if not ICON_ICO.is_file():
    print(f"[spec] WARNING: {ICON_ICO} not found. Run `python "
          f"apps/desktop/packaging/windows/make_icon.py` first.")
    icon_arg = None
else:
    icon_arg = str(ICON_ICO)

# Also ship Comni.ico as a runtime resource so windows_app._load_icon()
# can find it for the taskbar / system-tray icon. Without this, Qt has
# no .ico to load at runtime and falls back to a placeholder (the
# infamous "blue C square") even though Comni.exe itself has the right
# icon embedded in its PE resources.
ICON_PNG_SRC = APPS_ROOT / "desktop" / "packaging" / "macos" / "Comni.png"

# ------------------------------------------------------------
# Hidden imports — PySide6 plugins, QtWebEngineWidgets is NOT used
# ------------------------------------------------------------
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PIL.Image",
]

# ------------------------------------------------------------
# qrcode is imported lazily inside windows_app._make_qr_pixmap(),
# and its subpackages (qrcode.image.pil / qrcode.compat.etree) use
# conditional imports that PyInstaller's static analysis misses.
# collect_all() walks the whole qrcode package and adds every
# submodule + data file as a hiddenimport/datas/binary entry.
# ------------------------------------------------------------
_qr_datas, _qr_binaries, _qr_hidden = collect_all("qrcode")
hiddenimports += _qr_hidden

# ------------------------------------------------------------
# Resources shipped alongside Comni.exe
# The server / assets / frontend code is copied to `resources/apps/`
# so windows_app.py's path-resolution logic finds it.
# ------------------------------------------------------------
datas = list(_qr_datas)

# Ship Comni.ico (and the source PNG as a fallback) into resources/ so
# windows_app._load_icon() finds them in frozen mode.
if ICON_ICO.is_file():
    datas.append((str(ICON_ICO), "resources"))
if ICON_PNG_SRC.is_file():
    datas.append((str(ICON_PNG_SRC), "resources"))

# Directories whose contents must NEVER end up in the user-shipped bundle.
# Each one was a real footgun before:
#   * "packaging"   — build scripts + python-embed/ duplicate; left in
#                     once and ballooned the bundle by ~440 MB raw / ~150
#                     MB after LZMA. python-embed is already shipped
#                     separately at resources/python-embed/.
#   * "node_modules"— frontend build intermediates; sometimes 100s of MB
#                     even though the produced bundle in mobile/assets/
#                     is what we actually need.
#   * "data" / "sessions" — SessionRecorder writes audio replays here when
#                     the app runs from source. Recordings of dev sessions
#                     are private + irrelevant to end users.
#   * "tmp" / ".venv"/ "__pycache__" — usual suspects.
_EXCLUDE_DIR_NAMES = {
    "__pycache__", "tmp", ".venv",
    "packaging",
    "node_modules",
    "data", "sessions",
}


# Ship the apps/ folder as resources/apps/ — needed at runtime
def _collect_apps_folder():
    out = []
    include_dirs = ["server", "assets", "frontend", "desktop", "vad", "certs"]
    for sub in include_dirs:
        src = APPS_ROOT / sub
        if not src.is_dir():
            continue
        for root, dirs, files in os.walk(src):
            # Mutate dirs[] in place so os.walk skips excluded subtrees
            # entirely — this is the documented way to prune the walk.
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIR_NAMES]
            # Skip dev-generated session output trees like ".../tools/omni/output_19060".
            # They get written by the running C++ server. Source tree
            # shouldn't contain them, but a previous PyInstaller run with
            # the bundle still hot-loaded can leave them behind.
            dirs[:] = [d for d in dirs if not d.startswith("output_")]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                if f.endswith(".log"):
                    continue
                # Never ship a dev-time config.json — its llamacpp_root /
                # model_dir would be absolute paths that don't exist on the
                # user's machine. Ship only config.example.json; the GUI
                # regenerates a real config.json in the install location.
                if f == "config.json":
                    continue
                full = Path(root) / f
                rel_parent = full.parent.relative_to(APPS_ROOT)
                target = f"resources/apps/{rel_parent.as_posix()}"
                out.append((str(full), target))
    # Also ship top-level apps/requirements.txt for installer use
    req = APPS_ROOT / "requirements.txt"
    if req.is_file():
        out.append((str(req), "resources/apps"))
    return out

datas += _collect_apps_folder()

# ------------------------------------------------------------
# build_info.json — stamps the version into the bundle so the
# About dialog and the macOS menubar title can show "v1.0.x".
# Sourced from the COMNI_BUILD_VERSION env var, which build.ps1
# sets when the user passes -Version. If unset, version is empty
# and _format_version_tag() falls back to "dev".
# ------------------------------------------------------------
import json as _json
import tempfile as _tempfile
import datetime as _datetime
import subprocess as _subprocess

_bi_version = os.environ.get("COMNI_BUILD_VERSION", "").strip()
_bi_commit = ""
try:
    _bi_commit = _subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(REPO_ROOT), text=True,
        stderr=_subprocess.DEVNULL,
    ).strip()
except Exception:
    pass
_bi_payload = {
    "version": _bi_version,
    "build_time": _datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "build_commit": _bi_commit,
}
# PyInstaller copies the source file using its original basename into
# the destination directory; the file MUST already be named
# build_info.json on disk. Put it in its own temp subdir to avoid
# clobbering / being clobbered by other tools using TEMP.
_bi_dir = Path(_tempfile.mkdtemp(prefix="comni_bi_"))
_bi_path = _bi_dir / "build_info.json"
_bi_path.write_text(_json.dumps(_bi_payload), encoding="utf-8")
datas.append((str(_bi_path), "resources/apps"))
print(f"[spec] build_info.json: version={_bi_version or '(empty -> dev)'} "
      f"commit={_bi_commit or '(none)'}")

# ------------------------------------------------------------
# Ship the embedded Python distribution prepared by
#   apps/desktop/packaging/windows/make_python_embed.ps1
# This makes Comni.exe fully self-contained — no user-side
# Python / miniconda required to run worker.py / gateway.py.
# ------------------------------------------------------------
_embed_src = PACK_DIR / "python-embed"
_embed_files = 0
if _embed_src.is_dir() and (_embed_src / "python.exe").is_file():
    for root, dirs, files in os.walk(_embed_src):
        # Skip pip's internal caches / tests if present
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", "tests", "test")]
        for f in files:
            if f.endswith((".pyc", ".pyo")):
                continue
            src = Path(root) / f
            rel = src.parent.relative_to(_embed_src)
            target = f"resources/python-embed/{rel.as_posix()}" if str(rel) != "." \
                     else "resources/python-embed"
            datas.append((str(src), target))
            _embed_files += 1
    print(f"[spec] Bundling embedded Python: {_embed_files} files from {_embed_src}")
else:
    print(f"[spec] WARNING: no embedded Python found at {_embed_src}. "
          f"Run make_python_embed.ps1 first, otherwise Comni.exe cannot "
          f"run worker.py / gateway.py on machines without Python.")

# Optional: ship llama-server.exe if the user has already built it
binaries = list(_qr_binaries)
llama_exe = REPO_ROOT / "build" / "bin" / "Release" / "llama-server.exe"
if not llama_exe.is_file():
    llama_exe = REPO_ROOT / "build" / "bin" / "llama-server.exe"

if llama_exe.is_file():
    # Keep the same relative layout the app looks for
    datas.append((str(llama_exe), "resources/build/bin/Release"))
    # Also bundle ggml-*.dll / llama.dll / omni.dll next to llama-server.exe
    for dll in llama_exe.parent.glob("*.dll"):
        datas.append((str(dll), "resources/build/bin/Release"))
    print(f"[spec] Bundling llama-server.exe from {llama_exe}")

    # Bundle the CUDA runtime DLLs that llama-server / ggml-cuda.dll needs
    # at runtime. We copy them directly, and later we exclude them from
    # PyInstaller's auto-collected binaries (otherwise they'd be duplicated
    # into _internal/ at ~500 MB cost).
    cuda_bin = os.environ.get("CUDA_PATH", "")
    if not cuda_bin:
        # Try common install paths
        for p in [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3",
        ]:
            if os.path.isdir(p):
                cuda_bin = p
                break
    cuda_bin = os.path.join(cuda_bin, "bin") if cuda_bin else ""
    required_cuda_dlls = ["cudart64_12.dll", "cublas64_12.dll",
                          "cublasLt64_12.dll"]
    cuda_dlls_found = 0
    if cuda_bin and os.path.isdir(cuda_bin):
        for name in required_cuda_dlls:
            src = os.path.join(cuda_bin, name)
            if os.path.isfile(src):
                datas.append((src, "resources/build/bin/Release"))
                cuda_dlls_found += 1
                print(f"[spec] Bundling CUDA DLL: {name}")
    if cuda_dlls_found == 0:
        print("[spec] WARNING: no CUDA DLLs bundled. If llama-server.exe "
              "is CUDA-built, users need CUDA 12.x Runtime installed.")
else:
    print(f"[spec] WARNING: no llama-server.exe found. Build it before "
          f"packaging or drop it into the dist folder manually.")


# ------------------------------------------------------------
# Analysis
# ------------------------------------------------------------
block_cipher = None

a = Analysis(
    [MAIN_SCRIPT],
    pathex=[str(DESKTOP_DIR), str(APPS_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude the huge inference stack — not used by the GUI itself.
        "torch", "tensorflow", "transformers", "scipy", "matplotlib",
        "pandas", "sklearn", "onnxruntime", "librosa", "soundfile",
        "fastapi", "uvicorn", "starlette", "httpx", "websockets",
        "huggingface_hub", "aiohttp",
        "PIL.ImageQt",    # pulls in Qt from PIL side-by-side with PySide6
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# ------------------------------------------------------------
# Strip auto-collected DLLs that we're already shipping under
# resources/build/bin/Release/. These are picked up from %PATH%
# during analysis (because CUDA/bin is on PATH) and would balloon
# the bundle by ~600 MB if left in place.
# ------------------------------------------------------------
_dup_prefixes = (
    "cudart64_",
    "cublas64_",
    "cublaslt64_",
    "cufft64_",
    "curand64_",
    "cusparse64_",
    "cusolver64_",
    "nvrtc64_",
    "nvrtc-builtins64_",
    "nvjitlink_",
    "ggml-cuda",
    "ggml-cpu",
    "ggml-base",
    "ggml.",
    "llama.",
    "omni.",
    "mtmd.",
    "llama-server",
)

def _keep(binary_entry):
    dest = binary_entry[0].replace("\\", "/").lower()
    # Don't touch files that we explicitly shipped under resources/
    # (those need to live next to llama-server.exe at runtime).
    if dest.startswith("resources/"):
        return True
    name = os.path.basename(dest)
    for prefix in _dup_prefixes:
        if name.startswith(prefix):
            return False
    # Strip OpenSSL DLLs that came from python-embed and got auto-promoted
    # to _internal/ root by PyInstaller. The main Comni.exe Python process
    # is a conda build whose _ssl.pyd / _hashlib.pyd / cryptography._rust
    # all link against the "-x64" suffixed libssl-3-x64.dll / libcrypto-3-x64.dll.
    # The non-suffixed libssl-3.dll / libcrypto-3.dll only belong to the
    # python.org-style embedded distribution and must stay isolated under
    # resources/python-embed/. If both are present in _internal/ root, the
    # second one can get pulled in via DLL search order and crash with
    #   "OPENSSL_Uplink(...): no OPENSSL_Applink"
    # the moment any TLS handshake fires (e.g. huggingface_hub download).
    if name in ("libssl-3.dll", "libcrypto-3.dll"):
        return False
    return True

before = len(a.binaries)
a.binaries = [b for b in a.binaries if _keep(b)]
after = len(a.binaries)
print(f"[spec] Stripped {before - after} auto-collected CUDA/llama/OpenSSL "
      f"DLL duplicates from _internal/")

# Same guard for a.datas: any libssl/libcrypto whose dest is "_internal/" root
# (dest path doesn't start with "resources/") is the smoking gun for the
# OpenSSL Uplink crash. Keep them only when shipped under resources/.
def _keep_data(data_entry):
    dest = data_entry[0].replace("\\", "/").lower()
    name = os.path.basename(dest)
    if name in ("libssl-3.dll", "libcrypto-3.dll") and not dest.startswith("resources/"):
        return False
    return True

before_d = len(a.datas)
a.datas = [d for d in a.datas if _keep_data(d)]
after_d = len(a.datas)
if before_d != after_d:
    print(f"[spec] Stripped {before_d - after_d} stray OpenSSL DLLs from "
          f"_internal/ root (must only live under resources/python-embed/)")

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Comni",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # GUI application — no console window
    disable_windowed_traceback=False,
    icon=icon_arg,
    version=str(PACK_DIR / "version_info.txt") if (PACK_DIR / "version_info.txt").is_file() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Comni",
)
