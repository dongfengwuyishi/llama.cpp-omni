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

# ------------------------------------------------------------
# Hidden imports — PySide6 plugins, QtWebEngineWidgets is NOT used
# ------------------------------------------------------------
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

# ------------------------------------------------------------
# Resources shipped alongside Comni.exe
# The server / assets / frontend code is copied to `resources/apps/`
# so windows_app.py's path-resolution logic finds it.
# ------------------------------------------------------------
datas = []

# Ship the apps/ folder as resources/apps/ — needed at runtime
def _collect_apps_folder():
    out = []
    include_dirs = ["server", "assets", "frontend", "desktop", "vad"]
    for sub in include_dirs:
        src = APPS_ROOT / sub
        if not src.is_dir():
            continue
        for root, dirs, files in os.walk(src):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", "tmp", ".venv")]
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
binaries = []
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
    return True

before = len(a.binaries)
a.binaries = [b for b in a.binaries if _keep(b)]
after = len(a.binaries)
print(f"[spec] Stripped {before - after} auto-collected CUDA/llama DLL "
      f"duplicates from _internal/")

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
