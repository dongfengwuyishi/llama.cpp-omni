#!/usr/bin/env python3
"""Comni Desktop Launcher

一键启动 Comni 推理服务 + Web UI。
用户双击或命令行运行即可使用。

功能：
    1. 检测系统环境（GPU、内存）
    2. 检查模型是否就绪
    3. 生成/更新 config.json
    4. 启动 Worker + Gateway
    5. 自动打开浏览器

用法：
    python launcher.py                          # 默认启动
    python launcher.py --model-dir /path/to/gguf  # 指定模型目录
    python launcher.py --port 8006              # 指定端口
    python launcher.py --no-browser             # 不自动打开浏览器
"""

import os
import sys
import json
import time
import signal
import argparse
import platform
import subprocess
import webbrowser
import logging
from pathlib import Path
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("launcher")

_LAUNCHER_DIR = Path(__file__).resolve().parent
_APPS_ROOT = _LAUNCHER_DIR.parent
_SERVER_DIR = _APPS_ROOT / "server"
_REPO_ROOT = _APPS_ROOT.parent
_CONFIG_PATH = _SERVER_DIR / "config.json"
_CONFIG_EXAMPLE = _SERVER_DIR / "config.example.json"

# 与 IDE 常见 22400 转发错开，与 config.py / config.example.json 一致
DEFAULT_WORKER_BASE_PORT = 22700

_child_processes: List[subprocess.Popen] = []

_BUILD_INFO_PATH = _APPS_ROOT / "build_info.json"


def load_build_info() -> dict:
    """Load build fingerprint generated at packaging time."""
    if _BUILD_INFO_PATH.is_file():
        try:
            return json.loads(_BUILD_INFO_PATH.read_text())
        except Exception:
            pass
    return {}


def verify_integrity() -> bool:
    """Verify SHA-256 of critical files against build_info.json.
    Returns True if all checks pass or no build_info exists (dev mode)."""
    import hashlib
    bi = load_build_info()
    integrity = bi.get("integrity")
    if not integrity:
        return True

    file_map = {
        "gateway": _SERVER_DIR / "gateway.py",
        "worker": _SERVER_DIR / "worker.py",
        "launcher": Path(__file__).resolve(),
        "model_registry": _APPS_ROOT / "assets" / "model_registry.json",
    }
    llama_server = _REPO_ROOT / "build" / "bin" / "llama-server"
    if llama_server.is_file():
        file_map["llama_server"] = llama_server

    tampered = []
    for key, path in file_map.items():
        expected = integrity.get(key)
        if not expected or not path.is_file():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            tampered.append(key)

    if tampered:
        logger.warning(
            "⚠  Integrity check failed for: %s — "
            "this may not be an official Comni build.",
            ", ".join(tampered),
        )
        return False
    logger.info("✓ Build integrity verified (commit %s)", bi.get("build_commit", "?"))
    return True


def detect_system():
    """检测系统环境"""
    info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
    }

    if platform.system() == "Darwin":
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["memory_gb"] = round(mem_bytes / (1024**3), 1)
            info["gpu"] = "Apple Metal"

            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["chip"] = chip
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader"], text=True).strip()
            info["gpu"] = result
        except Exception:
            info["gpu"] = "CPU only"

    return info


def find_llama_server() -> Optional[str]:
    """查找预编译的 llama-server 二进制"""
    candidates = [
        _REPO_ROOT / "build" / "bin" / "llama-server",
        _REPO_ROOT / "build" / "bin" / "Release" / "llama-server",
        _REPO_ROOT / "build" / "bin" / "Release" / "llama-server.exe",
    ]
    if platform.system() != "Windows":
        candidates.append(
            _REPO_ROOT / "build-x64-linux-cuda-release" / "bin" / "llama-server")

    for c in candidates:
        if c.exists():
            return str(c)
    return None


def check_model_dir(model_dir: str) -> dict:
    """检查模型目录是否包含必需文件"""
    model_path = Path(model_dir)
    result = {"valid": True, "missing": [], "llm": None}

    for pattern in ["*Q4_K_M*.gguf", "*Q4_K_S*.gguf", "*Q8_0*.gguf", "*F16*.gguf"]:
        matches = [m for m in model_path.glob(pattern)
                   if m.parent == model_path]
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
            result["missing"].append("LLM GGUF (e.g., MiniCPM-o-4_5-Q4_K_M.gguf)")

    required_sub = {
        "audio": "audio/MiniCPM-o-4_5-audio-F16.gguf",
        "tts": "tts/MiniCPM-o-4_5-tts-F16.gguf",
        "projector": "tts/MiniCPM-o-4_5-projector-F16.gguf",
    }

    for name, rel_path in required_sub.items():
        if not (model_path / rel_path).exists():
            result["valid"] = False
            result["missing"].append(f"{name}: {rel_path}")

    return result


def ensure_config(model_dir: str, port: int = 8006, worker_base_port: int = DEFAULT_WORKER_BASE_PORT):
    """确保 config.json 存在并更新关键字段"""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r") as f:
            config = json.load(f)
    elif _CONFIG_EXAMPLE.exists():
        with open(_CONFIG_EXAMPLE, "r") as f:
            config = json.load(f)
    else:
        config = {
            "backend": "cpp",
            "model": {"model_path": "unused-for-cpp-backend"},
        }

    config["backend"] = "cpp"
    config.setdefault("cpp_backend", {})
    config["cpp_backend"]["model_dir"] = model_dir
    config["cpp_backend"].setdefault("llamacpp_root", str(_REPO_ROOT))
    config.setdefault("service", {})
    config["service"]["gateway_port"] = port
    config["service"]["worker_base_port"] = worker_base_port

    try:
        from server.model_hub import load_comni_config
        comni_cfg = load_comni_config()
        vb = comni_cfg.get("vision_backend", "auto")
        if vb in ("auto", "metal", "coreml"):
            config["cpp_backend"]["vision_backend"] = vb
    except Exception:
        pass

    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    logger.info(f"Config written to {_CONFIG_PATH}")


def _make_no_proxy_env(base_env: dict | None = None) -> dict:
    """Return a child-process env with all HTTP proxy settings disabled.
    Required to prevent loopback traffic (gateway<->worker<->llama-server) from
    being hijacked by host-level HTTP proxies (Clash/V2Ray on 127.0.0.1:7890)."""
    env = os.environ.copy() if base_env is None else dict(base_env)
    for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
               "http_proxy", "https_proxy", "all_proxy"):
        env.pop(_k, None)
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    return env


def start_worker(port: int = DEFAULT_WORKER_BASE_PORT, gpu_id: int = 0) -> subprocess.Popen:
    """启动 Worker 进程"""
    env = _make_no_proxy_env()
    env["PYTHONPATH"] = str(_SERVER_DIR)

    if platform.system() != "Darwin":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        sys.executable,
        str(_SERVER_DIR / "worker.py"),
        "--port", str(port),
        "--gpu-id", str(gpu_id),
        "--worker-index", "0",
    ]

    logger.info(f"Starting Worker on port {port}...")
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(_SERVER_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    _child_processes.append(proc)
    return proc


def start_gateway(port: int = 8006, worker_port: int = DEFAULT_WORKER_BASE_PORT,
                  use_https: bool = True) -> subprocess.Popen:
    """启动 Gateway 进程"""
    env = _make_no_proxy_env()
    env["PYTHONPATH"] = str(_SERVER_DIR)

    cmd = [
        sys.executable,
        str(_SERVER_DIR / "gateway.py"),
        "--port", str(port),
        "--workers", f"127.0.0.1:{worker_port}",
    ]
    if not use_https:
        cmd.append("--http")

    logger.info(f"Starting Gateway on port {port}...")
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(_SERVER_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    _child_processes.append(proc)
    return proc


def wait_for_health(url: str, timeout: int = 300, interval: float = 2.0) -> bool:
    """等待服务就绪"""
    import urllib.request
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # Bypass any host-level HTTP proxy: loopback probes must not be intercepted.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    for i in range(int(timeout / interval)):
        try:
            req = urllib.request.Request(f"{url}/health")
            resp = opener.open(req, timeout=3, context=ctx)
            if resp.status == 200:
                data = json.loads(resp.read())
                if data.get("model_loaded", True):
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def cleanup(*args):
    """清理子进程"""
    logger.info("Shutting down...")
    for proc in _child_processes:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description="Comni Desktop Launcher")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to GGUF model directory")
    parser.add_argument("--port", type=int, default=8006,
                        help="Gateway port (default: 8006)")
    parser.add_argument("--worker-port", type=int, default=DEFAULT_WORKER_BASE_PORT,
                        help=f"Worker port (default: {DEFAULT_WORKER_BASE_PORT})")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    parser.add_argument("--http", action="store_true",
                        help="Use HTTP instead of HTTPS")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 56)
    print("  Comni Desktop App")
    print("=" * 56)

    verify_integrity()

    bi = load_build_info()
    if bi:
        print(f"  Build:   {bi.get('build_commit', '?')} ({bi.get('build_time', '?')})")

    sys_info = detect_system()
    print(f"  System:  {sys_info['os']} {sys_info['arch']}")
    if "chip" in sys_info:
        print(f"  Chip:    {sys_info['chip']}")
    if "memory_gb" in sys_info:
        print(f"  Memory:  {sys_info['memory_gb']} GB")
    if "gpu" in sys_info:
        print(f"  GPU:     {sys_info['gpu']}")
    print(f"  Python:  {sys_info['python']}")
    print()

    server_bin = find_llama_server()
    if not server_bin:
        print("  [ERROR] llama-server binary not found!")
        print("  Please build it first:")
        print(f"    cd {_REPO_ROOT}")
        print("    cmake -B build -DCMAKE_BUILD_TYPE=Release")
        print("    cmake --build build --target llama-server -j")
        sys.exit(1)
    print(f"  Server:  {server_bin}")

    model_dir = args.model_dir
    if not model_dir:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH) as f:
                cfg = json.load(f)
            model_dir = cfg.get("cpp_backend", {}).get("model_dir", "")

    if not model_dir:
        print()
        print("  [ERROR] Model directory not specified!")
        print("  Please run with --model-dir:")
        print(f"    python {__file__} --model-dir /path/to/MiniCPM-o-4_5-gguf")
        print()
        print("  Or create apps/server/config.json:")
        print(f"    cp {_CONFIG_EXAMPLE} {_CONFIG_PATH}")
        print("    # Edit cpp_backend.model_dir")
        sys.exit(1)

    print(f"  Models:  {model_dir}")

    model_check = check_model_dir(model_dir)
    if not model_check["valid"]:
        print()
        print("  [WARNING] Model directory incomplete:")
        for m in model_check["missing"]:
            print(f"    - Missing: {m}")
        print()

    if model_check["llm"]:
        print(f"  LLM:     {model_check['llm']}")

    print("=" * 56)
    print()

    ensure_config(model_dir, args.port, worker_base_port=args.worker_port)

    protocol = "http" if args.http else "https"

    worker_proc = start_worker(port=args.worker_port)
    print(f"  Worker starting on port {args.worker_port}...")
    print(f"  Loading models... (this may take 30-90 seconds)")
    print()

    if not wait_for_health(f"http://127.0.0.1:{args.worker_port}", timeout=300):
        print("  [ERROR] Worker failed to start. Check logs.")
        cleanup()

    print(f"  Worker ready!")

    gateway_proc = start_gateway(
        port=args.port,
        worker_port=args.worker_port,
        use_https=not args.http,
    )

    time.sleep(3)
    url = f"{protocol}://localhost:{args.port}"

    print()
    print("=" * 56)
    print("  Service is running!")
    print(f"  App:     {url}")
    print(f"  API:     {url}/docs")
    print(f"  Admin:   {url}/admin")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 56)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        while True:
            for proc in _child_processes:
                if proc.poll() is not None:
                    logger.error(f"Process {proc.pid} exited unexpectedly")
                    cleanup()
            time.sleep(2)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
