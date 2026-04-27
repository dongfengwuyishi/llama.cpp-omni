"""Comni Model Hub — registry, verification, and HuggingFace download engine.

Independent of PyObjC; usable from both Desktop App and CLI.
"""

import hashlib
import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("comni.model_hub")

# ── Paths ────────────────────────────────────────────────

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_REGISTRY_PATH = _ASSETS_DIR / "model_registry.json"

# Paths are resolved lazily through comni_paths so that a user-provided
# COMNI_MODELS env var or a "models_home" override in ~/.comni/config.json
# takes effect without requiring callers to be restarted in a specific
# order. See apps/server/comni_paths.py for the resolution rules.
#
# Use the `server.` prefix so this file works regardless of whether the
# caller put `apps/` or `apps/server/` on sys.path. The desktop app only
# adds `apps/`, which made bare `from comni_paths import ...` fail at
# launch with ModuleNotFoundError and silently swallowed by PyObjC,
# leaving the menubar UI half-initialized.
from server.comni_paths import (
    comni_home as _comni_home_fn,
    models_home as _models_home_fn,
    cache_dir as _cache_dir_fn,
    comni_config_path as _comni_config_path_fn,
)


def _comni_home() -> Path:
    return _comni_home_fn()


def _models_home() -> Path:
    return _models_home_fn()


def _cache_dir() -> Path:
    return _cache_dir_fn()


def _comni_config_path() -> Path:
    return _comni_config_path_fn()


# ── Backwards-compat module-level aliases ────────────────────────────────
# Existing callers (windows_app.py, menubar_app.py, tests) read these
# constants. They now reflect the *current* resolved value at import time;
# new code should call _models_home() / _comni_home() to react to env
# changes without a process restart.
_COMNI_HOME = _comni_home()
_MODELS_HOME = _models_home()
_CACHE_DIR = _cache_dir()
_COMNI_CONFIG_PATH = _comni_config_path()

HF_MAIN_ENDPOINT = "https://huggingface.co"
# Built-in fallback mirror used when HuggingFace is unreachable and the user
# hasn't configured their own mirror. Can be overridden via comni config
# ("hf_mirror") or the ModelDownloader(mirror_url=...) ctor argument.
HF_DEFAULT_MIRROR = "https://hf-mirror.com"
# ModelScope (魔搭) — third-tier fallback. hf-mirror is sometimes unreachable
# behind certain ISPs/firewalls in China; ModelScope's CDN is independently
# operated and a different DNS/IP path, so it covers the long tail.
# Same repo layout (resolve/<rev>/<path>) so the download engine doesn't need
# per-file path mapping; only the URL template differs.
MS_DEFAULT_ENDPOINT = "https://www.modelscope.cn"
MANIFEST_CACHE_HOURS = 24
HEAD_TIMEOUT_SEC = 5

# User agent used for all HF / mirror requests. Avoid the generic "Comni/1.0"
# because some mirrors throttle unknown UAs; give them a descriptive name so
# they can whitelist / identify us.
_DEFAULT_UA = "comni-model-hub/1.0"

ProgressCallback = Callable[[str, int, int], None]  # (filename, downloaded, total)


# ── Networking config helpers ────────────────────────────
#
# Why these exist: on Chinese user machines Clash / V2Ray commonly hijack
# *all* outbound HTTP(S) traffic via HTTPS_PROXY, including requests to
# huggingface.co and hf-mirror.com. The proxy then either returns 502 (HF
# not in the proxy's ruleset) or performs a TLS MITM that breaks the
# handshake (seen in logs: "_ssl.c:993: The handshake operation timed out").
# By default we *drop* the proxy for all HF/mirror download traffic. Users
# who genuinely need a proxy (e.g. overseas users tunnelling back home) can
# re-enable it with   "respect_proxy": true   in ~/.comni/config.json.


def _should_respect_proxy() -> bool:
    """Default False — strip HTTP(S)_PROXY for HF/mirror traffic."""
    try:
        return bool(load_comni_config().get("respect_proxy", False))
    except Exception:
        return False


def _requests_proxies() -> Optional[dict]:
    """Return the value to pass as ``proxies=`` to requests.

    None => let requests read the env (respect_proxy=True).
    ``{"http": None, "https": None}`` => explicitly disable proxies.
    """
    return None if _should_respect_proxy() else {"http": None, "https": None}


def _urllib_opener():
    """urllib opener with proxies disabled by default."""
    import urllib.request
    if _should_respect_proxy():
        return urllib.request.build_opener()
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _should_telemetry_head() -> bool:
    """HEAD count pings HF main site to bump the per-file download counter.
    Pure telemetry — no benefit to the user, and each ping can block up to
    10s on networks where HF is unreachable. Default OFF."""
    try:
        return bool(load_comni_config().get("telemetry_head_count", False))
    except Exception:
        return False


def _should_verify_sha256() -> bool:
    """Post-download SHA256 verification. Default OFF because hashing 4+GB
    takes 30-60s and most users don't want the extra wait. Users on bit-rot
    prone storage or paranoid users can flip ``"verify_sha256": true``."""
    try:
        return bool(load_comni_config().get("verify_sha256", False))
    except Exception:
        return False


# ── Registry ─────────────────────────────────────────────

def load_registry() -> dict:
    """Load the bundled model_registry.json."""
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def list_available_models() -> List[dict]:
    """Return all model specs from the registry."""
    return load_registry().get("models", [])


def get_model_spec(model_id: str) -> Optional[dict]:
    """Look up a single model by its id."""
    for m in list_available_models():
        if m["id"] == model_id:
            return m
    return None


def match_spec_by_dir(dir_name: str) -> Optional[dict]:
    """Match a local directory name to a registry model spec."""
    for m in list_available_models():
        if m.get("dir_name") == dir_name:
            return m
    name_lower = dir_name.lower()
    for m in list_available_models():
        if m.get("dir_name", "").lower() == name_lower:
            return m
    return None


# ── Comni Config (App-level, separate from server config) ─

def load_comni_config() -> dict:
    p = _comni_config_path()
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_comni_config(cfg: dict):
    _comni_home().mkdir(parents=True, exist_ok=True)
    p = _comni_config_path()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)


def _resolve_endpoint_setting(key: str, default: str) -> str:
    """Shared resolver for endpoint-style config values.

    None / missing → ``default``. Empty / "disabled"/"none"/"off" → "" (skip).
    Any other string → trimmed value.
    """
    cfg_val = load_comni_config().get(key, None)
    if cfg_val is None:
        return default
    if isinstance(cfg_val, str):
        if cfg_val.strip().lower() in ("", "disabled", "none", "off"):
            return ""
        return cfg_val.strip()
    return default


def get_hf_mirror() -> str:
    """Return the configured HF mirror, or the built-in default.

    Priority:
      1. User-configured "hf_mirror" in ~/.comni/config.json
      2. HF_DEFAULT_MIRROR (https://hf-mirror.com)
    An empty / explicitly-disabled value can be set with "hf_mirror": "disabled".
    """
    return _resolve_endpoint_setting("hf_mirror", HF_DEFAULT_MIRROR)


def get_ms_endpoint() -> str:
    """Return the configured ModelScope endpoint, or the built-in default.

    Same precedence/disable-keywords as ``get_hf_mirror``. Set
    ``"ms_endpoint": "disabled"`` in the user config to drop ModelScope
    from the fallback chain.
    """
    return _resolve_endpoint_setting("ms_endpoint", MS_DEFAULT_ENDPOINT)


# Valid values for ``download_source`` in the user config — controls which
# endpoint we try first. The remaining endpoints are appended in a sensible
# order so users always have a fallback path even if they pin a primary.
_DOWNLOAD_PREF_VALUES = ("auto", "hf", "hf_mirror", "modelscope")


def get_download_preference() -> str:
    """Return the user's preferred primary download source.

    Values: ``auto`` (default — HF main first, with HF mirror + ModelScope as
    fallback), ``hf``, ``hf_mirror``, ``modelscope``. Anything else is
    coerced back to ``auto`` so the app never hard-fails on a typo.
    """
    cfg_val = load_comni_config().get("download_source", "auto")
    if not isinstance(cfg_val, str):
        return "auto"
    val = cfg_val.strip().lower()
    return val if val in _DOWNLOAD_PREF_VALUES else "auto"


# ── Verification ─────────────────────────────────────────

@dataclass
class VerifyResult:
    complete: bool = True
    missing: List[str] = field(default_factory=list)
    size_mismatch: List[str] = field(default_factory=list)
    verified: List[str] = field(default_factory=list)
    has_audio: bool = False
    has_tts: bool = False
    has_vision: bool = False
    has_vision_ane: bool = False
    llm: Optional[str] = None


def _check_file(model_path: Path, rel_path: str, expected_size: int) -> str:
    """Return '' if OK, or a description of the issue."""
    fp = model_path / rel_path
    if not fp.exists():
        return "missing"
    if expected_size > 0:
        actual = fp.stat().st_size
        if actual != expected_size:
            return f"size mismatch (expected {expected_size}, got {actual})"
    return ""


def verify_model_from_spec(model_dir: str, spec: dict,
                           chosen_quant: Optional[str] = None) -> VerifyResult:
    """Verify a model directory against a registry spec.

    Quick check: file existence + size comparison. No SHA256.
    """
    result = VerifyResult()
    model_path = Path(model_dir)
    if not model_path.exists():
        result.complete = False
        result.missing.append("(directory not found)")
        return result

    # LLM variant
    llm_found = False
    for v in spec.get("llm_variants", []):
        fp = model_path / v["file"]
        if fp.exists():
            issue = _check_file(model_path, v["file"], v.get("size", 0))
            if not issue:
                result.llm = v["file"]
                result.verified.append(v["file"])
                llm_found = True
                if chosen_quant and v.get("quant") == chosen_quant:
                    break
                if not chosen_quant:
                    break
            else:
                result.size_mismatch.append(v["file"])
    if not llm_found:
        result.complete = False
        result.missing.append("LLM GGUF (no variant found)")

    # Required files
    for entry in spec.get("required_files", []):
        rel = entry["path"]
        issue = _check_file(model_path, rel, entry.get("size", 0))
        if issue == "missing":
            result.complete = False
            result.missing.append(rel)
        elif issue:
            result.size_mismatch.append(rel)
        else:
            result.verified.append(rel)

    # Optional files (check but don't fail)
    for entry in spec.get("optional_files", []):
        rel = entry["path"]
        if entry.get("type") == "directory":
            if (model_path / rel).is_dir():
                result.verified.append(rel)
        else:
            issue = _check_file(model_path, rel, entry.get("size", 0))
            if not issue:
                result.verified.append(rel)

    # Component flags
    components = spec.get("components", {})
    for comp_name, files in components.items():
        all_ok = all((model_path / f).exists() for f in files)
        if comp_name == "audio":
            result.has_audio = all_ok
        elif comp_name == "tts":
            result.has_tts = all_ok
        elif comp_name == "vision":
            result.has_vision = all_ok
        elif comp_name == "vision_ane":
            result.has_vision_ane = all_ok

    return result


def verify_model_generic(model_dir: str) -> VerifyResult:
    """Fallback verification for models not in the registry."""
    result = VerifyResult()
    model_path = Path(model_dir)
    if not model_path.exists():
        result.complete = False
        result.missing.append("(directory not found)")
        return result

    for pattern in ["*Q4_K_M*.gguf", "*Q4_K_S*.gguf", "*Q8_0*.gguf", "*F16*.gguf"]:
        matches = [m for m in model_path.glob(pattern) if m.parent == model_path]
        if matches:
            result.llm = matches[0].name
            result.verified.append(matches[0].name)
            break
    if not result.llm:
        all_gguf = list(model_path.glob("*.gguf"))
        llm_candidates = [f for f in all_gguf
                          if not any(x in f.stem.lower()
                                     for x in ("audio", "vision", "tts", "projector"))]
        if llm_candidates:
            result.llm = llm_candidates[0].name
            result.verified.append(llm_candidates[0].name)
        else:
            result.complete = False
            result.missing.append("LLM GGUF model file")

    # Probe common component patterns
    for d in model_path.iterdir():
        if not d.is_dir():
            continue
        gguf_files = list(d.glob("*.gguf"))
        name_lower = d.name.lower()
        if "audio" in name_lower and gguf_files:
            result.has_audio = True
        elif "tts" in name_lower and gguf_files:
            result.has_tts = True
        elif "vision" in name_lower and gguf_files:
            result.has_vision = True
            mlmodelc = list(d.glob("*.mlmodelc"))
            if mlmodelc and mlmodelc[0].is_dir():
                result.has_vision_ane = True

    return result


def verify_model(model_dir: str, spec: Optional[dict] = None) -> VerifyResult:
    """Unified entry point: uses spec if available, else generic scan."""
    if spec is None:
        dir_name = Path(model_dir).name
        spec = match_spec_by_dir(dir_name)
    if spec:
        return verify_model_from_spec(model_dir, spec)
    return verify_model_generic(model_dir)


def sha256_file(filepath: str, progress_cb: Optional[ProgressCallback] = None) -> str:
    """Compute SHA256 of a file. Optionally report progress."""
    h = hashlib.sha256()
    total = os.path.getsize(filepath)
    done = 0
    name = Path(filepath).name
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            done += len(chunk)
            if progress_cb:
                progress_cb(name, done, total)
    return h.hexdigest()


# ── HF Manifest (remote file metadata) ──────────────────

def _manifest_cache_path(hf_repo: str) -> Path:
    safe = hf_repo.replace("/", "_")
    return _cache_dir() / f"manifest_{safe}.json"


def fetch_remote_manifest(hf_repo: str, force: bool = False) -> Dict[str, dict]:
    """Fetch file metadata from HF API. Returns {rel_path: {size, sha256}}.

    Caches result locally for MANIFEST_CACHE_HOURS.
    """
    cache_file = _manifest_cache_path(hf_repo)
    if not force and cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
            age_h = (time.time() - cached.get("_ts", 0)) / 3600
            cfg = load_comni_config()
            max_age = cfg.get("manifest_cache_hours", MANIFEST_CACHE_HOURS)
            if age_h < max_age:
                data = dict(cached)
                data.pop("_ts", None)
                return data
        except Exception:
            pass

    manifest: Dict[str, dict] = {}

    def _pull(endpoint: str) -> Dict[str, dict]:
        from huggingface_hub import HfApi
        api = HfApi(endpoint=endpoint)
        info = api.model_info(hf_repo, files_metadata=True)
        out: Dict[str, dict] = {}
        for sib in info.siblings or []:
            if not sib.rfilename.endswith(".gguf"):
                continue
            entry = {"size": sib.size or 0}
            if sib.lfs:
                entry["sha256"] = sib.lfs.get("sha256", "") if isinstance(sib.lfs, dict) else getattr(sib.lfs, "sha256", "")
            out[sib.rfilename] = entry
        return out

    logger.info("Fetching manifest from HF: %s", hf_repo)
    try:
        manifest = _pull(HF_MAIN_ENDPOINT)
    except Exception as e:
        logger.warning("Failed to fetch manifest from HF main: %s", e)
        mirror = get_hf_mirror()
        if mirror:
            logger.info("Retrying manifest via mirror: %s", mirror)
            try:
                manifest = _pull(mirror)
            except Exception as e2:
                logger.warning("Manifest fallback mirror also failed: %s", e2)
                manifest = {}
        # ModelScope fallback — different infra than HF/HF-mirror, useful
        # when both the main site and hf-mirror are blocked / overloaded.
        if not manifest:
            ms_url = get_ms_endpoint()
            if ms_url:
                try:
                    ms_ep = _Endpoint("modelscope", ms_url, hf_repo, "ms",
                                      "master")
                    logger.info("Retrying manifest via ModelScope: %s", ms_url)
                    manifest = _fetch_modelscope_manifest(ms_ep)
                except Exception as e3:
                    logger.warning("Manifest fallback ModelScope failed: %s",
                                   e3)
                    return manifest
            else:
                return manifest

    _cache_dir().mkdir(parents=True, exist_ok=True)
    to_save = dict(manifest)
    to_save["_ts"] = time.time()
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(to_save, f, indent=2)
    except Exception:
        pass
    return manifest


# ── Download Engine ──────────────────────────────────────

CHUNK_SIZE = 1024 * 1024          # 1 MB per read
SPEED_TEST_BYTES = 2 * 1024 * 1024  # test with first 2 MB
SPEED_TEST_TIMEOUT = 10           # seconds to download test bytes (HF only)
MIN_SPEED_BPS = 200 * 1024        # 200 KB/s — below this, switch to mirror
CONNECT_TIMEOUT = 8               # short: fail fast to trigger mirror fallback
READ_TIMEOUT = 60
MAX_PARALLEL = 4                  # concurrent download threads (file level)
MAX_RETRIES = 3                   # retry on transient network errors
RETRY_BACKOFF = (2, 5, 10)       # seconds to wait between retries

# ── Multipart (Range) download for large files ─────────────────────────
# Single-connection HF / hf-mirror downloads usually cap at ~1-5 MB/s per
# connection. With 8 parallel Range requests we routinely see 20-40 MB/s
# — close to the user's line rate. Only triggered for files at least this
# large (small files see no benefit and extra connections waste handshakes).
MULTIPART_MIN_SIZE = 50 * 1024 * 1024   # 50 MB
MULTIPART_PARTS = 8                     # per-file connections
MULTIPART_PROBE_TIMEOUT = 6             # seconds for the Range-support probe


@dataclass
class DownloadProgress:
    filename: str = ""
    file_index: int = 0
    total_files: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    speed_bps: float = 0.0
    active_workers: int = 0
    status: str = "pending"   # pending | downloading | verifying | done | error
    error: str = ""


# ── Endpoint chain ──────────────────────────────────────────────────────
#
# We treat HF main / HF mirror / ModelScope uniformly via a small
# ``_Endpoint`` value object. The download engine iterates the chain in
# priority order and locks onto the first endpoint that succeeds (so the
# rest of the files in a multi-file model use the same source — avoids
# thrashing between fast/slow endpoints mid-download).

@dataclass
class _Endpoint:
    name: str         # 'hf' | 'hf_mirror' | 'modelscope'
    base_url: str     # e.g. https://huggingface.co
    repo: str         # repo path for THIS endpoint (may differ between HF/MS)
    style: str        # 'hf' or 'ms' — picks URL template + listing API
    revision: str = "main"  # HF default 'main'; ModelScope default 'master'


def _build_endpoint_chain(spec: dict, *,
                          mirror_url: Optional[str] = None,
                          ms_endpoint: Optional[str] = None) -> List[_Endpoint]:
    """Build the prioritized list of endpoints to try for ``spec``.

    Order respects ``download_source`` from the user config:
      * auto       → HF main → HF mirror → ModelScope
      * hf         → HF main → HF mirror → ModelScope
      * hf_mirror  → HF mirror → HF main → ModelScope
      * modelscope → ModelScope → HF mirror → HF main

    Endpoints with empty/disabled URLs are silently dropped, so a user can
    e.g. set ``"hf_mirror": "disabled"`` to skip the mirror entirely.
    """
    if mirror_url is None:
        mirror_url = get_hf_mirror()
    if ms_endpoint is None:
        ms_endpoint = get_ms_endpoint()

    hf_repo = spec.get("hf_repo", "")
    # Allow per-spec override of ModelScope path / revision; default to the
    # same repo string. ModelScope namespaces are case-insensitive in our
    # tests, so reusing the HF path "owner/name" usually works.
    ms_repo = spec.get("ms_repo") or hf_repo
    ms_revision = spec.get("ms_revision", "master")

    hf = _Endpoint("hf", HF_MAIN_ENDPOINT, hf_repo, "hf", "main") \
        if hf_repo else None
    mir = _Endpoint("hf_mirror", mirror_url, hf_repo, "hf", "main") \
        if (hf_repo and mirror_url) else None
    ms = _Endpoint("modelscope", ms_endpoint, ms_repo, "ms", ms_revision) \
        if (ms_repo and ms_endpoint) else None

    pref = get_download_preference()
    if pref == "hf_mirror":
        order = [mir, hf, ms]
    elif pref == "modelscope":
        order = [ms, mir, hf]
    else:  # 'auto' or 'hf'
        order = [hf, mir, ms]
    return [e for e in order if e is not None]


def _resolve_download_url(rel_path: str, endpoint: "_Endpoint") -> str:
    """Build the per-file URL for ``endpoint``.

    Both HF and ModelScope expose ``…/resolve/<rev>/<path>``; only the prefix
    differs (``/{repo}`` vs ``/models/{repo}``).
    """
    if endpoint.style == "ms":
        return (f"{endpoint.base_url}/models/{endpoint.repo}/"
                f"resolve/{endpoint.revision}/{rel_path}")
    return (f"{endpoint.base_url}/{endpoint.repo}/"
            f"resolve/{endpoint.revision}/{rel_path}")


def _list_modelscope_tree(endpoint: "_Endpoint",
                          rel_dir: str) -> Optional[List[dict]]:
    """List files under ``rel_dir`` on ModelScope.

    Uses the public ``/api/v1/models/{repo}/repo/files?Recursive=true`` API
    which returns the full tree in one call (no auth needed for public
    repos). We filter to ``Type==blob`` entries whose ``Path`` lives under
    ``rel_dir``.

    Returns ``[{path, size}]`` (paths relative to repo root) or None on
    failure. Best-effort: a network failure here just means the caller will
    move on to the next endpoint in the chain.
    """
    import requests
    api_url = (f"{endpoint.base_url}/api/v1/models/{endpoint.repo}/"
               f"repo/files?Recursive=true")
    try:
        resp = requests.get(
            api_url,
            headers={"User-Agent": _DEFAULT_UA},
            timeout=(CONNECT_TIMEOUT, 30),
            proxies=_requests_proxies(),
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("ModelScope list_repo_tree failed for %s: %s",
                       endpoint.repo, e)
        return None
    if not isinstance(data, dict) or not data.get("Success"):
        logger.warning("ModelScope list_repo_tree non-success: %s",
                       (data or {}).get("Message"))
        return None
    files = (data.get("Data") or {}).get("Files") or []
    rel_dir_norm = rel_dir.rstrip("/") + "/"
    out: List[dict] = []
    for item in files:
        if item.get("Type") != "blob":
            continue
        path = item.get("Path", "")
        if not (path == rel_dir or path.startswith(rel_dir_norm)):
            continue
        out.append({"path": path, "size": int(item.get("Size", 0) or 0)})
    return out


def _fetch_modelscope_manifest(endpoint: "_Endpoint") -> Dict[str, dict]:
    """Pull the GGUF-only file manifest (size + sha256) from ModelScope.

    Uses the same listing endpoint as ``_list_modelscope_tree`` but keeps
    every ``.gguf`` blob — the result is shape-compatible with the HF
    manifest cache (``{rel_path: {size, sha256}}``).
    """
    import requests
    api_url = (f"{endpoint.base_url}/api/v1/models/{endpoint.repo}/"
               f"repo/files?Recursive=true")
    out: Dict[str, dict] = {}
    resp = requests.get(
        api_url,
        headers={"User-Agent": _DEFAULT_UA},
        timeout=(CONNECT_TIMEOUT, 30),
        proxies=_requests_proxies(),
    )
    resp.raise_for_status()
    data = resp.json()
    files = (data.get("Data") or {}).get("Files") or []
    for item in files:
        if item.get("Type") != "blob":
            continue
        path = item.get("Path", "")
        if not path.endswith(".gguf"):
            continue
        entry: dict = {"size": int(item.get("Size", 0) or 0)}
        sha = item.get("Sha256")
        if sha:
            entry["sha256"] = sha
        out[path] = entry
    return out


class ModelDownloader:
    """Parallel model downloader.

    Tries a chain of endpoints (HF main → HF mirror → ModelScope by default;
    user can reorder via ``download_source`` in ~/.comni/config.json) and
    "locks" onto the first one that succeeds for the rest of the model so
    multi-file downloads don't thrash between endpoints.
    """

    def __init__(self, spec: dict, quant: str, dest_dir: Optional[str] = None,
                 mirror_url: str = "", ms_endpoint: str = ""):
        self.spec = spec
        self.quant = quant
        self.dest_dir = Path(dest_dir) if dest_dir else _models_home() / spec["dir_name"]
        # Public attrs kept for backwards compatibility with existing UI code
        # that reads them for diagnostics. Treat them as best-effort labels;
        # the real source of truth is self._endpoint_chain.
        self.mirror_url = mirror_url or get_hf_mirror()
        self.ms_endpoint = ms_endpoint or get_ms_endpoint()
        self.hf_repo = spec["hf_repo"]
        self._endpoint_chain: List[_Endpoint] = _build_endpoint_chain(
            spec, mirror_url=self.mirror_url, ms_endpoint=self.ms_endpoint)
        if not self._endpoint_chain:
            # Should never happen unless spec has no hf_repo AND ms is disabled
            raise ValueError(
                f"No download endpoints available for {spec.get('id', '?')}")
        logger.info(
            "Endpoint chain: %s",
            " → ".join(f"{e.name}({e.base_url})" for e in self._endpoint_chain),
        )
        self.progress = DownloadProgress()
        self._cancel = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._progress_cb: Optional[Callable[[DownloadProgress], None]] = None
        # Once any thread succeeds we record the winning endpoint name here
        # so subsequent files skip the chain walk and go straight to it.
        self._locked_endpoint: Optional[str] = None
        self._lock = threading.Lock()
        self._file_progress: Dict[str, int] = {}   # per-file bytes done
        self._completed_count = 0
        self._speed_window_bytes = 0
        self._speed_window_start = 0.0

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def cancel(self):
        self._cancel.set()

    # -- locked-endpoint access (lock-protected: multiple worker threads
    # read/write concurrently from _download_one / _stream_download) --

    def _get_locked_endpoint(self) -> Optional[str]:
        with self._lock:
            return self._locked_endpoint

    def _set_locked_endpoint(self, name: str, reason: str = "") -> None:
        """Lock onto an endpoint by name. First writer wins — we never
        un-lock. Avoids per-file thrash where one thread briefly sees HF
        succeed while others are already on the mirror."""
        with self._lock:
            if self._locked_endpoint is not None:
                return
            self._locked_endpoint = name
            logger.info("Endpoint locked: %s (%s)", name, reason or "decision")

    def _resolved_chain(self) -> List[_Endpoint]:
        """Return the chain to walk for the next file.

        Once locked we only return the locked endpoint so workers don't waste
        time probing already-known-bad sources. Pre-lock we walk the full
        chain in priority order.
        """
        locked = self._get_locked_endpoint()
        if locked is None:
            return list(self._endpoint_chain)
        for ep in self._endpoint_chain:
            if ep.name == locked:
                return [ep]
        # Locked endpoint disappeared from the chain (shouldn't happen) →
        # fall back to whatever we have.
        return list(self._endpoint_chain)

    def _files_to_download(self) -> List[dict]:
        """Build ordered list of files: chosen LLM variant + required + optional.

        Skips entries with type=directory (handled separately by _download_directories).
        Also skips platform-specific entries not matching current OS.
        """
        import platform as _plat
        current_platform = _plat.system().lower()
        files = []
        for v in self.spec.get("llm_variants", []):
            if v["quant"] == self.quant:
                files.append({"path": v["file"], "size": v.get("size", 0)})
                break
        for entry in self.spec.get("required_files", []):
            files.append({"path": entry["path"], "size": entry.get("size", 0)})
        for entry in self.spec.get("optional_files", []):
            if entry.get("type") == "directory":
                continue
            plat = entry.get("platform", "")
            if plat and plat != current_platform:
                continue
            files.append({"path": entry["path"], "size": entry.get("size", 0)})
        return files

    def _directories_to_download(self) -> List[dict]:
        """Return optional directory entries (e.g. .mlmodelc) for current platform."""
        import platform as _plat
        current_platform = _plat.system().lower()
        dirs = []
        for entry in self.spec.get("optional_files", []):
            if entry.get("type") != "directory":
                continue
            plat = entry.get("platform", "")
            if plat and plat != current_platform:
                continue
            dirs.append(entry)
        return dirs

    def _list_repo_tree(self, endpoint: "_Endpoint", rel_dir: str) \
            -> Optional[List[dict]]:
        """List files under ``rel_dir`` for ``endpoint``.

        Returns ``[{path, size}]`` (relative paths from repo root) on
        success, or None on failure. Each style uses a different listing
        API:
          * hf / hf_mirror → huggingface_hub.HfApi.list_repo_tree
          * modelscope     → ModelScope JSON files API
        """
        if endpoint.style == "ms":
            return _list_modelscope_tree(endpoint, rel_dir)

        try:
            from huggingface_hub import HfApi, RepoFile
        except ImportError:
            logger.warning("huggingface_hub not available, "
                           "cannot list directory via %s", endpoint.name)
            return None
        try:
            api = HfApi(endpoint=endpoint.base_url)
            tree = list(api.list_repo_tree(
                endpoint.repo, path_in_repo=rel_dir, recursive=True))
        except Exception as e:
            logger.warning("list_repo_tree failed on %s: %s", endpoint.name, e)
            return None
        out: List[dict] = []
        for item in tree:
            if isinstance(item, RepoFile):
                out.append({
                    "path": item.rfilename,
                    "size": getattr(item, "size", 0) or 0,
                })
        return out

    def _download_directory(self, dir_entry: dict):
        """Download a directory tree (e.g. .mlmodelc).

        Walks the endpoint chain to find a working listing endpoint and
        locks onto it for the rest of the model. Per-file downloads
        delegate to _download_one which has its own per-endpoint fallback.
        """
        rel_dir = dir_entry["path"]
        dest_dir = self.dest_dir / rel_dir

        if dest_dir.is_dir() and any(dest_dir.iterdir()):
            logger.info("Skipping directory (exists): %s", rel_dir)
            return

        logger.info("Downloading directory: %s", rel_dir)

        file_list: Optional[List[dict]] = None
        winning_endpoint: Optional[_Endpoint] = None
        for ep in self._resolved_chain():
            file_list = self._list_repo_tree(ep, rel_dir)
            if file_list is not None:
                winning_endpoint = ep
                break

        if file_list is None or winning_endpoint is None:
            logger.warning("Unable to list directory %s from any endpoint", rel_dir)
            return
        if not file_list:
            logger.warning("No files found in directory: %s (via %s)",
                           rel_dir, winning_endpoint.name)
            return

        # Lock the chain to whichever endpoint produced a listing — it'll
        # almost certainly be reachable for the per-file downloads too.
        if self._get_locked_endpoint() is None:
            self._set_locked_endpoint(
                winning_endpoint.name, f"list_repo_tree via {winning_endpoint.name}")

        try:
            for finfo in file_list:
                if self._cancel.is_set():
                    raise Exception("Cancelled by user")
                self._download_one(finfo)
            logger.info("Directory download complete: %s (%d files)",
                        rel_dir, len(file_list))
        except Exception as e:
            logger.warning("Failed to download directory %s: %s", rel_dir, e)

    def _send_head_for_count(self, rel_path: str):
        """Optional HEAD ping to HF main to bump the download counter.

        Default OFF — can be enabled with ``"telemetry_head_count": true``.
        Runs in a daemon thread so it never blocks the actual download. Uses
        a proxy-free opener so Clash/V2Ray can't hijack the request.

        Always pings HF main regardless of which endpoint actually serves the
        bytes (the counter only lives on huggingface.co)."""
        if not _should_telemetry_head():
            return

        def _head():
            try:
                import urllib.request
                hf_only = _Endpoint(
                    "hf", HF_MAIN_ENDPOINT, self.hf_repo, "hf", "main")
                url = _resolve_download_url(rel_path, hf_only)
                req = urllib.request.Request(url, method="HEAD")
                req.add_header("User-Agent", _DEFAULT_UA)
                _urllib_opener().open(req, timeout=5)
                logger.debug("HEAD count OK: %s", rel_path)
            except Exception as e:
                logger.debug("HEAD count failed: %s (%s, non-fatal)", rel_path, e)
        threading.Thread(target=_head, daemon=True).start()

    def _update_aggregate_progress(self, rel_path: str, chunk_bytes: int,
                                   file_done: int):
        """Thread-safe: accumulate per-file progress and push to UI."""
        with self._lock:
            self._file_progress[rel_path] = file_done
            self._speed_window_bytes += chunk_bytes
            total_done = sum(self._file_progress.values())
            now = time.time()
            elapsed = now - self._speed_window_start
            speed = self._speed_window_bytes / max(elapsed, 0.01) if elapsed > 0 else 0
            if elapsed > 5.0:
                self._speed_window_start = now
                self._speed_window_bytes = 0
            self.progress.bytes_done = total_done
            self.progress.speed_bps = speed
            self.progress.file_index = self._completed_count
        self._notify()

    def _multipart_download(self, url: str, dest: Path, rel_path: str,
                            expected_size: int) -> bool:
        """Download ``url`` using ``MULTIPART_PARTS`` parallel Range requests.

        Returns True on success. Returns False if the server doesn't support
        Range (HTTP 200 instead of 206) so the caller can fall back to a
        single-connection download. Raises on any network failure mid-way.

        Assumptions (enforced by caller):
          * expected_size >= MULTIPART_MIN_SIZE (partitioning makes sense)
          * resume_from == 0 (we always start from scratch and pre-allocate)
          * endpoint has already been decided (no speed probe here)
        """
        import requests
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 1) Range-capability probe. We ask for byte 0-0 and check for 206.
        try:
            probe = requests.get(
                url,
                headers={"User-Agent": _DEFAULT_UA, "Range": "bytes=0-0"},
                timeout=(CONNECT_TIMEOUT, MULTIPART_PROBE_TIMEOUT),
                stream=True,
                allow_redirects=True,
                proxies=_requests_proxies(),
            )
            status = probe.status_code
            probe.close()
        except Exception as e:
            logger.debug("Range probe raised (%s) — falling back", e)
            return False

        if status != 206:
            logger.info(
                "Server returned %d for Range probe on %s — falling back "
                "to single connection", status, rel_path,
            )
            return False

        # 2) Pre-allocate the target file. Using truncate() creates a sparse
        # file on APFS / ext4 / NTFS so this is instant.
        try:
            with open(dest, "wb") as f:
                f.truncate(expected_size)
        except OSError as e:
            logger.warning("Pre-allocation failed for %s: %s — falling back",
                           rel_path, e)
            return False

        # 3) Slice into MULTIPART_PARTS byte ranges. Last part absorbs the
        # remainder so rounding doesn't drop the final bytes.
        part_size = expected_size // MULTIPART_PARTS
        ranges: List[tuple] = []
        for i in range(MULTIPART_PARTS):
            start = i * part_size
            end = (i + 1) * part_size - 1 if i < MULTIPART_PARTS - 1 \
                else expected_size - 1
            ranges.append((i, start, end))

        parts_done = [0] * MULTIPART_PARTS  # bytes written per part
        failure = threading.Event()          # any worker flips this on failure

        def _fetch_range(idx: int, start: int, end: int) -> None:
            local_done = 0
            last_report = time.time()
            try:
                with requests.get(
                    url,
                    headers={
                        "User-Agent": _DEFAULT_UA,
                        "Range": f"bytes={start}-{end}",
                    },
                    stream=True,
                    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                    allow_redirects=True,
                    proxies=_requests_proxies(),
                ) as resp:
                    resp.raise_for_status()
                    # NB: each thread opens its own FD to avoid seek/write
                    # races. Writes go to disjoint byte ranges so there is
                    # no overlap.
                    with open(dest, "r+b") as f:
                        f.seek(start)
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            if failure.is_set() or self._cancel.is_set():
                                raise Exception("Cancelled by user")
                            if not chunk:
                                continue
                            f.write(chunk)
                            local_done += len(chunk)
                            parts_done[idx] = local_done

                            now = time.time()
                            if now - last_report >= 0.3:
                                total_done = sum(parts_done)
                                self._update_aggregate_progress(
                                    rel_path, len(chunk), total_done)
                                last_report = now
            except Exception:
                failure.set()
                raise

        # 4) Launch all workers; bail the moment any one fails.
        try:
            with ThreadPoolExecutor(max_workers=MULTIPART_PARTS,
                                    thread_name_prefix="mpart") as pool:
                futures = [pool.submit(_fetch_range, i, s, e)
                           for (i, s, e) in ranges]
                for fut in as_completed(futures):
                    exc = fut.exception()
                    if exc is not None:
                        failure.set()
                        # Wait for remaining workers to exit cleanly — their
                        # writes to the file are already abandoned; the
                        # caller will re-download the whole file.
                        raise exc
        except Exception:
            # Leave the (possibly partial) file on disk so the caller's
            # retry can decide whether to resume or discard. The
            # _sanitize_resume() helper will keep actual<=expected bytes,
            # but since we pre-allocated to expected_size the size check
            # can't distinguish partial from complete. So discard on
            # failure to avoid a false "complete" verdict.
            try:
                dest.unlink()
            except OSError:
                pass
            raise

        # 5) Final progress flush.
        total_done = sum(parts_done)
        self._update_aggregate_progress(rel_path, 0, total_done)
        return True

    def _stream_download(self, url: str, dest: Path, rel_path: str,
                         expected_size: int, resume_from: int = 0,
                         probe_slow_speed: bool = False) -> bool:
        """Stream-download a file with real progress reporting.

        Returns:
            True  — download finished (caller must still size-verify).
            False — speed too slow / timeout (only when ``probe_slow_speed``
                    is set; signals the caller to switch to the next
                    endpoint in the chain).

        Raises any network/HTTP exception so callers can fall back quickly.
        """
        import requests

        # We probe slow-speed on the *first* attempt of the *first* endpoint
        # (chain head) so we can quickly bail to the mirror / ModelScope if
        # HF is reachable-but-throttled. After endpoint lock-in the probe is
        # off; we trust the choice and let the download finish.
        probing = bool(probe_slow_speed)

        headers = {"User-Agent": _DEFAULT_UA}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        resp = requests.get(url, headers=headers, stream=True,
                            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                            allow_redirects=True,
                            proxies=_requests_proxies())
        resp.raise_for_status()

        done = resume_from
        mode = "ab" if resume_from > 0 else "wb"
        t0 = time.time()
        last_report = t0
        local_bytes = 0

        with open(dest, mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if self._cancel.is_set():
                    resp.close()
                    raise Exception("Cancelled by user")

                f.write(chunk)
                done += len(chunk)
                local_bytes += len(chunk)

                now = time.time()

                # Slow-speed probe — only runs while caller asked for it.
                # Returning False signals "switch endpoint", lets the chain
                # walker promote us to the next endpoint without burning a
                # retry attempt.
                if probing:
                    if local_bytes >= SPEED_TEST_BYTES:
                        speed = local_bytes / max(now - t0, 0.01)
                        if speed < MIN_SPEED_BPS:
                            resp.close()
                            logger.info("Speed below %d KB/s (%.0f KB/s) — "
                                        "switching endpoint",
                                        MIN_SPEED_BPS // 1024, speed / 1024)
                            return False
                        logger.info("Endpoint speed OK (%.0f KB/s) — "
                                    "continuing", speed / 1024)
                        probing = False
                    elif now - t0 > SPEED_TEST_TIMEOUT:
                        resp.close()
                        logger.info("Speed probe timeout (%ds, %d KB) — "
                                    "switching endpoint",
                                    SPEED_TEST_TIMEOUT, local_bytes // 1024)
                        return False

                if now - last_report >= 0.3:
                    self._update_aggregate_progress(rel_path, len(chunk), done)
                    last_report = now

        self._update_aggregate_progress(rel_path, 0, done)
        return True

    def _sanitize_resume(self, dest_file: Path, rel_path: str,
                         expected_size: int) -> int:
        """Decide how many bytes we can safely resume from.

        Returns the resume offset (0 = start over). If the existing file is
        bigger than expected (e.g. a previous run wrote a proxy error page
        into it, or an Apple AV scanner appended quarantine metadata), we
        delete it — continuing would produce a size mismatch only at the
        very end, after wasting the user's time re-downloading everything."""
        if not dest_file.exists():
            return 0
        actual = dest_file.stat().st_size
        if expected_size > 0 and actual > expected_size:
            logger.warning(
                "Local %s is larger than expected (%d > %d) — discarding",
                rel_path, actual, expected_size,
            )
            try:
                dest_file.unlink()
            except OSError as e:
                logger.warning("Failed to remove oversized local file: %s", e)
            return 0
        # actual <= expected: safe to resume. A zero-byte leftover is fine
        # (resume=0 is effectively a fresh download).
        return actual

    def _download_file_from(self, url: str, dest_file: Path, rel_path: str,
                            expected_size: int, resume_from: int,
                            probe_slow_speed: bool = False) -> bool:
        """Unified download wrapper: tries multipart first when eligible,
        else falls back to single-connection ``_stream_download``.

        Returns True on normal completion, False only for the slow-speed
        signal (bubbled up from _stream_download — caller should switch
        endpoints). Raises on genuine failures.
        """
        # Multipart needs a stable endpoint to slice the file across — if we
        # haven't locked in yet (chain head, probe still active) we stick
        # with single-conn so a slow probe can fail fast.
        eligible = (
            expected_size >= MULTIPART_MIN_SIZE
            and resume_from == 0
            and not probe_slow_speed
        )
        if eligible:
            try:
                logger.info(
                    "Multipart download %s (%s, %d parts)",
                    rel_path, _fmt_size(expected_size), MULTIPART_PARTS,
                )
                if self._multipart_download(url, dest_file, rel_path,
                                            expected_size):
                    return True
                # False = Range not supported; fall through to single conn.
                logger.info("Multipart not supported for %s — single conn",
                            rel_path)
            except Exception as e:
                # Partial file already discarded by _multipart_download.
                # Bubble up so caller can retry (possibly via mirror).
                if "Cancelled" in str(e):
                    raise
                logger.warning(
                    "Multipart failed for %s (%s) — will fall back to "
                    "single connection on next attempt", rel_path, e,
                )
                raise

        return self._stream_download(url, dest_file, rel_path,
                                     expected_size, resume_from,
                                     probe_slow_speed=probe_slow_speed)

    def _download_one(self, finfo: dict):
        """Download a single file with endpoint-chain fallback + retry.

        Strategy (per file):
          1. If an endpoint is already locked in (e.g. by an earlier file or
             the speed probe), only try that one with full retry budget.
          2. Otherwise walk the chain in priority order — HF main → HF
             mirror → ModelScope by default. Each endpoint gets ONE shot
             before we move to the next; once all endpoints have been tried
             we burn retry attempts.
        """
        rel_path = finfo["path"]
        expected_size = finfo.get("size", 0)
        dest_file = self.dest_dir / rel_path

        if dest_file.exists() and expected_size > 0:
            if dest_file.stat().st_size == expected_size:
                logger.info("Skipping (complete): %s", rel_path)
                with self._lock:
                    self._file_progress[rel_path] = expected_size
                    self._completed_count += 1
                    self.progress.file_index = self._completed_count
                self._notify()
                return

        dest_file.parent.mkdir(parents=True, exist_ok=True)
        self._send_head_for_count(rel_path)

        last_err: Optional[BaseException] = None
        for attempt in range(MAX_RETRIES):
            if self._cancel.is_set():
                raise Exception("Cancelled by user")

            if attempt > 0:
                wait = RETRY_BACKOFF[min(attempt - 1, len(RETRY_BACKOFF) - 1)]
                logger.warning("Retry %d/%d for %s (wait %ds)",
                               attempt + 1, MAX_RETRIES, rel_path, wait)
                time.sleep(wait)

            chain = self._resolved_chain()
            chain_locked = self._get_locked_endpoint() is not None

            for ep_idx, endpoint in enumerate(chain):
                if self._cancel.is_set():
                    raise Exception("Cancelled by user")

                resume_from = self._sanitize_resume(
                    dest_file, rel_path, expected_size)
                url = _resolve_download_url(rel_path, endpoint)
                # Speed probe only on first attempt of the chain head, while
                # we're still picking. After lock-in we trust the choice.
                probe_slow = (
                    not chain_locked
                    and ep_idx == 0
                    and attempt == 0
                    and len(chain) > 1
                )
                try:
                    logger.info(
                        "Downloading %s via %s (resume=%d%s)",
                        rel_path, endpoint.name, resume_from,
                        ", probing speed" if probe_slow else "",
                    )
                    ok = self._download_file_from(
                        url, dest_file, rel_path, expected_size, resume_from,
                        probe_slow_speed=probe_slow,
                    )
                    if ok:
                        if self._verify_size(dest_file, rel_path, expected_size):
                            self._mark_file_done(rel_path)
                            if not chain_locked:
                                self._set_locked_endpoint(
                                    endpoint.name,
                                    f"{endpoint.name} succeeded for {rel_path}",
                                )
                            return
                        raise IOError(
                            f"Size mismatch via {endpoint.name}: {rel_path}")
                    # ok == False → slow probe says move on
                    logger.info("%s too slow — trying next endpoint",
                                endpoint.name)
                    last_err = Exception(f"{endpoint.name} too slow")
                    continue
                except Exception as e:
                    err_msg = str(e)
                    if "Cancelled" in err_msg:
                        raise
                    last_err = e
                    logger.warning("%s failed for %s: %s",
                                   endpoint.name, rel_path, err_msg)
                    # Fall through to next endpoint in the chain
                    continue

            # All endpoints in the chain failed for this attempt — loop
            # around to retry (with backoff) before giving up.

        raise Exception(
            f"Failed after {MAX_RETRIES} retries across "
            f"{len(self._resolved_chain())} endpoint(s): {rel_path}: {last_err}"
        )

    def _verify_size(self, dest_file: Path, rel_path: str,
                     expected_size: int) -> bool:
        if expected_size <= 0 or not dest_file.exists():
            return True
        actual = dest_file.stat().st_size
        if actual != expected_size:
            logger.warning("Size mismatch: %s (expected %d, got %d)",
                           rel_path, expected_size, actual)
            return False
        return True

    def _mark_file_done(self, rel_path: str):
        with self._lock:
            self._completed_count += 1
            self.progress.file_index = self._completed_count
        dest_file = self.dest_dir / rel_path
        if dest_file.exists():
            logger.info("Done: %s (%s)", rel_path,
                        _fmt_size(dest_file.stat().st_size))
        else:
            logger.info("Done: %s", rel_path)

    def _probe_speed(self, files: List[dict]) -> List[dict]:
        """Download the smallest file first to lock in an endpoint.

        ``_download_one`` already walks the chain, runs the speed probe and
        locks the endpoint on first success — this method just picks the
        cheapest file to use as the probe so we don't waste a 5GB GGUF
        downloading from the wrong source.

        If only one endpoint exists (rare; user disabled the others), no
        probing is needed — just lock onto it and skip ahead.

        Returns the remaining files to download in parallel.
        """
        if len(self._endpoint_chain) == 1:
            self._set_locked_endpoint(
                self._endpoint_chain[0].name, "single endpoint")
            return files

        probe = min(files, key=lambda f: f.get("size", 0))
        rest = [f for f in files if f["path"] != probe["path"]]

        logger.info("Speed probe: %s (%s)", probe["path"],
                    _fmt_size(probe.get("size", 0)))
        self._download_one(probe)  # raises only if all endpoints fail

        if self._get_locked_endpoint() is None:
            # Probe completed without explicit lock (e.g. probe file was too
            # small for the speed test to trigger). Default-lock to chain
            # head — it succeeded, so it's at least reachable.
            self._set_locked_endpoint(
                self._endpoint_chain[0].name, "probe finished without lock")
        return rest

    def _run(self):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        files = self._files_to_download()
        total_bytes = sum(f.get("size", 0) for f in files)
        self.progress.total_files = len(files)
        self.progress.bytes_total = total_bytes
        self.progress.status = "downloading"
        self._speed_window_start = time.time()
        self._notify()

        try:
            remaining = self._probe_speed(files)
        except Exception as e:
            logger.exception("Speed probe failed")
            self.progress.status = "error"
            self.progress.error = str(e)
            self._notify()
            return

        if self._cancel.is_set():
            self.progress.status = "error"
            self.progress.error = "Cancelled"
            self._notify()
            return

        # Parallel download of remaining files
        if remaining:
            workers = min(MAX_PARALLEL, len(remaining))
            self.progress.active_workers = workers
            self._notify()
            logger.info("Parallel download: %d files, %d workers", len(remaining), workers)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._download_one, f): f for f in remaining}
                for future in as_completed(futures):
                    finfo = futures[future]
                    exc = future.exception()
                    if exc:
                        logger.error("Download failed: %s — %s",
                                     finfo["path"], exc, exc_info=exc)
                        self._cancel.set()
                        self.progress.status = "error"
                        self.progress.error = f"{finfo['path']}: {exc}"
                        self._notify()
                        pool.shutdown(wait=False, cancel_futures=True)
                        return

        self.progress.active_workers = 0

        # Download optional directories (e.g. CoreML .mlmodelc)
        if not self._cancel.is_set():
            for dir_entry in self._directories_to_download():
                if self._cancel.is_set():
                    break
                try:
                    self._download_directory(dir_entry)
                except Exception as e:
                    logger.warning("Optional directory download failed (non-fatal): %s — %s",
                                   dir_entry["path"], e)

        self.progress.status = "verifying"
        self._notify()

        vr = verify_model_from_spec(str(self.dest_dir), self.spec, self.quant)
        if not vr.complete:
            self.progress.status = "error"
            self.progress.error = f"Verification failed: missing {vr.missing}"
            self._notify()
            return

        self.progress.status = "done"
        self.progress.bytes_done = total_bytes
        self._notify()
        logger.info("Model download complete: %s (%s)", self.spec["display_name"], self.quant)

    def _notify(self):
        if self._progress_cb:
            try:
                self._progress_cb(self.progress)
            except Exception:
                pass

    def start(self, progress_cb: Optional[Callable[[DownloadProgress], None]] = None):
        """Start download in background thread."""
        self._progress_cb = progress_cb
        self._cancel.clear()
        self._completed_count = 0
        self._file_progress.clear()
        self._speed_window_bytes = 0
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def wait(self, timeout: Optional[float] = None):
        if self._thread:
            self._thread.join(timeout=timeout)


def _fmt_size(n: int) -> str:
    if n >= 1024**3:
        return f"{n/1024**3:.1f} GB"
    if n >= 1024**2:
        return f"{n/1024**2:.1f} MB"
    return f"{n/1024:.0f} KB"
