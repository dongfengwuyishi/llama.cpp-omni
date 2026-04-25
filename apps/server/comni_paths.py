"""Single source of truth for user data paths (models, cache, config).

Resolution order (high → low):

    1. Environment variable
         - COMNI_HOME   → overrides the entire ~/.comni/ root
         - COMNI_MODELS → overrides just the models directory
       Same spirit as Ollama's OLLAMA_MODELS — friendly to power users
       and CI / scripts without touching any config file.

    2. ~/.comni/config.json
         - models_home → set by the desktop app's settings dialog
       Persistent so the GUI can move models to a non-default drive once
       and remember it.

    3. Default
         - ~/.comni/models/      (Windows: %USERPROFILE%\\.comni\\models\\)

The point of this indirection is to keep huge GGUF files off the
system drive on Windows machines whose C: is small.

All three callers (model_hub.py / windows_app.py / menubar_app.py) MUST
go through these helpers — otherwise overrides won't take effect.
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("comni.paths")

_DEFAULT_HOME_NAME = ".comni"


def _expand(p: str | os.PathLike) -> Path:
    """Resolve ~ and environment vars; return an absolute path."""
    return Path(os.path.expandvars(str(p))).expanduser().resolve()


def comni_home() -> Path:
    """Root user data directory (default: ~/.comni)."""
    env = os.environ.get("COMNI_HOME")
    if env:
        return _expand(env)
    return Path.home() / _DEFAULT_HOME_NAME


def comni_config_path() -> Path:
    """Path to the JSON file that persists user preferences."""
    return comni_home() / "config.json"


def cache_dir() -> Path:
    """Where ephemeral things (HF manifest cache, etc.) live."""
    return comni_home() / "cache"


def _read_user_config() -> dict:
    p = comni_config_path()
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # corrupt JSON, missing perms, etc.
        logger.warning("failed to read %s: %s", p, e)
        return {}


def _write_user_config(data: dict) -> None:
    p = comni_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


def models_home() -> Path:
    """Where downloaded / imported models are stored.

    Order: COMNI_MODELS env > config.json["models_home"] > <comni_home>/models
    """
    env = os.environ.get("COMNI_MODELS")
    if env:
        return _expand(env)
    saved = _read_user_config().get("models_home")
    if saved:
        try:
            return _expand(saved)
        except Exception as e:
            logger.warning("invalid models_home in config (%s): %s", saved, e)
    return comni_home() / "models"


def set_models_home(path: str | os.PathLike) -> Path:
    """Persist a new models directory in ~/.comni/config.json.

    Creates the directory if missing. Returns the resolved Path.
    Note: this does NOT move existing model files — callers are responsible
    for that (typically the GUI offers an explicit move option).
    """
    p = _expand(path)
    p.mkdir(parents=True, exist_ok=True)
    cfg = _read_user_config()
    cfg["models_home"] = str(p)
    _write_user_config(cfg)
    logger.info("models_home set to %s", p)
    return p


def clear_models_home_override() -> None:
    """Remove the persistent override; revert to env-var or default."""
    cfg = _read_user_config()
    if "models_home" in cfg:
        cfg.pop("models_home")
        _write_user_config(cfg)
        logger.info("cleared models_home override")


def models_home_source() -> str:
    """Diagnostic: where is the current models_home value coming from?

    Returns one of: "env", "config", "default".
    """
    if os.environ.get("COMNI_MODELS"):
        return "env"
    if _read_user_config().get("models_home"):
        return "config"
    return "default"


# ── Free disk-space helper (used by GUI before downloading) ──────────────

def free_bytes(path: str | os.PathLike) -> Optional[int]:
    """Return free bytes on the volume containing `path`, or None if unknown.

    Walks up the path until something exists; useful when the target dir
    hasn't been created yet.
    """
    p = Path(path).expanduser()
    while not p.exists():
        if p.parent == p:
            return None
        p = p.parent
    try:
        import shutil
        return shutil.disk_usage(p).free
    except Exception:
        return None
