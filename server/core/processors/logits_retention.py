"""Logits ``.safetensors`` retention policy.

Concern: ``/v1/chat`` and ``/v1/duplex_offline`` can write per-request
``.safetensors`` blobs (one decode-token row of bf16 logits is ~300 KB; a
single 30-token chat answer is ~9 MB; an RL data collection run easily
generates tens of GB per day). Without a janitor these files stack up in
``/tmp/minicpm_logits`` (the historical default) until the disk fills.

Design:

1. **Date-bucketed layout.** Instead of writing flat into ``output_dir``
   we first append a ``YYYY-MM-DD`` subdirectory, e.g.::

       /data/logits/2026-05-21/chat_round0.safetensors
       /data/logits/2026-05-21/duplex_<rid>.safetensors
       /data/logits/2026-05-22/...

   This makes "delete files older than N days" a cheap directory-level
   operation (one ``rmtree`` per stale bucket, no recursive ``stat``).

2. **Background daemon thread.** Each worker process starts one
   :func:`start_cleanup_thread`-spawned daemon that wakes every
   ``OMNI_LOGITS_CLEANUP_INTERVAL_S`` seconds, drops any date subdir
   strictly older than ``OMNI_LOGITS_RETENTION_DAYS``, and (optionally)
   evicts oldest-first until total size is within
   ``OMNI_LOGITS_MAX_TOTAL_BYTES``. Multiple workers sharing the same base
   dir is fine — ``unlink`` / ``rmtree`` are idempotent and races just
   mean one wins.

3. **Opt-out / opt-in via env.** The whole feature is governed by env
   variables so production / RL pipelines can override without code
   changes:

   - ``OMNI_LOGITS_OUTPUT_DIR``       default base dir if request
                                      doesn't pin one. ``""`` means
                                      ``"/tmp/minicpm_logits"`` (legacy).
   - ``OMNI_LOGITS_RETENTION_DAYS``   integer days. ``0`` disables
                                      age-based eviction. Default ``7``.
   - ``OMNI_LOGITS_MAX_TOTAL_BYTES``  size cap across the whole base
                                      dir. ``0`` disables size-based
                                      eviction. Default ``0``.
   - ``OMNI_LOGITS_CLEANUP_INTERVAL_S`` how often the daemon wakes.
                                      Default ``600`` (10 min).

The "what is the request's actual write-dir" decision is kept as a single
helper :func:`resolve_output_dir` so chat (cpp_backend) and duplex
(worker) paths are reproducibly bucketed.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

logger = logging.getLogger("logits_retention")


# YYYY-MM-DD subdirectory format (UTC date — matches what we write).
_DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Default base dir if neither request nor env pins one. Matches the
# historical default in ``_build_consolidated_logits_payload``.
_DEFAULT_BASE_DIR = "/tmp/minicpm_logits"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            f"[logits_retention] {name}={raw!r} is not an integer; using default {default}"
        )
        return default


def _today_bucket() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def base_dir_from_spec(spec_output_dir: Optional[str]) -> str:
    """Pick the base directory under which date subdirs live.

    Priority:
      1. explicit ``spec_output_dir`` from the request (already strip
         trailing date bucket if caller pre-pinned one — see
         :func:`resolve_output_dir` for the second-level handling)
      2. ``$OMNI_LOGITS_OUTPUT_DIR``
      3. ``/tmp/minicpm_logits``
    """
    if spec_output_dir:
        # Strip any trailing date-bucket if the caller already nested
        # one — keep the resolution idempotent so we don't end up with
        # ``/data/logits/2026-05-21/2026-05-21/...``.
        head, tail = os.path.split(spec_output_dir.rstrip("/\\"))
        if _DATE_DIR_RE.match(tail) and head:
            return head
        return spec_output_dir
    env_dir = os.environ.get("OMNI_LOGITS_OUTPUT_DIR")
    if env_dir:
        return env_dir
    return _DEFAULT_BASE_DIR


def resolve_output_dir(spec_output_dir: Optional[str]) -> str:
    """Return the actual write-target dir for a logits ``.safetensors``.

    Always returns ``<base>/<YYYY-MM-DD>/`` (with trailing separator
    stripped) and ensures it exists. The bucket key is "today in UTC" —
    so a long-running RL pipeline that writes across midnight produces
    one bucket per UTC day, which is the natural retention granularity.

    The function is idempotent: passing it a path that is already
    bucketed (i.e. already ends with ``/YYYY-MM-DD``) won't re-bucket;
    passing it ``None`` falls back to env / default.
    """
    base = base_dir_from_spec(spec_output_dir)
    bucket = os.path.join(base, _today_bucket())
    os.makedirs(bucket, exist_ok=True)
    return bucket


# ---------------------------------------------------------------------------
# Cleanup daemon
# ---------------------------------------------------------------------------

def _list_date_buckets(base: str) -> List[Tuple[str, str]]:
    """Return ``[(bucket_name, full_path), ...]`` sorted by bucket_name asc."""
    if not os.path.isdir(base):
        return []
    out: List[Tuple[str, str]] = []
    for name in os.listdir(base):
        if not _DATE_DIR_RE.match(name):
            continue
        full = os.path.join(base, name)
        if os.path.isdir(full):
            out.append((name, full))
    out.sort(key=lambda x: x[0])
    return out


def _delete_old_buckets(base: str, retention_days: int) -> int:
    """Remove date-bucket subdirs strictly older than ``retention_days``.

    Returns count of buckets removed.
    """
    if retention_days <= 0:
        return 0
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=retention_days))
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    n_removed = 0
    for name, path in _list_date_buckets(base):
        if name < cutoff_str:
            try:
                shutil.rmtree(path, ignore_errors=False)
                n_removed += 1
                logger.info(f"[logits_retention] dropped stale bucket {path}")
            except Exception as e:
                logger.warning(f"[logits_retention] rmtree {path} failed: {e}")
    return n_removed


def _evict_to_size_budget(base: str, max_total_bytes: int) -> int:
    """Oldest-bucket-first eviction to stay under ``max_total_bytes``.

    We delete whole date buckets (not individual files) so we don't end
    up with half-pruned days. Returns count of buckets removed.
    """
    if max_total_bytes <= 0:
        return 0
    buckets = _list_date_buckets(base)
    if not buckets:
        return 0

    def _bucket_size(path: str) -> int:
        total = 0
        for root, _dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        return total

    sizes = [(name, path, _bucket_size(path)) for (name, path) in buckets]
    total = sum(s for _, _, s in sizes)
    if total <= max_total_bytes:
        return 0
    n_removed = 0
    # Delete oldest first (we sorted ascending). Stop once under budget,
    # but never delete the most recent bucket — that's the one this
    # process is actively writing to.
    for i, (name, path, sz) in enumerate(sizes[:-1]):
        if total <= max_total_bytes:
            break
        try:
            shutil.rmtree(path, ignore_errors=False)
            total -= sz
            n_removed += 1
            logger.info(
                f"[logits_retention] size-cap eviction: dropped {path} "
                f"({sz/1024/1024:.1f} MB), total now {total/1024/1024:.1f} MB"
            )
        except Exception as e:
            logger.warning(f"[logits_retention] size-cap rmtree {path} failed: {e}")
    return n_removed


def cleanup_once(
    base: Optional[str] = None,
    retention_days: Optional[int] = None,
    max_total_bytes: Optional[int] = None,
) -> Tuple[int, int]:
    """Run one cleanup pass. Exposed for tests.

    Returns ``(n_age_evicted, n_size_evicted)``.
    """
    if base is None:
        base = base_dir_from_spec(None)
    if retention_days is None:
        retention_days = _env_int("OMNI_LOGITS_RETENTION_DAYS", 7)
    if max_total_bytes is None:
        max_total_bytes = _env_int("OMNI_LOGITS_MAX_TOTAL_BYTES", 0)

    if not os.path.isdir(base):
        return (0, 0)
    n_age = _delete_old_buckets(base, retention_days)
    n_size = _evict_to_size_budget(base, max_total_bytes)
    return (n_age, n_size)


_cleanup_thread: Optional[threading.Thread] = None
_cleanup_stop_event: Optional[threading.Event] = None
_cleanup_lock = threading.Lock()


def start_cleanup_thread() -> Optional[threading.Thread]:
    """Spawn (or return existing) cleanup daemon for this process.

    Idempotent: returns the existing thread if already running.

    Returns ``None`` if retention is fully disabled
    (``OMNI_LOGITS_RETENTION_DAYS=0`` AND ``OMNI_LOGITS_MAX_TOTAL_BYTES=0``)
    so we don't burn a thread doing nothing.
    """
    global _cleanup_thread, _cleanup_stop_event

    retention_days = _env_int("OMNI_LOGITS_RETENTION_DAYS", 7)
    max_total_bytes = _env_int("OMNI_LOGITS_MAX_TOTAL_BYTES", 0)
    if retention_days <= 0 and max_total_bytes <= 0:
        logger.info(
            "[logits_retention] disabled (OMNI_LOGITS_RETENTION_DAYS=0 and "
            "OMNI_LOGITS_MAX_TOTAL_BYTES=0)"
        )
        return None

    with _cleanup_lock:
        if _cleanup_thread is not None and _cleanup_thread.is_alive():
            return _cleanup_thread

        interval_s = max(60, _env_int("OMNI_LOGITS_CLEANUP_INTERVAL_S", 600))
        base = base_dir_from_spec(None)

        stop_event = threading.Event()

        def _loop():
            logger.info(
                f"[logits_retention] janitor started: base={base!r}, "
                f"retention_days={retention_days}, "
                f"max_total_bytes={max_total_bytes}, interval_s={interval_s}"
            )
            while not stop_event.is_set():
                try:
                    n_age, n_size = cleanup_once(base, retention_days, max_total_bytes)
                    if n_age or n_size:
                        logger.info(
                            f"[logits_retention] swept: {n_age} aged buckets, "
                            f"{n_size} size-capped buckets"
                        )
                except Exception as e:
                    logger.error(f"[logits_retention] sweep failed: {e}", exc_info=True)
                stop_event.wait(timeout=interval_s)

        t = threading.Thread(target=_loop, name="logits-retention", daemon=True)
        t.start()
        _cleanup_thread = t
        _cleanup_stop_event = stop_event
        return t


def stop_cleanup_thread(timeout: float = 5.0) -> None:
    """Best-effort graceful stop of the cleanup daemon (used at shutdown)."""
    global _cleanup_thread, _cleanup_stop_event
    with _cleanup_lock:
        if _cleanup_stop_event is not None:
            _cleanup_stop_event.set()
        t = _cleanup_thread
    if t is not None:
        t.join(timeout=timeout)
