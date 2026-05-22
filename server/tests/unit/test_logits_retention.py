"""Unit tests for ``core.processors.logits_retention``.

Covers:
  * ``resolve_output_dir`` honors spec.output_dir / env / default in the
    documented priority order.
  * ``resolve_output_dir`` is idempotent (re-bucketing doesn't nest).
  * ``cleanup_once`` deletes only buckets strictly older than the
    retention window.
  * Size-cap eviction drops oldest-first while never wiping the most
    recent (active) bucket.

We DO NOT spawn the daemon thread in unit tests — its loop is just
``cleanup_once`` in a sleep. Testing ``cleanup_once`` covers the logic.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


@pytest.fixture
def fresh_env(monkeypatch):
    """Wipe all OMNI_LOGITS_* env so tests start from defaults."""
    for k in list(os.environ.keys()):
        if k.startswith("OMNI_LOGITS_"):
            monkeypatch.delenv(k, raising=False)
    yield monkeypatch


def _touch_bucket(base: Path, date_str: str, n_files: int, file_size: int = 1024) -> Path:
    """Create ``base/<date_str>`` with ``n_files`` files of given size."""
    bucket = base / date_str
    bucket.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (bucket / f"f{i}.safetensors").write_bytes(b"x" * file_size)
    return bucket


def test_resolve_output_dir_with_explicit_spec(tmp_path, fresh_env):
    from core.processors.logits_retention import resolve_output_dir, _today_bucket

    out = resolve_output_dir(str(tmp_path))
    today = _today_bucket()
    assert os.path.isdir(out), f"resolve_output_dir should mkdir: {out}"
    assert out == str(tmp_path / today)


def test_resolve_output_dir_uses_env_when_spec_none(tmp_path, fresh_env):
    fresh_env.setenv("OMNI_LOGITS_OUTPUT_DIR", str(tmp_path / "custom_base"))
    from core.processors.logits_retention import resolve_output_dir, _today_bucket

    out = resolve_output_dir(None)
    today = _today_bucket()
    assert out == str(tmp_path / "custom_base" / today)
    assert os.path.isdir(out)


def test_resolve_output_dir_is_idempotent_when_already_bucketed(tmp_path, fresh_env):
    """If caller passes ``base/<YYYY-MM-DD>`` we shouldn't double-nest."""
    from core.processors.logits_retention import resolve_output_dir, _today_bucket

    today = _today_bucket()
    pre_bucketed = tmp_path / today
    out = resolve_output_dir(str(pre_bucketed))
    # Should still write into base/today, not base/today/today.
    assert out == str(tmp_path / today)
    assert "/" + today + "/" + today not in out


def test_cleanup_drops_only_buckets_older_than_retention(tmp_path, fresh_env):
    from core.processors.logits_retention import cleanup_once

    today = datetime.now(timezone.utc).date()

    today_b   = _touch_bucket(tmp_path, today.strftime("%Y-%m-%d"), n_files=2)
    yest_b    = _touch_bucket(tmp_path, (today - timedelta(days=1)).strftime("%Y-%m-%d"), 2)
    week_old  = _touch_bucket(tmp_path, (today - timedelta(days=7)).strftime("%Y-%m-%d"), 2)
    month_old = _touch_bucket(tmp_path, (today - timedelta(days=30)).strftime("%Y-%m-%d"), 2)

    n_age, n_size = cleanup_once(
        base=str(tmp_path), retention_days=7, max_total_bytes=0
    )
    # 7-day window means buckets strictly older than (today - 7 days)
    # are dropped: month_old falls, the 7-day-old bucket is on the boundary
    # (cutoff = today - 7 → "today - 7" is NOT strictly older, so kept).
    assert month_old.exists() is False
    assert week_old.exists() is True
    assert yest_b.exists()  is True
    assert today_b.exists() is True
    assert n_age == 1
    assert n_size == 0


def test_cleanup_size_cap_evicts_oldest_first_keeps_active(tmp_path, fresh_env):
    from core.processors.logits_retention import cleanup_once

    today = datetime.now(timezone.utc).date()

    # 4 buckets, ~10KB each => 40KB total. Cap at 25KB → must drop the
    # 2 oldest. The most-recent bucket must NEVER be dropped (active
    # writer protection).
    b_d4 = _touch_bucket(tmp_path, (today - timedelta(days=4)).strftime("%Y-%m-%d"), 1, 10_000)
    b_d3 = _touch_bucket(tmp_path, (today - timedelta(days=3)).strftime("%Y-%m-%d"), 1, 10_000)
    b_d2 = _touch_bucket(tmp_path, (today - timedelta(days=2)).strftime("%Y-%m-%d"), 1, 10_000)
    b_today = _touch_bucket(tmp_path, today.strftime("%Y-%m-%d"), 1, 10_000)

    n_age, n_size = cleanup_once(
        base=str(tmp_path), retention_days=0, max_total_bytes=25_000
    )

    assert b_today.exists(), "active (most-recent) bucket must be preserved"
    assert b_d4.exists() is False
    assert b_d3.exists() is False
    assert b_d2.exists() is True
    assert n_age == 0
    assert n_size == 2


def test_cleanup_disabled_when_both_zero(tmp_path, fresh_env):
    from core.processors.logits_retention import cleanup_once

    today = datetime.now(timezone.utc).date()
    old = _touch_bucket(tmp_path, (today - timedelta(days=365)).strftime("%Y-%m-%d"), 1)
    new = _touch_bucket(tmp_path, today.strftime("%Y-%m-%d"), 1)

    n_age, n_size = cleanup_once(
        base=str(tmp_path), retention_days=0, max_total_bytes=0
    )
    assert old.exists() and new.exists()
    assert n_age == 0 and n_size == 0


def test_cleanup_ignores_unrelated_subdirs(tmp_path, fresh_env):
    from core.processors.logits_retention import cleanup_once

    today = datetime.now(timezone.utc).date()
    _touch_bucket(tmp_path, (today - timedelta(days=30)).strftime("%Y-%m-%d"), 1)
    not_a_date = tmp_path / "scratch"
    not_a_date.mkdir()
    (not_a_date / "important.txt").write_bytes(b"hello")

    cleanup_once(base=str(tmp_path), retention_days=7, max_total_bytes=0)
    # Non-date subdirs must be left alone — janitor only owns YYYY-MM-DD.
    assert (not_a_date / "important.txt").exists()


def test_start_cleanup_thread_disabled_returns_none(tmp_path, fresh_env):
    fresh_env.setenv("OMNI_LOGITS_RETENTION_DAYS", "0")
    fresh_env.setenv("OMNI_LOGITS_MAX_TOTAL_BYTES", "0")
    from core.processors.logits_retention import start_cleanup_thread

    t = start_cleanup_thread()
    assert t is None
