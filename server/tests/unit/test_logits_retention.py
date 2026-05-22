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


# ---------------------------------------------------------------------------
# make_logits_filename — 多 worker 命名防撞
# ---------------------------------------------------------------------------
#
# 不变量（必须）：
#   1. 同一 (worker_idx, pid, seq) 三元组下文件名唯一
#   2. 跨 worker_idx 永不撞名（即使 pid + seq 相同）
#   3. client request_id 不参与唯一性，纯 debug 后缀；脏 unicode / 路径
#      分隔符全部 sanitize 成 ``_``
#   4. 进程内 atomic seq 在多线程并发 ``next()`` 下不重复（itertools.count
#      在 CPython GIL 下原子）
#   5. 默认前缀允许的字符集 ``[A-Za-z0-9_]+`` —— 不能出现 ``/`` ``..`` ``\``
#      让 client 的 request_id 越权写到上级目录
import re
import threading


_FILENAME_RE = re.compile(
    r"^(chat|duplex)_w(\d+)_p([0-9a-f]{5})_(\d{8})(?:_([A-Za-z0-9_]+))?\.safetensors$"
)


def test_make_logits_filename_basic_format():
    from core.processors.logits_retention import make_logits_filename

    fn = make_logits_filename("chat", 0, None,
                              _pid_override=0x1F4A, _seq_override=123)
    m = _FILENAME_RE.match(fn)
    assert m, f"format mismatch: {fn!r}"
    assert m.group(1) == "chat"
    assert m.group(2) == "0"
    assert m.group(3) == "01f4a"
    assert m.group(4) == "00000123"
    assert m.group(5) is None
    assert fn == "chat_w0_p01f4a_00000123.safetensors"


def test_make_logits_filename_with_request_id_appended():
    from core.processors.logits_retention import make_logits_filename

    fn = make_logits_filename("duplex", 2, "user_abc",
                              _pid_override=0x3B81, _seq_override=456)
    assert fn == "duplex_w2_p03b81_00000456_user_abc.safetensors"


def test_make_logits_filename_kind_validation():
    from core.processors.logits_retention import make_logits_filename

    with pytest.raises(ValueError):
        make_logits_filename("invalid_kind", 0, None)


def test_make_logits_filename_sanitizes_path_traversal():
    """Hostile request_id must not let client escape the bucket dir."""
    from core.processors.logits_retention import make_logits_filename

    for evil in [
        "../../etc/passwd",
        "/abs/path/to/somewhere",
        "..\\..\\Windows",
        "with spaces",
        "rid;rm -rf /",
        "中文请求",        # all non-ASCII → fully stripped
        "rid\nwith\tnewlines",
        ".",
        "..",
    ]:
        fn = make_logits_filename("chat", 0, evil,
                                  _pid_override=0x1, _seq_override=1)
        m = _FILENAME_RE.match(fn)
        assert m, f"sanitized name should still match strict regex: {fn!r} (input={evil!r})"
        # The sanitized rid suffix (group 5) must not contain anything
        # outside the safe alphabet; the regex enforces this, but we
        # explicitly check no path separators slipped through.
        rid_part = m.group(5) or ""
        assert "/" not in rid_part
        assert "\\" not in rid_part
        assert ".." not in rid_part


def test_make_logits_filename_request_id_truncated():
    """Long client rids capped to 32 chars to stay under ext4 path limits."""
    from core.processors.logits_retention import make_logits_filename

    long_rid = "a" * 128
    fn = make_logits_filename("chat", 0, long_rid,
                              _pid_override=0x1, _seq_override=0)
    m = _FILENAME_RE.match(fn)
    assert m, fn
    assert m.group(5) is not None
    assert len(m.group(5)) == 32


def test_make_logits_filename_empty_rid_omits_suffix():
    from core.processors.logits_retention import make_logits_filename

    for rid in [None, "", "   ", "!!!", "中文", "..."]:
        fn = make_logits_filename("chat", 0, rid,
                                  _pid_override=0x1, _seq_override=0)
        m = _FILENAME_RE.match(fn)
        assert m, fn
        assert m.group(5) is None, f"expected no rid suffix for {rid!r}, got {fn!r}"


def test_make_logits_filename_seq_increments_atomically_under_threads():
    """1000 threads × 1 call each — produced filenames must all be unique.

    This catches both:
      - itertools.count ``next()`` not being atomic (it is on CPython, so
        we expect this to pass), AND
      - any future refactor that replaces it with a non-atomic counter.
    """
    from core.processors.logits_retention import make_logits_filename

    N = 1000
    out: list = []
    lock = threading.Lock()

    def _worker():
        fn = make_logits_filename("chat", 0, None,
                                  _pid_override=0xABCDE)
        with lock:
            out.append(fn)

    threads = [threading.Thread(target=_worker) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(out) == N
    assert len(set(out)) == N, (
        f"collision detected: {N - len(set(out))} duplicates among {N} names"
    )


def test_make_logits_filename_cross_worker_never_collides():
    """Same (pid, seq) but different worker_idx → must produce distinct names.

    This is the **multi-worker safety property**: even if two worker
    processes happen to land on the same PID-low-bits and same seq value
    (extremely unlikely but possible after fork/exec), the worker_idx in
    the filename guarantees no overwrite.
    """
    from core.processors.logits_retention import make_logits_filename

    names = {
        make_logits_filename("chat", w_idx, None,
                             _pid_override=0xDEAD, _seq_override=42)
        for w_idx in range(8)
    }
    assert len(names) == 8


def test_make_logits_filename_chat_and_duplex_dont_collide():
    """Same worker, same pid, same seq, but different kind → distinct names."""
    from core.processors.logits_retention import make_logits_filename

    a = make_logits_filename("chat", 0, None,
                             _pid_override=0x1, _seq_override=99)
    b = make_logits_filename("duplex", 0, None,
                             _pid_override=0x1, _seq_override=99)
    assert a != b
    assert a.startswith("chat_")
    assert b.startswith("duplex_")
