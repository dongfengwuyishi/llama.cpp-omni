"""Unit tests for ``gateway_modules.worker_pool.WorkerPool``.

We **do not** spin up real workers here — we patch ``_refresh_worker_status``
to mark workers as IDLE locally. The goal is to exercise FIFO queueing,
dispatching, cancellation and queue-full semantics in isolation.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway_modules.models import EtaConfig, GatewayWorkerStatus
from gateway_modules.worker_pool import WorkerPool

pytestmark = pytest.mark.unit


@pytest.fixture
def pool_factory():
    """Build a pool with N "fake idle" workers (skip real health probe)."""
    pools = []

    async def _build(n: int = 2, max_queue_size: int = 10):
        pool = WorkerPool(
            worker_addresses=[f"127.0.0.1:{40000 + i}" for i in range(n)],
            max_queue_size=max_queue_size,
            request_timeout=10.0,
            eta_config=EtaConfig(),
        )

        # Skip the real network refresh: just mark every worker idle.
        async def _fake_refresh(self):
            for w in self.workers.values():
                w.status = GatewayWorkerStatus.IDLE

        # Disable health-check loop in tests
        async def _no_health(self):
            return

        WorkerPool._refresh_all_status = _fake_refresh  # type: ignore[method-assign]
        WorkerPool._health_check_loop = _no_health  # type: ignore[method-assign]

        await pool.start()
        pools.append(pool)
        return pool

    yield _build

    async def _shutdown():
        for p in pools:
            await p.stop()

    asyncio.get_event_loop().run_until_complete(_shutdown()) if pools else None


# ============================================================
# Basic FIFO behaviour
# ============================================================


async def test_enqueue_immediate_assigns_idle_worker(pool_factory):
    pool = await pool_factory(n=2)
    ticket, future = pool.enqueue("chat")
    assert future.done()
    worker = future.result()
    assert worker is not None
    assert worker.status == GatewayWorkerStatus.BUSY_CHAT
    assert ticket.position == 0


async def test_second_request_waits_when_all_busy(pool_factory):
    pool = await pool_factory(n=1)
    t1, f1 = pool.enqueue("chat")
    assert f1.done()  # immediately assigned

    t2, f2 = pool.enqueue("chat")
    assert not f2.done()
    assert t2.position == 1

    # Release the only worker → second request gets dispatched
    pool.release_worker(f1.result(), request_type="chat", duration_s=0.01)

    # Allow the event loop to flush the future resolution
    await asyncio.sleep(0)
    assert f2.done()
    assert f2.result() is not None


async def test_cancel_releases_slot(pool_factory):
    pool = await pool_factory(n=1)
    t1, f1 = pool.enqueue("chat")
    t2, f2 = pool.enqueue("chat")
    assert not f2.done()

    ok = pool.cancel(t2.ticket_id)
    assert ok is True
    assert f2.cancelled() or f2.done()

    # Cancelling an unknown ticket returns False
    assert pool.cancel("q_doesnotexist") is False


async def test_queue_full(pool_factory):
    pool = await pool_factory(n=1, max_queue_size=2)
    pool.enqueue("chat")            # immediately busy
    pool.enqueue("chat")            # queued #1
    pool.enqueue("chat")            # queued #2 (queue is now full per FIFO)

    with pytest.raises(WorkerPool.QueueFullError):
        pool.enqueue("chat")


async def test_release_triggers_dispatch_for_next(pool_factory):
    pool = await pool_factory(n=1)
    _, f1 = pool.enqueue("chat")
    _, f2 = pool.enqueue("chat")
    _, f3 = pool.enqueue("chat")
    assert f1.done()
    assert not f2.done()
    assert not f3.done()

    pool.release_worker(f1.result(), request_type="chat", duration_s=0.01)
    await asyncio.sleep(0)
    assert f2.done()
    assert not f3.done()

    pool.release_worker(f2.result(), request_type="chat", duration_s=0.01)
    await asyncio.sleep(0)
    assert f3.done()


# ============================================================
# ETA accounting
# ============================================================


async def test_eta_record_duration_updates_ema(pool_factory):
    pool = await pool_factory(n=1)
    tracker = pool.eta_tracker
    base = tracker.get_eta("chat")

    for _ in range(5):
        tracker.record_duration("chat", duration_s=base / 5)

    ema = tracker.get_eta("chat")
    assert ema < base, f"EMA should drop after fast samples, got {ema} vs base {base}"


async def test_queue_status_reports_positions(pool_factory):
    pool = await pool_factory(n=1)
    _, f1 = pool.enqueue("chat")  # immediately busy
    t2, _ = pool.enqueue("chat")
    t3, _ = pool.enqueue("chat")

    status = pool.get_queue_status()
    assert status.queue_length == 2
    positions = sorted(item.position for item in status.items)
    assert positions == [1, 2]
    ticket_ids = {it.ticket_id for it in status.items}
    assert t2.ticket_id in ticket_ids
    assert t3.ticket_id in ticket_ids


async def test_running_tasks_listed(pool_factory):
    pool = await pool_factory(n=2)
    _, f1 = pool.enqueue("chat")
    _, f2 = pool.enqueue("audio_duplex")

    status = pool.get_queue_status()
    types = {t.request_type for t in status.running_tasks}
    assert types == {"chat", "audio_duplex"}
