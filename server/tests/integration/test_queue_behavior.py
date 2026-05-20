"""Integration tests for FIFO queue behaviour visible through HTTP."""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


async def test_queue_status_reflects_pending(batch_server_client):
    def slow(state, _i):
        state.chat_delay = 0.4

    client, _, _ = await batch_server_client(num_workers=1, configure=slow)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    # Fire-and-forget two requests in the background so we can sample the
    # queue mid-flight.
    task1 = asyncio.create_task(client.post("/v1/chat", json=body))
    task2 = asyncio.create_task(client.post("/v1/chat", json=body))

    # Give them a moment to enqueue
    await asyncio.sleep(0.05)

    snap = (await client.get("/v1/queue")).json()
    assert snap["queue_length"] in (0, 1)  # one queued or already dispatched

    await asyncio.gather(task1, task2)


async def test_running_tasks_visible_during_processing(batch_server_client):
    def slow(state, _i):
        state.chat_delay = 0.3

    client, _, _ = await batch_server_client(num_workers=2, configure=slow)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    t1 = asyncio.create_task(client.post("/v1/chat", json=body))
    t2 = asyncio.create_task(client.post("/v1/chat", json=body))
    await asyncio.sleep(0.05)

    status = (await client.get("/v1/status")).json()
    assert status["busy_workers"] >= 1
    assert any(t["request_type"] == "chat" for t in status["running_tasks"])

    await asyncio.gather(t1, t2)


async def test_eta_tightens_with_samples(batch_server_client):
    """After several quick requests, EMA should drop the chat ETA."""
    def fast(state, _i):
        state.chat_delay = 0.02

    client, _, mod = await batch_server_client(num_workers=1, configure=fast)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    # Send sequentially so each one feeds the EMA
    for _ in range(5):
        r = await client.post("/v1/chat", json=body)
        assert r.status_code == 200

    tracker = mod.worker_pool.eta_tracker
    eta = tracker.get_eta("chat")
    assert eta < 5.0  # baseline is 15s; should drop after 5 fast samples
