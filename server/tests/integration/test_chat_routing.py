"""Integration tests for /v1/chat: routing, queueing, success/failure paths."""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


async def test_single_chat_roundtrip(batch_server_client):
    client, procs, _ = await batch_server_client(num_workers=1)
    body = {"messages": [{"role": "user", "content": "hello"}]}
    r = await client.post("/v1/chat", json=body)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["success"] is True
    assert payload["text"].startswith("[mock-reply]")
    assert "queue_wait_ms" in payload
    assert "ticket_id" in payload
    assert payload["worker_id"] == "worker_0"
    # worker received our payload
    assert procs[0].state.chat_calls == 1
    assert procs[0].state.last_chat_payload == body


async def test_chat_with_tts_returns_audio(batch_server_client):
    client, procs, _ = await batch_server_client(num_workers=1)
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tts": {"enabled": True},
    }
    r = await client.post("/v1/chat", json=body)
    assert r.status_code == 200
    payload = r.json()
    assert payload["success"] is True
    assert payload["audio_data"] is not None
    assert payload["audio_sample_rate"] == 24000


async def test_concurrent_requests_queue_up(batch_server_client):
    # one worker, but with a noticeable delay → second request will queue
    def slow(state, _i):
        state.chat_delay = 0.3

    client, procs, _ = await batch_server_client(num_workers=1, configure=slow)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    r1, r2 = await asyncio.gather(
        client.post("/v1/chat", json=body),
        client.post("/v1/chat", json=body),
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    # one of them must have queued (queue_wait_ms > 100ms)
    waits = sorted([r1.json()["queue_wait_ms"], r2.json()["queue_wait_ms"]])
    assert waits[0] < 100   # first served immediately
    assert waits[1] >= 100  # second waited
    assert procs[0].state.chat_calls == 2


async def test_load_balanced_across_workers(batch_server_client):
    # two workers, each slow → both should be touched in parallel
    def slow(state, _i):
        state.chat_delay = 0.2

    client, procs, _ = await batch_server_client(num_workers=2, configure=slow)
    body = {"messages": [{"role": "user", "content": "hi"}]}

    responses = await asyncio.gather(
        *(client.post("/v1/chat", json=body) for _ in range(2))
    )
    for r in responses:
        assert r.status_code == 200

    assert procs[0].state.chat_calls + procs[1].state.chat_calls == 2
    # Each worker should serve exactly one request (idle round-robin via dispatcher)
    assert procs[0].state.chat_calls == 1
    assert procs[1].state.chat_calls == 1


async def test_queue_full_returns_503(batch_server_client):
    def slow(state, _i):
        state.chat_delay = 0.5

    client, procs, _ = await batch_server_client(
        num_workers=1, configure=slow, max_queue_size=1
    )

    body = {"messages": [{"role": "user", "content": "hi"}]}
    # Fire 3 in parallel: one runs, one queues (slot 1/1), one should hit 503.
    responses = await asyncio.gather(
        client.post("/v1/chat", json=body),
        client.post("/v1/chat", json=body),
        client.post("/v1/chat", json=body),
        return_exceptions=True,
    )
    statuses = sorted(r.status_code if hasattr(r, "status_code") else 0 for r in responses)
    assert 503 in statuses, f"expected at least one 503, got {statuses}"


async def test_worker_500_propagates_as_http_error(batch_server_client):
    def break_it(state, _i):
        state.raise_on_chat = True

    client, _, _ = await batch_server_client(num_workers=1, configure=break_it)
    r = await client.post("/v1/chat", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code in (500, 502)
