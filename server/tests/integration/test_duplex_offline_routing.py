"""Integration tests for /v1/duplex_offline."""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.integration


async def test_duplex_offline_basic(batch_server_client):
    client, procs, _ = await batch_server_client(num_workers=1)
    body = {
        "system_prompt": "You are a friendly assistant.",
        "user_audio_path": "/tmp/fake.wav",
        "config": {"force_listen_count": 2, "chunk_ms": 1000},
        "request_id": "case_001",
    }
    r = await client.post("/v1/duplex_offline", json=body)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["success"] is True
    assert p["request_id"] == "case_001"
    assert p["worker_id"] == "worker_0"
    assert p["total_chunks"] >= 1
    assert p["speak_chunks"] + p["listen_chunks"] == p["total_chunks"]
    assert p["full_text"]  # mock generates at least one chunk of text


async def test_duplex_offline_merged_audio_present(batch_server_client):
    client, _, _ = await batch_server_client(num_workers=1)
    body = {
        "user_audio_path": "/tmp/fake.wav",
        "return_merged_audio": True,
    }
    r = await client.post("/v1/duplex_offline", json=body)
    assert r.status_code == 200
    p = r.json()
    if p["speak_chunks"] > 0:
        assert p["merged_audio_data"] is not None
        assert p["merged_audio_sample_rate"] == 24000


async def test_duplex_offline_per_chunk_audio_toggle(batch_server_client):
    client, _, _ = await batch_server_client(num_workers=1)
    body = {
        "user_audio_path": "/tmp/fake.wav",
        "return_per_chunk_audio": False,
        "return_merged_audio": True,
        "include_text_timeline": True,
    }
    r = await client.post("/v1/duplex_offline", json=body)
    assert r.status_code == 200
    p = r.json()
    # When per-chunk audio is suppressed, audio_data in chunks should be None
    for c in p["chunks"]:
        assert c["audio_data"] is None


async def test_duplex_offline_concurrency_queues_correctly(batch_server_client):
    def slow(state, _i):
        state.duplex_offline_delay = 0.3

    client, procs, _ = await batch_server_client(num_workers=1, configure=slow)
    body = {"user_audio_path": "/tmp/fake.wav"}

    rs = await asyncio.gather(
        client.post("/v1/duplex_offline", json=body),
        client.post("/v1/duplex_offline", json=body),
    )
    for r in rs:
        assert r.status_code == 200
    waits = sorted(r.json()["queue_wait_ms"] for r in rs)
    assert waits[1] >= 100  # second one queued
    assert procs[0].state.duplex_calls == 2


async def test_duplex_offline_routes_to_idle_worker_across_pool(batch_server_client):
    def slow(state, _i):
        state.duplex_offline_delay = 0.2

    client, procs, _ = await batch_server_client(num_workers=2, configure=slow)
    body = {"user_audio_path": "/tmp/fake.wav"}

    rs = await asyncio.gather(
        *(client.post("/v1/duplex_offline", json=body) for _ in range(2))
    )
    for r in rs:
        assert r.status_code == 200

    assert procs[0].state.duplex_calls + procs[1].state.duplex_calls == 2
    assert procs[0].state.duplex_calls == 1
    assert procs[1].state.duplex_calls == 1


async def test_duplex_offline_payload_propagated(batch_server_client):
    client, procs, _ = await batch_server_client(num_workers=1)
    body = {
        "system_prompt": "be brief",
        "user_audio_path": "/data/a.wav",
        "config": {"force_listen_count": 5, "chunk_ms": 500},
        "stop_on_end_of_turn": True,
        "max_chunks": 8,
        "request_id": "dup_42",
    }
    r = await client.post("/v1/duplex_offline", json=body)
    assert r.status_code == 200
    seen = procs[0].state.last_duplex_payload
    assert seen is not None
    assert seen["system_prompt"] == "be brief"
    assert seen["config"]["force_listen_count"] == 5
    assert seen["stop_on_end_of_turn"] is True
    assert seen["max_chunks"] == 8
    assert seen["request_id"] == "dup_42"
