"""Integration tests for /v1/health, /v1/workers, /v1/queue, /v1/status."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


async def test_health_basic(batch_server_client):
    client, procs, _ = await batch_server_client(num_workers=2)
    r = await client.get("/v1/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["workers_total"] == 2
    assert payload["workers_idle"] >= 1


async def test_health_compat_path(batch_server_client):
    client, _, _ = await batch_server_client(num_workers=1)
    r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["workers_total"] == 1


async def test_workers_endpoint(batch_server_client):
    client, procs, _ = await batch_server_client(num_workers=3)
    r = await client.get("/v1/workers")
    assert r.status_code == 200
    payload = r.json()
    assert payload["total"] == 3
    assert len(payload["workers"]) == 3
    expected_addrs = {p.address for p in procs}
    got_addrs = {f"{w['host']}:{w['port']}" for w in payload["workers"]}
    assert got_addrs == expected_addrs


async def test_queue_empty(batch_server_client):
    client, _, _ = await batch_server_client(num_workers=1)
    r = await client.get("/v1/queue")
    assert r.status_code == 200
    p = r.json()
    assert p["queue_length"] == 0
    assert p["max_queue_size"] >= 1
    assert p["items"] == []


async def test_status_aggregate(batch_server_client):
    client, _, _ = await batch_server_client(num_workers=2)
    r = await client.get("/v1/status")
    assert r.status_code == 200
    p = r.json()
    assert p["total_workers"] == 2
    assert p["idle_workers"] + p["busy_workers"] == 2
    assert "running_tasks" in p
