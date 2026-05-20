"""Pytest configuration & shared fixtures.

Why so much plumbing?
=====================

The server is split across two long-lived processes (``batch_server`` + N
``worker``). For unit/integration tests we want:

1. *No* GPU or model loading. → use ``mock_worker.MockWorkerProcess``.
2. *No* subprocess management headaches. → run each mock worker on a random
   TCP port inside its own background thread, and run ``batch_server`` inside
   the pytest event loop via ``httpx.ASGITransport``.
3. Strict isolation between tests. → each fixture-scope creates a fresh pool.

Fixtures provided
=================

- ``unused_tcp_port``: get one free localhost port
- ``unused_tcp_ports``: factory: ``unused_tcp_ports(n)`` returns n free ports
- ``mock_worker_factory``: factory producing N mock workers with knobs
- ``batch_server_client``: ``httpx.AsyncClient`` talking directly to the
  batch_server FastAPI app via ASGITransport, with a pre-wired worker pool
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import pytest

# Make ``server/`` importable as the top-level package (so ``from config
# import ...`` and friends work the same way as in production).
SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


# ============================================================
# Networking helpers
# ============================================================


def _pick_free_port() -> int:
    """Ask the OS for a free TCP port on localhost (race-free enough for tests)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def unused_tcp_port() -> int:
    return _pick_free_port()


@pytest.fixture
def unused_tcp_ports() -> Callable[[int], List[int]]:
    def _factory(n: int) -> List[int]:
        # Reserve them all up-front to reduce collision probability.
        return [_pick_free_port() for _ in range(n)]
    return _factory


# ============================================================
# Mock worker pool
# ============================================================


@pytest.fixture
def mock_worker_factory(unused_tcp_ports):
    """Spin up N mock workers, return ``(processes, addresses)``.

    Caller controls per-worker behaviour via the optional ``configure``
    callback ``configure(state, index) -> None``.

    Usage::

        procs, addrs = mock_worker_factory(2)
        # or with knobs:
        procs, addrs = mock_worker_factory(
            3, configure=lambda st, i: setattr(st, "chat_delay", 0.5)
        )
    """
    from tests.mock_worker import MockWorkerProcess, MockWorkerState

    started: List[MockWorkerProcess] = []

    def _factory(
        n: int,
        configure: Optional[Callable[[object, int], None]] = None,
    ):
        ports = unused_tcp_ports(n)
        procs: List[MockWorkerProcess] = []
        for i, p in enumerate(ports):
            state = MockWorkerState(gpu_id=i)
            if configure is not None:
                configure(state, i)
            proc = MockWorkerProcess(host="127.0.0.1", port=p, state=state)
            proc.start()
            procs.append(proc)
            started.append(proc)
        addrs = [p.address for p in procs]
        return procs, addrs

    yield _factory

    for p in started:
        try:
            p.stop()
        except Exception:
            pass


# ============================================================
# batch_server in-process client
# ============================================================


@pytest.fixture
async def batch_server_client(mock_worker_factory):
    """Return ``(client, mock_procs, batch_server_module)`` for integration tests.

    The factory function signature is ``await build(num_workers, configure=None)``::

        client, procs, mod = await build(2)
        resp = await client.post("/v1/chat", json={...})

    This fixture sets up:

    - N mock workers on random ports
    - A fresh ``batch_server.WorkerPool`` wired to those workers
    - An ``httpx.AsyncClient`` that talks to the batch_server FastAPI app via
      ASGITransport (no real socket binding for batch_server itself).
    """
    import httpx

    import batch_server
    from gateway_modules.models import EtaConfig

    clients: List[httpx.AsyncClient] = []
    pools = []

    async def _build(num_workers: int = 1, configure=None, eta_config: Optional[EtaConfig] = None,
                     max_queue_size: int = 100, request_timeout: float = 10.0):
        procs, addrs = mock_worker_factory(num_workers, configure=configure)

        # Build a private WorkerPool and inject into batch_server module global.
        # NOTE: batch_server uses a module-level global ``worker_pool``; replacing
        # it bypasses the lifespan startup so we own its lifecycle from tests.
        from gateway_modules.worker_pool import WorkerPool

        pool = WorkerPool(
            worker_addresses=addrs,
            max_queue_size=max_queue_size,
            request_timeout=request_timeout,
            eta_config=eta_config or EtaConfig(),
        )
        await pool.start()
        batch_server.worker_pool = pool
        pools.append(pool)

        transport = httpx.ASGITransport(app=batch_server.app)
        client = httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=request_timeout + 5.0,
        )
        clients.append(client)
        return client, procs, batch_server

    yield _build

    for c in clients:
        await c.aclose()
    for pool in pools:
        try:
            await pool.stop()
        except Exception:
            pass

    # Reset the global so other tests aren't tainted.
    import batch_server as _bs
    _bs.worker_pool = None


# ============================================================
# Marker auto-skip
# ============================================================


def pytest_collection_modifyitems(config, items):
    """Skip e2e tests by default unless ``--run-e2e`` is given."""
    if config.getoption("--run-e2e", default=False):
        return
    skip_e2e = pytest.mark.skip(reason="needs --run-e2e (requires GPU + model)")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests that require a real GPU + model.",
    )
