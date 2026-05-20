"""End-to-end smoke / demo script.

Hits the live batch_server with a sequence of progressively harder requests
and prints a clean human-readable report. Generated audio is dropped under
``--output-dir`` (default: ``./smoke_out``).

What it does:
    1) GET  /v1/health  — sanity check
    2) GET  /v1/workers
    3) POST /v1/chat (text-only, no TTS)        → expect non-empty text
    4) POST /v1/chat (text-only, with TTS)      → expect text + 24kHz wav
    5) POST /v1/duplex_offline (audio→audio)    → expect timeline + merged wav
    6) Two concurrent /v1/duplex_offline calls  → expect FIFO queueing

Run with:

    .venv/bin/python scripts/smoke_demo.py \
        --endpoint http://localhost:8080 \
        --user-audio assets/ref_audio/ref_minicpm_signature.wav \
        --output-dir smoke_out
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
import numpy as np
import soundfile as sf

# ANSI for friendlier output
G, Y, R, B, D = "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"


def banner(title: str) -> None:
    print(f"\n{B}{'─' * 70}\n{title}\n{'─' * 70}{D}", flush=True)


def ok(msg: str) -> None:
    print(f"  {G}PASS{D}  {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  {R}FAIL{D}  {msg}", flush=True)


def info(label: str, value: Any) -> None:
    print(f"        {label:<28} {value}", flush=True)


def warn(msg: str) -> None:
    print(f"  {Y}WARN{D}  {msg}", flush=True)


# ============================================================
# Audio helpers
# ============================================================


def _save_audio(b64: Optional[str], path: Path, sr: int = 24000) -> Optional[Path]:
    if not b64:
        return None
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sr)
    return path


def _audio_summary(b64: Optional[str], sr: int = 24000) -> str:
    if not b64:
        return "<none>"
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.float32)
    return f"{len(arr)} samples ({len(arr)/sr:.2f}s @ {sr}Hz)"


# ============================================================
# Smoke steps
# ============================================================


async def step_health(client: httpx.AsyncClient) -> bool:
    banner("[1/6] GET /v1/health")
    r = await client.get("/v1/health")
    if r.status_code != 200:
        fail(f"got {r.status_code}: {r.text}")
        return False
    p = r.json()
    info("status", p["status"])
    info("workers_total", p["workers_total"])
    info("workers_idle", p["workers_idle"])
    info("queue_length", f"{p['queue_length']}/{p['max_queue_size']}")
    ok("health ok")
    return p["ready"] and p["workers_idle"] >= 1


async def step_workers(client: httpx.AsyncClient) -> None:
    banner("[2/6] GET /v1/workers")
    r = await client.get("/v1/workers")
    p = r.json()
    info("total", p["total"])
    for w in p["workers"]:
        info(f"  {w['worker_id']}", f"{w['host']}:{w['port']}  GPU{w['gpu_id']}  {w['status']}")
    ok("workers listed")


async def step_chat_text(client: httpx.AsyncClient) -> None:
    banner("[3/6] POST /v1/chat — text only, no TTS")
    body = {
        "messages": [
            {"role": "user", "content": "用一句话介绍 MiniCPM-o 的多模态能力。"}
        ],
        "generation": {"max_new_tokens": 80, "do_sample": False},
        "tts": {"enabled": False},
    }
    t0 = time.time()
    r = await client.post("/v1/chat", json=body, timeout=120)
    dur = time.time() - t0
    if r.status_code != 200:
        fail(f"got {r.status_code}: {r.text}")
        return
    p = r.json()
    info("worker", p.get("worker_id"))
    info("queue_wait_ms", p.get("queue_wait_ms"))
    info("duration_ms", round(p.get("duration_ms") or 0, 1))
    info("tokens_generated", p.get("tokens_generated"))
    info("client elapsed s", round(dur, 2))
    info("text", repr((p.get("text") or "")[:120]))
    if p.get("success") and p.get("text", "").strip():
        ok("chat text returned")
    else:
        fail("empty text")


async def step_chat_with_tts(client: httpx.AsyncClient, out_dir: Path) -> None:
    banner("[4/6] POST /v1/chat — text + TTS")
    body = {
        "messages": [
            {"role": "user", "content": "请用一句话欢迎一位远方来的朋友。"}
        ],
        "generation": {"max_new_tokens": 60, "do_sample": False},
        "tts": {"enabled": True, "mode": "audio_assistant"},
    }
    t0 = time.time()
    r = await client.post("/v1/chat", json=body, timeout=300)
    dur = time.time() - t0
    if r.status_code != 200:
        fail(f"got {r.status_code}: {r.text}")
        return
    p = r.json()
    info("worker", p.get("worker_id"))
    info("text", repr((p.get("text") or "")[:120]))
    info("audio", _audio_summary(p.get("audio_data"), p.get("audio_sample_rate") or 24000))
    info("client elapsed s", round(dur, 2))

    saved = _save_audio(
        p.get("audio_data"),
        out_dir / "chat_tts.wav",
        sr=p.get("audio_sample_rate") or 24000,
    )
    if saved:
        ok(f"chat+TTS audio → {saved}")
    elif p.get("success"):
        warn("chat ok but TTS audio missing (configuration / vocoder issue?)")
    else:
        fail(f"error: {p.get('error')}")


async def step_duplex_offline(
    client: httpx.AsyncClient, user_audio: Path, out_dir: Path
) -> Optional[Dict[str, Any]]:
    banner("[5/6] POST /v1/duplex_offline — audio in, audio out")
    if not user_audio.exists():
        fail(f"user audio not found: {user_audio}")
        return None

    body = {
        "system_prompt": "请简短回应用户。",
        "user_audio_path": str(user_audio.resolve()),
        "config": {
            "force_listen_count": 3,
            "chunk_ms": 1000,
            "max_new_speak_tokens_per_chunk": 30,
            "temperature": 0.7,
        },
        "stop_on_end_of_turn": False,
        "return_per_chunk_audio": False,
        "return_merged_audio": True,
        "include_text_timeline": True,
        "request_id": "smoke_dup_1",
    }
    t0 = time.time()
    try:
        r = await client.post("/v1/duplex_offline", json=body, timeout=900)
    except httpx.RequestError as e:
        fail(f"network: {e}")
        return None
    dur = time.time() - t0
    if r.status_code != 200:
        fail(f"got {r.status_code}: {r.text[:400]}")
        return None
    p = r.json()
    info("worker", p.get("worker_id"))
    info("queue_wait_ms", p.get("queue_wait_ms"))
    info("total_duration_ms", round(p.get("total_duration_ms") or 0, 1))
    info("client elapsed s", round(dur, 2))
    info("total_chunks", p.get("total_chunks"))
    info("speak / listen", f"{p.get('speak_chunks')} / {p.get('listen_chunks')}")
    info("stopped_reason", p.get("stopped_reason"))
    info("full_text", repr((p.get("full_text") or "")[:120]))
    info("merged_audio", _audio_summary(p.get("merged_audio_data"), 24000))

    saved = _save_audio(p.get("merged_audio_data"), out_dir / "duplex_offline.wav", sr=24000)
    if saved:
        ok(f"duplex audio → {saved}")
    elif p.get("success") and p.get("speak_chunks", 0) == 0:
        warn("model stayed in LISTEN for the whole input (no speak chunks; "
             "try a louder/longer user utterance or lower force_listen_count)")
    elif p.get("success"):
        warn("duplex ok but merged audio missing")
    else:
        fail(f"error: {p.get('error')}")
    return p


async def step_concurrent_duplex(
    client: httpx.AsyncClient, user_audio: Path, out_dir: Path
) -> None:
    banner("[6/6] Concurrent duplex_offline — verify FIFO queue")
    if not user_audio.exists():
        fail(f"user audio not found: {user_audio}")
        return

    def make_body(rid: str) -> Dict[str, Any]:
        return {
            "system_prompt": "请用一句话回应。",
            "user_audio_path": str(user_audio.resolve()),
            "config": {
                "force_listen_count": 3,
                "chunk_ms": 1000,
                "max_new_speak_tokens_per_chunk": 25,
            },
            "stop_on_end_of_turn": False,
            "return_per_chunk_audio": False,
            "return_merged_audio": False,
            "include_text_timeline": False,
            "request_id": rid,
        }

    t0 = time.time()
    # Sample queue state midway in another task
    async def queue_sampler():
        await asyncio.sleep(2.0)
        for _ in range(2):
            q = (await client.get("/v1/queue")).json()
            info("queue snapshot", f"len={q['queue_length']}  running={len(q['running_tasks'])}")
            await asyncio.sleep(2.0)

    sampler = asyncio.create_task(queue_sampler())
    rs = await asyncio.gather(
        client.post("/v1/duplex_offline", json=make_body("c_a"), timeout=900),
        client.post("/v1/duplex_offline", json=make_body("c_b"), timeout=900),
    )
    sampler.cancel()
    dur = time.time() - t0

    if any(r.status_code != 200 for r in rs):
        fail(f"statuses: {[r.status_code for r in rs]}")
        return

    a, b = rs[0].json(), rs[1].json()
    info("total elapsed s", round(dur, 2))
    info("A waits ms",  round(a.get("queue_wait_ms", 0), 1))
    info("A workdone s", round((a.get('total_duration_ms') or 0) / 1000, 2))
    info("B waits ms",  round(b.get("queue_wait_ms", 0), 1))
    info("B workdone s", round((b.get('total_duration_ms') or 0) / 1000, 2))
    a_w, b_w = a.get("queue_wait_ms", 0), b.get("queue_wait_ms", 0)
    if max(a_w, b_w) > min(a_w, b_w) + 1000:
        ok("FIFO queueing observed (one waited >>1s while the other ran)")
    else:
        warn("queue_wait deltas look small; FIFO behaviour weak")


# ============================================================
# Main
# ============================================================


async def main(args):
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"endpoint: {args.endpoint}")
    print(f"output:   {out_dir}")
    print(f"user wav: {args.user_audio}")
    async with httpx.AsyncClient(base_url=args.endpoint, timeout=30.0) as client:
        ready = await step_health(client)
        if not ready:
            fail("worker not idle; aborting")
            sys.exit(1)
        await step_workers(client)
        await step_chat_text(client)
        await step_chat_with_tts(client, out_dir)
        await step_duplex_offline(client, Path(args.user_audio), out_dir)
        if not args.skip_concurrent:
            await step_concurrent_duplex(client, Path(args.user_audio), out_dir)

    banner("Done.")
    print(f"Generated artefacts saved under: {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="batch_server smoke / demo")
    p.add_argument("--endpoint", default="http://localhost:8080")
    p.add_argument(
        "--user-audio",
        default="assets/ref_audio/ref_minicpm_signature.wav",
        help="path to a 16kHz mono wav to feed /v1/duplex_offline",
    )
    p.add_argument("--output-dir", default="smoke_out")
    p.add_argument("--skip-concurrent", action="store_true",
                   help="skip the final 2-way concurrent step (faster)")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
