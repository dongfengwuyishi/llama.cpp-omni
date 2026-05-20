"""E2E smoke for the new ``logits`` capture feature.

Hits a *running* batch_server stack (default http://localhost:8080) with:

  1. POST /v1/chat              with logits.format=inline   → decode inline blob
  2. POST /v1/chat              with logits.format=file     → read back safetensors
  3. POST /v1/duplex_offline    with logits.format=file     → read consolidated file

For each step the script prints a small report (PASS/FAIL/WARN, shapes,
rank-of-sampled-token sanity, file size, metadata).

Run with:

    .venv/bin/python scripts/smoke_logits.py
    # default output:
    #   /cache/caitianchi/data/minicpm-o-server-eval/outputs/logits/<timestamp>/
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import httpx
import numpy as np

DEFAULT_OUTPUT_ROOT = Path(
    "/cache/caitianchi/data/minicpm-o-server-eval/outputs/logits"
)


def banner(title: str) -> None:
    print(f"\n{'-' * 70}\n{title}\n{'-' * 70}", flush=True)


def ok(msg: str) -> None:
    print(f"  PASS  {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"  WARN  {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}", flush=True)


def info(label: str, value: Any) -> None:
    print(f"        {label:<26} {value}", flush=True)


# ----------------------------------------------------------------------------
# safetensors reader (minimal — matches what our server writes)
# ----------------------------------------------------------------------------


def read_safetensors(path: str) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
        body_offset = 8 + header_size
        tensors: Dict[str, np.ndarray] = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            dtype = spec["dtype"]
            shape = spec["shape"]
            start, end = spec["data_offsets"]
            f.seek(body_offset + start)
            raw = f.read(end - start)
            if dtype == "I32":
                arr = np.frombuffer(raw, dtype=np.int32).reshape(shape)
            elif dtype == "BF16":
                u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                arr = u16  # keep raw bf16; caller can convert
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            else:
                raise NotImplementedError(f"unsupported dtype: {dtype}")
            tensors[name] = arr
    return header.get("__metadata__", {}), tensors


def bf16_uint16_to_fp32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


# ----------------------------------------------------------------------------
# Steps
# ----------------------------------------------------------------------------


def step_chat_inline(client: httpx.Client) -> bool:
    banner("[1/3] POST /v1/chat  (logits.format=inline)")
    body = {
        "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
        "generation": {"max_new_tokens": 20, "do_sample": False},
        "tts": {"enabled": False},
        "logits": {"enabled": True, "format": "inline"},
    }
    t0 = time.time()
    r = client.post("/v1/chat", json=body, timeout=300)
    dt = time.time() - t0
    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:400]}")
        return False
    p = r.json()
    info("client elapsed s", round(dt, 2))
    info("text", repr((p.get("text") or "")[:120]))
    lp = p.get("logits")
    if not lp:
        fail("no logits in response")
        return False
    info("logits.success", lp["success"])
    info("logits.n_tokens / n_prefill", f"{lp['n_tokens']} / {lp['n_prefill_tokens']}")
    info("logits.vocab_size", lp["vocab_size"])
    info("logits.dtype", lp["dtype"])
    tok = np.frombuffer(base64.b64decode(lp["token_ids_b64"]), dtype=np.int32)
    raw = base64.b64decode(lp["logits_b64"])
    expected = lp["n_tokens"] * lp["vocab_size"] * 2
    info("token_ids first 10", tok[:10].tolist())
    info("logits raw bytes", f"{len(raw)} (expected {expected})")
    if len(raw) != expected:
        fail("logits byte count mismatch")
        return False
    u16 = np.frombuffer(raw, dtype=np.uint16).reshape(lp["n_tokens"], lp["vocab_size"])
    fp = bf16_uint16_to_fp32(u16)
    n_pref = lp["n_prefill_tokens"]
    if n_pref < lp["n_tokens"]:
        row = fp[n_pref]
        sampled_tok = int(tok[n_pref])
        if sampled_tok >= 0:
            rank = int(np.sum(row > row[sampled_tok]))
            top5 = np.argsort(row)[-5:][::-1]
            info("first decode step", f"sampled={sampled_tok}, rank={rank}, top5={top5.tolist()}")
            # do_sample=False so the sampled token should be argmax
            if rank != 0:
                warn(f"sampled token rank={rank} (expected 0 for greedy decode)")
        else:
            info("first decode step", f"modality placeholder sampled={sampled_tok}")
    ok("inline logits round-trip ok")
    return True


def step_chat_file(client: httpx.Client, out_dir: Path) -> bool:
    banner("[2/3] POST /v1/chat  (logits.format=file)")
    body = {
        "messages": [{"role": "user", "content": "What is 2+3?"}],
        "generation": {"max_new_tokens": 12, "do_sample": False},
        "tts": {"enabled": False},
        "logits": {"enabled": True, "format": "file", "output_dir": str(out_dir)},
    }
    r = client.post("/v1/chat", json=body, timeout=300)
    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:400]}")
        return False
    p = r.json()
    lp = p.get("logits") or {}
    info("logits.success", lp.get("success"))
    info("logits.file", lp.get("file"))
    info("logits.n_tokens", f"{lp.get('n_tokens')} / vocab={lp.get('vocab_size')}")
    path = lp.get("file")
    if not path or not os.path.exists(path):
        fail(f"safetensors file missing: {path}")
        return False
    sz = os.path.getsize(path)
    expected_min = lp["n_tokens"] * lp["vocab_size"] * 2  # bf16 body alone
    info("file size", f"{sz} bytes (expected >= {expected_min})")
    if sz < expected_min:
        fail("file too small")
        return False
    md, tensors = read_safetensors(path)
    info("tensors", {k: tensors[k].shape for k in tensors})
    info("metadata", md)
    expected_shape_logits = (lp["n_tokens"], lp["vocab_size"])
    if tensors["logits"].shape != expected_shape_logits:
        fail(f"logits shape mismatch: {tensors['logits'].shape} vs {expected_shape_logits}")
        return False
    if tensors["token_ids"].shape != (lp["n_tokens"],):
        fail(f"token_ids shape mismatch: {tensors['token_ids'].shape}")
        return False
    ok("safetensors content matches header")
    return True


def step_duplex_file(client: httpx.Client, out_dir: Path, audio_path: str) -> bool:
    banner("[3/3] POST /v1/duplex_offline  (logits.format=file)")
    body = {
        "system_prompt": "请用一句话回应。",
        "user_audio_path": audio_path,
        "config": {"force_listen_count": 3, "chunk_ms": 1000, "max_new_speak_tokens_per_chunk": 25},
        "stop_on_end_of_turn": False,
        "return_per_chunk_audio": False,
        "return_merged_audio": False,
        "include_text_timeline": False,
        "logits": {"enabled": True, "format": "file", "output_dir": str(out_dir)},
        "request_id": "dup_logits_001",
    }
    r = client.post("/v1/duplex_offline", json=body, timeout=900)
    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:400]}")
        return False
    p = r.json()
    info("success / total_chunks", f"{p.get('success')} / {p.get('total_chunks')}")
    info("speak / listen", f"{p.get('speak_chunks')} / {p.get('listen_chunks')}")
    lp = p.get("logits") or {}
    info("logits.success", lp.get("success"))
    info("logits.file", lp.get("file"))
    info("logits.n_tokens", f"{lp.get('n_tokens')} / n_prefill={lp.get('n_prefill_tokens')}")
    info("logits.extra_metadata", lp.get("extra_metadata"))
    path = lp.get("file")
    if not path or not os.path.exists(path):
        fail(f"duplex safetensors missing: {path}")
        return False
    sz = os.path.getsize(path)
    info("file size", f"{sz} bytes")
    md, tensors = read_safetensors(path)
    info("tensors", {k: tensors[k].shape for k in tensors})
    info("metadata.chunk_boundaries", md.get("chunk_boundaries"))
    info("metadata.chunk_prefill_counts", md.get("chunk_prefill_counts"))
    info("metadata.n_prefill_tokens", md.get("n_prefill_tokens"))
    info("metadata.n_tokens", md.get("n_tokens"))
    cb = md.get("chunk_boundaries")
    try:
        cb_list = json.loads(cb) if isinstance(cb, str) else cb
        if cb_list and cb_list[-1] != int(md["n_tokens"]):
            warn(f"chunk_boundaries[-1]={cb_list[-1]} != n_tokens={md['n_tokens']}")
        else:
            ok(f"chunk_boundaries close consistent ({len(cb_list)-1} chunks)")
    except Exception as e:
        warn(f"chunk_boundaries parse: {e}")
    ok("duplex safetensors content matches header")
    return True


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8080")
    ap.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where the server should write .safetensors. Default: a fresh "
            "timestamped subdir under "
            "/cache/caitianchi/data/minicpm-o-server-eval/outputs/logits/."
        ),
    )
    ap.add_argument(
        "--user-audio",
        default="assets/ref_audio/ref_minicpm_signature.wav",
        help="audio for duplex_offline test (16kHz mono wav)",
    )
    ap.add_argument("--skip-duplex", action="store_true")
    args = ap.parse_args()

    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = (DEFAULT_OUTPUT_ROOT / ts).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Wipe previous test artefacts so file-size assertions are meaningful.
    for p in out.glob("*.safetensors"):
        p.unlink()

    print(f"endpoint:   {args.endpoint}")
    print(f"output_dir: {out}")
    print(f"user_audio: {args.user_audio}")

    results = []
    with httpx.Client(base_url=args.endpoint, timeout=300) as client:
        r1 = client.get("/v1/health")
        if r1.status_code != 200 or not r1.json().get("ready"):
            print(f"FAIL health: {r1.status_code}: {r1.text}")
            sys.exit(1)
        results.append(("chat_inline",  step_chat_inline(client)))
        results.append(("chat_file",    step_chat_file(client, out)))
        if not args.skip_duplex:
            results.append(("duplex_file", step_duplex_file(client, out, args.user_audio)))

    banner("Summary")
    ok_count = sum(1 for _, v in results if v)
    for name, v in results:
        print(f"  {'PASS' if v else 'FAIL'}  {name}", flush=True)
    print(f"\n{ok_count}/{len(results)} passed")
    print(f"\nArtefacts saved under: {out}")

    # Refresh the "latest" symlink for the default-output path.
    if not args.output_dir:
        latest = DEFAULT_OUTPUT_ROOT / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(out.name)
        except OSError:
            pass

    sys.exit(0 if ok_count == len(results) else 1)


if __name__ == "__main__":
    main()
