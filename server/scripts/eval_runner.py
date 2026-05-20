"""批量评测客户端示例

把一份 JSONL 评测集喂给 `batch_server`，并发送 /v1/chat 或 /v1/duplex_offline
请求；结果落盘到 ``--output`` 指定的 JSONL（每行一条）。

JSONL 行格式
============

单工 chat：
::

    {"id": "case_001",
     "task": "chat",
     "messages": [{"role": "user", "content": "..."}],
     "tts_enabled": true}

双工非流式：
::

    {"id": "dlg_001",
     "task": "duplex",
     "system_prompt": "...",
     "user_audio_path": "/data/eval/001.wav",
     "image_paths": ["/data/eval/001_frame_0.jpg", ...],
     "config": {"force_listen_count": 3, "chunk_ms": 1000},
     "stop_on_end_of_turn": false}

并发：通过 ``--concurrency`` 控制 client 侧的并发数；server 侧的并发上限由
worker 数量决定，超出的请求会进 server 的 FIFO 队列。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


async def _post(client: httpx.AsyncClient, url: str, body: Dict[str, Any],
                timeout: float) -> Dict[str, Any]:
    resp = await client.post(url, json=body, timeout=timeout)
    if resp.status_code != 200:
        return {"success": False, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}
    return resp.json()


def _build_chat_body(case: Dict[str, Any]) -> Dict[str, Any]:
    """把 JSONL 行转成 ChatRequest"""
    return {
        "messages": case.get("messages", []),
        "tts": {"enabled": bool(case.get("tts_enabled", False))},
        "generation": case.get("generation", {}),
    }


def _build_duplex_body(case: Dict[str, Any]) -> Dict[str, Any]:
    """把 JSONL 行转成 DuplexBatchRequest"""
    return {
        "system_prompt": case.get("system_prompt", "You are a helpful assistant."),
        "user_audio_path": case.get("user_audio_path"),
        "user_audio_base64": case.get("user_audio_base64"),
        "image_paths": case.get("image_paths"),
        "image_base64_list": case.get("image_base64_list"),
        "ref_audio_path": case.get("ref_audio_path"),
        "config": case.get("config") or {},
        "stop_on_end_of_turn": bool(case.get("stop_on_end_of_turn", False)),
        "max_chunks": case.get("max_chunks"),
        "return_per_chunk_audio": bool(case.get("return_per_chunk_audio", False)),
        "return_merged_audio": bool(case.get("return_merged_audio", True)),
        "include_text_timeline": bool(case.get("include_text_timeline", True)),
        "request_id": case.get("id"),
    }


async def _process_one(
    case: Dict[str, Any],
    *,
    endpoint: str,
    client: httpx.AsyncClient,
    timeout: float,
    audio_out_dir: Optional[Path],
) -> Dict[str, Any]:
    task = case.get("task") or "chat"
    case_id = case.get("id") or "<unset>"
    t0 = time.time()

    if task == "chat":
        body = _build_chat_body(case)
        result = await _post(client, f"{endpoint}/v1/chat", body, timeout)
    elif task == "duplex":
        body = _build_duplex_body(case)
        result = await _post(client, f"{endpoint}/v1/duplex_offline", body, timeout)
    else:
        return {"id": case_id, "success": False, "error": f"unknown task: {task}"}

    elapsed_s = time.time() - t0

    # 如果指定了 audio_out_dir，把 audio_data / merged_audio_data 落盘
    if audio_out_dir is not None and result.get("success", True):
        try:
            audio_b64 = result.get("merged_audio_data") or result.get("audio_data")
            if audio_b64:
                import numpy as np
                import soundfile as sf
                audio_out_dir.mkdir(parents=True, exist_ok=True)
                arr = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
                sr = (
                    result.get("merged_audio_sample_rate")
                    or result.get("audio_sample_rate")
                    or 24000
                )
                out_path = audio_out_dir / f"{case_id}.wav"
                sf.write(str(out_path), arr, sr)
                result["audio_file"] = str(out_path)
                # 不把 base64 写到 JSONL 里，太大
                result["merged_audio_data"] = None
                result["audio_data"] = None
                # 同时压缩 per-chunk audio 字段以减少日志体积
                if isinstance(result.get("chunks"), list):
                    for c in result["chunks"]:
                        if isinstance(c, dict) and c.get("audio_data"):
                            c["audio_data"] = None
        except Exception as e:
            result["audio_save_error"] = str(e)

    summary = {
        "id": case_id,
        "task": task,
        "elapsed_s": round(elapsed_s, 2),
        "success": result.get("success", True),
        "response": result,
    }
    return summary


async def _main(args):
    cases = []
    with open(args.input_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    print(f"Loaded {len(cases)} cases from {args.input_list}", flush=True)

    audio_out_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else None
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    out_lock = asyncio.Lock()

    async def _bounded_one(case, fh):
        async with sem:
            async with httpx.AsyncClient() as client:
                result = await _process_one(
                    case,
                    endpoint=args.endpoint,
                    client=client,
                    timeout=args.timeout,
                    audio_out_dir=audio_out_dir,
                )
        async with out_lock:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            fh.flush()
            tag = "OK" if result["success"] else "FAIL"
            print(f"  [{tag}] {result['id']:<24} {result['elapsed_s']}s", flush=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        tasks = [_bounded_one(c, fh) for c in cases]
        await asyncio.gather(*tasks)

    print(f"\nDone. Results: {output_path}", flush=True)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch eval client for batch_server.py")
    p.add_argument("--endpoint", default="http://localhost:8080",
                   help="batch_server base URL (default: http://localhost:8080)")
    p.add_argument("--input-list", required=True,
                   help="JSONL file, one case per line")
    p.add_argument("--output", required=True,
                   help="Output JSONL path")
    p.add_argument("--audio-dir", default=None,
                   help="If set, write generated audio (merged) to this dir as .wav")
    p.add_argument("--concurrency", type=int, default=4,
                   help="Client-side concurrency (default 4)")
    p.add_argument("--timeout", type=float, default=900.0,
                   help="Per-request timeout (seconds, default 900)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    asyncio.run(_main(args))
