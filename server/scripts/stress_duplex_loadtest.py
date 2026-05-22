#!/usr/bin/env python3
"""离线压测 (双工)：MiniCPM-o batch_server /v1/duplex_offline。

和 ``stress_loadtest.py`` 是姊妹脚本：那个测的是 ``/v1/chat`` 单工路径，这个
专门压 ``/v1/duplex_offline``。两条路径在 server 端走完全不同的状态机：

* chat：``stream_prefill``（一次性塞所有 prompt）→ ``stream_decode``
  （阻塞拿完整 SSE）→ 立即归还 worker。
* duplex_offline：把 user audio 切成 ``chunk_ms`` 大小的小块，逐块
  prefill + decode + 可选 TTS + 可选 T2W；server 内部维护 listen/speak
  状态机；每次请求独占 worker 数十秒。

多 worker 共享同一个日期 bucket dir 时，**duplex 的 .safetensors 文件名
也必须不撞**（fix: ``make_logits_filename`` 现在 chat / duplex 共用同一套
``{kind}_w{idx}_p{pid_hex7}_{seq:08d}[_{rid}]`` 模板）。这次压测的核心断言：

  N 次 duplex_offline + format=file → N 个 .safetensors 文件 → basename
  100% unique 且 100% 匹配新 regex。

为什么并发数低？每个 duplex 请求约 30~60s 独占 worker，c=4 worker 只能跑
~4 个并发；过高的 c 都是排队，不增加压力维度。所以默认 ``--concurrency 4
--total-per-worker 3`` ⇒ 共 12 个请求，跑约 3~5 分钟，足够覆盖：

  - 跨 worker 同时落盘文件不撞名
  - FIFO 队列在 c > worker 数时正常分配
  - GPU mem 在长尾请求下不持续涨
  - 状态机长时间运行无 traceback / oom / cudaMalloc 失败
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ============================================================
# 单次 duplex_offline 调用
# ============================================================


@dataclass
class CallResult:
    ok: bool
    status: int
    latency_ms: float
    err: Optional[str] = None
    worker_id: Optional[str] = None
    queue_wait_ms: Optional[float] = None
    full_text: str = ""
    total_chunks: int = 0
    speak_chunks: int = 0
    audio_duration_s: float = 0.0
    logits_file: Optional[str] = None     # if logits.format='file' succeeded
    request_id: Optional[str] = None
    user_audio_path: Optional[str] = None
    system_prompt: str = ""


def make_body(*, user_audio: str, request_id: str, with_logits_file: bool,
              logits_dir: str) -> Dict[str, Any]:
    """构造一个 duplex_offline 请求 body。

    我们把 ``return_per_chunk_audio`` / ``return_merged_audio`` 都关掉
    —— 这次压测**只**关心服务可用性 + logits 文件命名，不关心音频回放质量；
    多余的字段会让 SSE 响应变得很大、徒增 RAM 压力。

    ``include_text_timeline=False`` 同理：每个 chunk 的文本时间线对压测无
    意义，关掉减少 JSON 序列化开销。

    ``max_new_speak_tokens_per_chunk=24`` + ``max_chunks=8`` 是为了让单次
    请求**有上限**，避免某个测试 case 触发 attractor 导致跑几分钟才返回。
    real prod 默认更大，但这里我们只是做服务侧故障注入。
    """
    # ⚠️ 关键参数说明（曾经踩过两个互相打架的坑）：
    #
    #   1. ``max_chunks`` 是 ``DuplexBatchRequest`` 的**顶层字段**，不是
    #      ``config`` 内的字段。把它放 config 里等于白填，server 会用
    #      None=不限来跑；当 audio 跑完后 LLM 偶尔会陷入 attractor，KV
    #      cache 一路涨到 32K，单条请求十几分钟不返回。
    #
    #   2. ``stop_on_end_of_turn=True`` 反而**有害**：duplex 模型在
    #      chunk 1 几乎一定会 emit ``end_of_turn=True``（因为还在 listen
    #      phase 时 EOT bit 总是 True），server 就立刻 short-circuit 退出
    #      ``stopped_reason=end_of_turn``，total_chunks=1, speak_chunks=0,
    #      full_text=""。所以这里固定用 False，让 audio_exhausted 或
    #      max_chunks 兜底退出。
    body: Dict[str, Any] = {
        "system_prompt": "请简短回应用户。",
        "user_audio_path": user_audio,
        "config": {
            "force_listen_count": 3,
            "chunk_ms": 1000,
            "max_new_speak_tokens_per_chunk": 24,
            "temperature": 0.7,
        },
        "stop_on_end_of_turn": False,
        "max_chunks": 16,
        "return_per_chunk_audio": False,
        "return_merged_audio": False,
        "include_text_timeline": False,
        "request_id": request_id,
    }
    if with_logits_file:
        body["logits"] = {
            "enabled": True,
            "format": "file",
            "output_dir": logits_dir,
        }
    return body


async def call_duplex(
    client: httpx.AsyncClient,
    *,
    user_audio: str,
    request_id: str,
    timeout_s: float,
    logits_dir: str,
) -> CallResult:
    body = make_body(user_audio=user_audio, request_id=request_id,
                     with_logits_file=True, logits_dir=logits_dir)
    sysp = str(body.get("system_prompt") or "")
    t0 = time.perf_counter()
    try:
        r = await client.post("/v1/duplex_offline", json=body,
                              timeout=timeout_s)
    except httpx.TimeoutException as e:
        return CallResult(ok=False, status=-1,
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          err=f"timeout: {e!s}",
                          request_id=request_id, user_audio_path=user_audio,
                          system_prompt=sysp)
    except httpx.RequestError as e:
        return CallResult(ok=False, status=-1,
                          latency_ms=(time.perf_counter() - t0) * 1000,
                          err=f"network: {type(e).__name__}: {e!s}",
                          request_id=request_id, user_audio_path=user_audio,
                          system_prompt=sysp)
    lat = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return CallResult(ok=False, status=r.status_code, latency_ms=lat,
                          err=r.text[:300],
                          request_id=request_id, user_audio_path=user_audio,
                          system_prompt=sysp)
    try:
        p = r.json()
    except Exception as e:
        return CallResult(ok=False, status=r.status_code, latency_ms=lat,
                          err=f"json-decode: {e!s}",
                          request_id=request_id, user_audio_path=user_audio,
                          system_prompt=sysp)
    success = bool(p.get("success"))
    err = None if success else (p.get("error") or "success=false")
    logits_blob = p.get("logits") or {}
    logits_file = (logits_blob.get("file") if isinstance(logits_blob, dict)
                   else None)
    return CallResult(
        ok=success, status=r.status_code, latency_ms=lat, err=err,
        worker_id=p.get("worker_id"),
        queue_wait_ms=p.get("queue_wait_ms"),
        full_text=p.get("full_text", "") or "",
        total_chunks=int(p.get("total_chunks") or 0),
        speak_chunks=int(p.get("speak_chunks") or 0),
        audio_duration_s=float(p.get("audio_duration_s") or 0.0),
        logits_file=logits_file,
        request_id=p.get("request_id") or request_id,
        user_audio_path=user_audio,
        system_prompt=sysp,
    )


# ============================================================
# 监控（复用 stress_loadtest 那套）
# ============================================================


@dataclass
class Snapshot:
    t: float
    worker_states: Dict[str, int]
    worker_total: int
    queue_size: Optional[int]
    gpu_mem_used: List[int]


def query_gpu_mem(gpu_ids: List[int]) -> List[int]:
    if not gpu_ids:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode("utf-8", errors="ignore")
        m: Dict[int, int] = {}
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit():
                m[int(parts[0])] = int(parts[1])
        return [m.get(g, 0) for g in gpu_ids]
    except Exception:
        return [0] * len(gpu_ids)


async def monitor_loop(
    client: httpx.AsyncClient,
    interval_s: float,
    stop_evt: asyncio.Event,
    snapshots: List[Snapshot],
    gpu_ids: List[int],
) -> None:
    while not stop_evt.is_set():
        t = time.perf_counter()
        states: Dict[str, int] = {}
        total = 0
        try:
            r = await client.get("/v1/workers", timeout=5)
            if r.status_code == 200:
                p = r.json()
                total = p.get("total") or 0
                for w in p.get("workers") or []:
                    s = str(w.get("status", "unknown"))
                    states[s] = states.get(s, 0) + 1
        except Exception:
            states["__monitor_error__"] = states.get("__monitor_error__", 0) + 1
        qsize: Optional[int] = None
        try:
            r = await client.get("/v1/queue", timeout=5)
            if r.status_code == 200:
                p = r.json()
                qsize = p.get("queue_length")
                if qsize is None:
                    qsize = p.get("size") or 0
        except Exception:
            qsize = None
        mem = query_gpu_mem(gpu_ids)
        snapshots.append(Snapshot(t=t, worker_states=states,
                                  worker_total=total, queue_size=qsize,
                                  gpu_mem_used=mem))
        try:
            await asyncio.wait_for(stop_evt.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


# ============================================================
# 报告
# ============================================================


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


# 多 worker 防撞命名 regex（要和 logits_retention.make_logits_filename 对齐）。
# 路径例：``/tmp/minicpm_logits_duplex_stress/2026-05-21/duplex_w2_p0003b81_00000456_<rid>.safetensors``
_FILENAME_RE = re.compile(
    r"^(chat|duplex)_w(\d+)_p([0-9a-f]{7})_(\d{8})(?:_([A-Za-z0-9_]+))?\.safetensors$"
)


async def main_async(args: argparse.Namespace) -> int:
    # ``--user-audio`` 既可以是单个 wav，也可以是 manifest.json（由
    # ``synth_questions_voxcpm.py`` 生成）。manifest 模式下我们会带上
    # 每条 wav 的"已知文本（即合成时的 prompt）"作为 ``ref_question_text``
    # 写进 dump，方便人审"听不懂音频"时也能直接看文本判断模型回答得对不对。
    user_audio_pool: List[Tuple[str, str]] = []  # (wav_path, ref_text)
    user_audio_arg = os.path.abspath(args.user_audio)
    if not os.path.exists(user_audio_arg):
        print(f"[!] user audio not found: {user_audio_arg}", file=sys.stderr)
        return 2
    if user_audio_arg.endswith(".json"):
        with open(user_audio_arg, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, list) or not manifest:
            print(f"[!] manifest empty/malformed: {user_audio_arg}",
                  file=sys.stderr)
            return 2
        for rec in manifest:
            wav = rec.get("wav")
            txt = rec.get("text", "")
            if wav and os.path.exists(wav):
                user_audio_pool.append((wav, txt))
        if not user_audio_pool:
            print(f"[!] no valid wav in manifest", file=sys.stderr)
            return 2
        print(f"[+] loaded {len(user_audio_pool)} wavs from manifest "
              f"{user_audio_arg}")
    elif os.path.isdir(user_audio_arg):
        for f in sorted(os.listdir(user_audio_arg)):
            if f.endswith(".wav"):
                user_audio_pool.append((os.path.join(user_audio_arg, f), ""))
        if not user_audio_pool:
            print(f"[!] no .wav under dir {user_audio_arg}", file=sys.stderr)
            return 2
        print(f"[+] loaded {len(user_audio_pool)} wavs from dir "
              f"{user_audio_arg}")
    else:
        user_audio_pool.append((user_audio_arg, ""))
    user_audio = user_audio_pool[0][0]  # back-compat: 老 print 用

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] \
        if args.gpu_ids else []

    transport_limits = httpx.Limits(
        max_connections=args.concurrency * 2,
        max_keepalive_connections=args.concurrency,
    )

    async with httpx.AsyncClient(
        base_url=args.base_url,
        limits=transport_limits,
    ) as client:
        try:
            r = await client.get("/v1/health", timeout=10)
            if r.status_code != 200:
                print(f"[!] /v1/health -> {r.status_code}: {r.text[:200]}",
                      file=sys.stderr)
                return 2
        except Exception as e:
            print(f"[!] gateway health check failed: {e}", file=sys.stderr)
            return 2
        try:
            r = await client.get("/v1/workers", timeout=10)
            n_workers = (r.json() or {}).get("total", 0)
        except Exception:
            n_workers = 0
        print(f"[+] gateway healthy, workers reported: {n_workers}")
        print(f"[+] user_audio: {user_audio}")
        print(f"[+] logits dir: {args.logits_dir}")
        print(f"[+] plan: c={args.concurrency} × per-task-N={args.total_per_worker}"
              f" = {args.concurrency * args.total_per_worker} requests")

        os.makedirs(args.logits_dir, exist_ok=True)

        # ``--dump-conversations`` 把每条 (system_prompt, user_audio_path,
        # full_text) 写到 JSONL，方便事后人工审听音频 + 对照模型回复文本。
        # 注意 duplex 请求的"输入"是音频不是文本，所以人审时通常需要播
        # ``user_audio_path`` 听一遍才能判断 ``full_text`` 回答得对不对。
        dump_fp = None
        if args.dump_conversations:
            dump_path = os.path.abspath(args.dump_conversations)
            os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
            dump_fp = open(dump_path, "w", encoding="utf-8")
            print(f"[+] dumping conversations to {dump_path}")

        snapshots: List[Snapshot] = []
        stop_evt = asyncio.Event()
        mon_task = asyncio.create_task(
            monitor_loop(client, args.monitor_interval_s,
                         stop_evt, snapshots, gpu_ids))

        results: List[CallResult] = []
        results_lock = asyncio.Lock()
        progress = [0]
        total_target = args.concurrency * args.total_per_worker

        # 把当前请求用的"已知问句文本"传给 _dump（CallResult 里没有这一
        # 项；我们在闭包里另存一份）。这样 JSONL 每行就有 ``user_question``
        # = 我们合成 wav 时的 prompt，人审能直接对照。
        cur_ref: Dict[str, str] = {}

        def _dump_with_ref(res: CallResult) -> None:
            if dump_fp is None:
                return
            rec = {
                "request_id":          res.request_id,
                "ok":                  res.ok,
                "status":              res.status,
                "latency_ms":          round(res.latency_ms, 1),
                "worker_id":           res.worker_id,
                "system_prompt":       res.system_prompt,
                "user_audio_path":     res.user_audio_path,
                # 注意：``user_question`` 是我们合成 wav 用的 prompt，
                # 不是 server 端 ASR 出来的实际识别文本——server 端没暴露
                # 中间识别。但只要 VoxCPM2 没明显合成失误，二者基本一致。
                "user_question":       cur_ref.get(res.request_id or "", ""),
                "full_text":           res.full_text,
                "total_chunks":        res.total_chunks,
                "speak_chunks":        res.speak_chunks,
                "audio_duration_s":    res.audio_duration_s,
                "logits_file":         res.logits_file,
                "err":                 res.err,
            }
            dump_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dump_fp.flush()

        async def worker_task(wid: int):
            for k in range(args.total_per_worker):
                # 轮询问句池 —— 每个并发任务从不同偏移开始，避免所有 worker
                # 第一发都打同一段 wav 导致 server 端 cache 命中假装变快。
                wav_path, ref_text = user_audio_pool[
                    (wid * args.total_per_worker + k) % len(user_audio_pool)]
                rid = f"stress_dup_{wid}_{k}"
                cur_ref[rid] = ref_text
                res = await call_duplex(
                    client,
                    user_audio=wav_path,
                    request_id=rid,
                    timeout_s=args.timeout_s,
                    logits_dir=args.logits_dir,
                )
                async with results_lock:
                    results.append(res)
                    _dump_with_ref(res)
                    progress[0] += 1
                    text_preview = (res.full_text or "")[:60].replace("\n", " ")
                    print(f"    [progress {progress[0]}/{total_target}] "
                          f"wid={wid} k={k} ok={res.ok} status={res.status} "
                          f"lat={res.latency_ms/1000:.1f}s "
                          f"chunks={res.total_chunks} "
                          f"speak={res.speak_chunks} "
                          f"worker={res.worker_id} "
                          f"file={Path(res.logits_file).name if res.logits_file else '-'} "
                          f"q={ref_text!r} "
                          f"text={text_preview!r}")
                    if not res.ok:
                        print(f"      err: {res.err}")

        t0 = time.perf_counter()
        await asyncio.gather(*[asyncio.create_task(worker_task(i))
                               for i in range(args.concurrency)])
        elapsed = time.perf_counter() - t0
        stop_evt.set()
        await mon_task

        n = len(results)
        ok = sum(1 for r in results if r.ok)
        fail_5xx = sum(1 for r in results if not r.ok and 500 <= r.status < 600)
        fail_4xx = sum(1 for r in results if not r.ok and 400 <= r.status < 500)
        fail_net = sum(1 for r in results if r.status == -1)
        fail_app = sum(1 for r in results if not r.ok and r.status == 200)
        lats = [r.latency_ms for r in results if r.ok]
        by_worker: Dict[str, int] = {}
        for r in results:
            if r.worker_id:
                by_worker[r.worker_id] = by_worker.get(r.worker_id, 0) + 1

        files_returned = [r.logits_file for r in results
                          if r.logits_file is not None]
        files_on_disk: List[str] = []
        if os.path.isdir(args.logits_dir):
            for root, _dirs, fs in os.walk(args.logits_dir):
                for f in fs:
                    if f.endswith(".safetensors"):
                        files_on_disk.append(f)
        unique_basenames = sorted({Path(f).name for f in files_returned})
        unique_disk = sorted(set(files_on_disk))
        unmatched = [b for b in unique_disk if not _FILENAME_RE.match(b)]
        only_duplex = [b for b in unique_disk if b.startswith("duplex_")]

        worker_states_seen: set = set()
        max_qsize = 0
        ever_error = False
        for s in snapshots:
            for k in s.worker_states.keys():
                worker_states_seen.add(k)
                if k.lower() in ("error", "errored", "failed", "dead"):
                    ever_error = True
            if s.queue_size is not None:
                max_qsize = max(max_qsize, s.queue_size)
        if snapshots and snapshots[0].gpu_mem_used:
            n_gpu = len(snapshots[0].gpu_mem_used)
            gpu_drift = []
            for i in range(n_gpu):
                vals = [s.gpu_mem_used[i] for s in snapshots
                        if i < len(s.gpu_mem_used)]
                gpu_drift.append({
                    "gpu_idx": i,
                    "min": min(vals), "max": max(vals),
                    "drift_mb": max(vals) - min(vals),
                    "first": vals[0], "last": vals[-1],
                })
        else:
            gpu_drift = []

        report = {
            "base_url": args.base_url,
            "user_audio": user_audio,
            "concurrency": args.concurrency,
            "total_per_worker": args.total_per_worker,
            "n_workers_reported": n_workers,
            "elapsed_s": round(elapsed, 1),
            "n": n, "ok": ok, "fail_5xx": fail_5xx, "fail_4xx": fail_4xx,
            "fail_net": fail_net, "fail_app": fail_app,
            "p50_ms": round(percentile(lats, 50), 1),
            "p95_ms": round(percentile(lats, 95), 1),
            "p99_ms": round(percentile(lats, 99), 1),
            "max_ms": round(max(lats) if lats else 0.0, 1),
            "by_worker": by_worker,
            "logits_files_returned": len(files_returned),
            "logits_files_unique_basenames": len(unique_basenames),
            "logits_files_on_disk_unique": len(unique_disk),
            "logits_filename_pattern_unmatched": unmatched,
            "logits_files_starting_with_duplex": len(only_duplex),
            "monitor_states_seen": sorted(worker_states_seen),
            "monitor_ever_error_state": ever_error,
            "monitor_max_queue_size": max_qsize,
            "gpu": gpu_drift,
        }

        fails: List[str] = []
        warns: List[str] = []
        if fail_5xx > 0:
            fails.append(f"5xx={fail_5xx}")
        if fail_net > 0:
            fails.append(f"network/timeout={fail_net}")
        if ever_error:
            fails.append(f"worker error state seen: {sorted(worker_states_seen)}")
        if files_returned and len(unique_basenames) != len(files_returned):
            fails.append(
                f"filename collision in returned paths: "
                f"{len(files_returned)} returned vs "
                f"{len(unique_basenames)} unique basenames"
            )
        if files_returned and len(unique_disk) < len(unique_basenames):
            fails.append(
                f"filename collision on disk: "
                f"{len(unique_disk)} on disk vs "
                f"{len(unique_basenames)} returned"
            )
        if unmatched:
            fails.append(f"filenames not matching new pattern: {unmatched[:3]}")
        if fail_app > 0:
            warns.append(f"success=false (200) ={fail_app}")
        if fail_4xx > 0:
            warns.append(f"4xx={fail_4xx}")
        for g in gpu_drift:
            if g["drift_mb"] > 2000:
                warns.append(
                    f"GPU{g['gpu_idx']} mem drift={g['drift_mb']}MB "
                    f"(first={g['first']}, last={g['last']})"
                )
        passed = len(fails) == 0
        report["passed"] = passed
        report["reasons"] = fails + [f"WARN: {w}" for w in warns]

        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".",
                        exist_ok=True)
            with open(args.report, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"[+] report written to {args.report}")

        print("\n" + "=" * 70)
        print("DUPLEX LOAD TEST REPORT")
        print("=" * 70)
        print(f"  duration         : {elapsed:.1f}s, c={args.concurrency}")
        print(f"  n={n} ok={ok} 5xx={fail_5xx} net={fail_net} "
              f"app_err={fail_app}")
        print(f"  latency: p50={report['p50_ms']:.0f}ms "
              f"p95={report['p95_ms']:.0f}ms p99={report['p99_ms']:.0f}ms "
              f"max={report['max_ms']:.0f}ms")
        print(f"  by_worker: {by_worker}")
        print(f"  logits files returned   : {len(files_returned)}")
        print(f"  logits files unique     : {len(unique_basenames)}")
        print(f"  logits files on disk    : {len(unique_disk)}")
        print(f"  filename pattern OK     : {not unmatched} "
              f"({len(unmatched)} mismatches)")
        print(f"  monitor: states={sorted(worker_states_seen)} "
              f"max_qsize={max_qsize}")
        for g in gpu_drift:
            print(f"  GPU{g['gpu_idx']}: first={g['first']}MB last={g['last']}MB "
                  f"drift={g['drift_mb']}MB")
        print("-" * 70)
        if dump_fp is not None:
            dump_fp.close()
        if passed:
            print("RESULT: PASS")
            for w in report["reasons"]:
                print(f"  {w}")
            return 0
        print("RESULT: FAIL")
        for w in report["reasons"]:
            print(f"  {w}")
        return 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--user-audio", required=True,
                    help=("16kHz mono wav 路径；也可为 manifest.json（由 "
                          "synth_questions_voxcpm.py 生成，每条含 wav+text）"
                          "或目录（其下所有 .wav 轮询）"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--total-per-worker", type=int, default=3,
                    help="N requests serially per concurrent task")
    ap.add_argument("--timeout-s", type=float, default=300,
                    help="single duplex_offline request timeout")
    ap.add_argument("--monitor-interval-s", type=float, default=5.0)
    ap.add_argument("--gpu-ids", default="",
                    help="comma-separated host GPU indices for nvidia-smi sampling")
    ap.add_argument("--logits-dir", default="/tmp/minicpm_logits_duplex_stress")
    ap.add_argument("--report", default="")
    ap.add_argument("--dump-conversations", default="",
                    help=("dump per-request {system_prompt, user_audio_path, "
                          "full_text, ...} as JSONL for human review"))
    return ap.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
