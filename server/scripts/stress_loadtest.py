#!/usr/bin/env python3
"""离线压测：MiniCPM-o batch_server (gateway + N worker)

目标
----

在 4 卡部署（gateway:8080 + worker × 4）上做端到端压力测试，确认：

1. 阶梯并发下不出 5xx / 不超时
2. /v1/workers 状态始终是 idle/busy（不出 error）
3. GPU 显存不持续增长（无明显 leak）
4. p50/p95/p99 latency 在合理范围
5. 多种 chat 路径（greedy / sampled+seed / logits inline / logits file）混合都能正常返回

不测 /v1/duplex_offline —— 它需要 user_audio_path 文件、且单次几十秒会拖住 worker，
对于"看会不会有故障"这个目标，单工 chat 已能覆盖大部分故障路径（FIFO 队列、
sampler 重建、logits 写盘、TTS attractor 兜底）。如果想加，给 --include-duplex
加一个固定 wav 路径即可（暂未实现）。

用法
----
    python3 scripts/stress_loadtest.py --base-url http://localhost:8080 \
        --duration 300 --max-concurrency 32

注意
----
- 这个脚本**只**通过 HTTP 黑盒打 server，不读任何内部代码 / 文件，方便在另一台
  机器上做远程压测。
- 报告里所有 latency 单位 = 毫秒。
- 输出最后会打印 "PASS" 或 "FAIL: <原因>"，CI 可直接消费 exit code。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ============================================================
# 请求模板池（每个 case 一个 prompt + 一组参数）
# ============================================================

PROMPTS = [
    "用一句话介绍 MiniCPM-o 的多模态能力。",
    "解释什么是大语言模型的注意力机制。",
    "中国四大名著是哪四本？请用一句话回答。",
    "如何用 Python 计算斐波那契数列？请只给思路。",
    "请写一句温暖的早安问候。",
    "杭州有哪些值得一去的景点？请列举三个。",
    "请用一句话总结相对论。",
    "什么是 RLHF？请用一句话解释。",
]


def make_chat_body(case: str, idx: int) -> Dict[str, Any]:
    """构造一条 chat 请求 body。case 决定走哪条路径。"""

    prompt = PROMPTS[idx % len(PROMPTS)]
    base = {
        "messages": [{"role": "user", "content": prompt}],
        "tts": {"enabled": False},
        "request_id": f"stress_{case}_{idx}",
    }

    if case == "greedy":
        base["generation"] = {"max_new_tokens": 32, "do_sample": False}

    elif case == "sampled_seed":
        base["generation"] = {
            "max_new_tokens": 32,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": 1000 + (idx % 1024),
        }

    elif case == "logits_inline":
        base["generation"] = {"max_new_tokens": 8, "do_sample": False}
        base["logits"] = {"enabled": True, "format": "inline"}

    elif case == "logits_file":
        base["generation"] = {"max_new_tokens": 16, "do_sample": False}
        base["logits"] = {
            "enabled": True,
            "format": "file",
            "output_dir": "/tmp/minicpm_logits_stress",
        }

    elif case == "long_greedy":
        base["generation"] = {"max_new_tokens": 64, "do_sample": False}

    else:
        raise ValueError(f"unknown case: {case!r}")

    return base


CASE_WEIGHTS: List[Tuple[str, int]] = [
    ("greedy",        4),
    ("sampled_seed",  3),
    ("logits_inline", 2),
    ("logits_file",   2),
    ("long_greedy",   1),
]


def pick_case(rng: random.Random) -> str:
    """按权重抽 case。"""

    total = sum(w for _, w in CASE_WEIGHTS)
    r = rng.uniform(0, total)
    upto = 0.0
    for case, w in CASE_WEIGHTS:
        upto += w
        if r <= upto:
            return case
    return CASE_WEIGHTS[-1][0]


# ============================================================
# 单条请求统计
# ============================================================


@dataclass
class CallResult:
    case: str
    ok: bool
    status: int
    latency_ms: float
    err: Optional[str] = None
    worker_id: Optional[str] = None
    queue_wait_ms: Optional[float] = None
    tokens_generated: Optional[int] = None
    # ---- 人工审阅：保存请求 prompt 和模型回复文本，用于事后看回答质量 ----
    # 单工 chat 的 ChatResponse.text 字段，是 detokenize 后的可读字符串。
    # 注意：use_tts=true 路径下，LLM 在 ``<|tts_bos|>`` 后会继续吐 token；
    # 这些 token 大多数是 audio embedding 编号（id < 0 的占位符或 codec
    # token），detokenize 出来可能是 garbage —— 这是**预期行为**，因为
    # ``<|tts_bos|>`` 把模型推向了语音通道。RL rollout 真正应消费 token_ids
    # + logits（走 ``logits_export``），而不是 ``text``。这里 dump 出来纯粹
    # 为了"人工大致看一眼回答方向对不对"，不是质量打分依据。
    prompt: str = ""
    response_text: str = ""
    request_id: Optional[str] = None


# ============================================================
# 单次 chat 调用（带超时和异常捕获）
# ============================================================


async def call_chat(
    client: httpx.AsyncClient,
    case: str,
    idx: int,
    timeout_s: float,
) -> CallResult:
    body = make_chat_body(case, idx)
    # 抽出 user prompt 用于后续 dump（即使请求失败，也能让人知道是什么 prompt）
    prompt = ""
    msgs = body.get("messages", [])
    if msgs:
        c = msgs[0].get("content", "")
        prompt = c if isinstance(c, str) else str(c)
    rid = body.get("request_id")

    t0 = time.perf_counter()
    try:
        r = await client.post("/v1/chat", json=body, timeout=timeout_s)
    except httpx.TimeoutException as e:
        return CallResult(
            case=case, ok=False, status=-1,
            latency_ms=(time.perf_counter() - t0) * 1000,
            err=f"timeout: {e!s}",
            prompt=prompt, request_id=rid,
        )
    except httpx.RequestError as e:
        return CallResult(
            case=case, ok=False, status=-1,
            latency_ms=(time.perf_counter() - t0) * 1000,
            err=f"network: {type(e).__name__}: {e!s}",
            prompt=prompt, request_id=rid,
        )
    lat = (time.perf_counter() - t0) * 1000
    if r.status_code != 200:
        return CallResult(
            case=case, ok=False, status=r.status_code, latency_ms=lat,
            err=r.text[:300],
            prompt=prompt, request_id=rid,
        )
    try:
        p = r.json()
    except Exception as e:
        return CallResult(
            case=case, ok=False, status=r.status_code, latency_ms=lat,
            err=f"json-decode: {e!s}",
            prompt=prompt, request_id=rid,
        )
    success = bool(p.get("success"))
    err = None if success else (p.get("error") or "success=false")
    return CallResult(
        case=case, ok=success, status=r.status_code, latency_ms=lat,
        err=err,
        worker_id=p.get("worker_id"),
        queue_wait_ms=p.get("queue_wait_ms"),
        tokens_generated=p.get("tokens_generated"),
        prompt=prompt,
        response_text=p.get("text") or "",
        request_id=p.get("request_id") or rid,
    )


# ============================================================
# 阶段执行器：保持 N 并发，跑 duration_s 秒
# ============================================================


@dataclass
class PhaseStats:
    name: str
    concurrency: int
    duration_s: float
    started_at: float = 0.0
    ended_at: float = 0.0
    results: List[CallResult] = field(default_factory=list)


async def run_phase(
    client: httpx.AsyncClient,
    phase: PhaseStats,
    rng: random.Random,
    timeout_s: float,
    on_progress=None,
    dump_writer=None,
) -> None:
    """跑 N 个 worker task，每个 task 串行打 /v1/chat 直到时间到。

    ``dump_writer`` 可选，签名 ``(phase_name, CallResult) -> None``，每条
    请求完成后被调用一次（用于 ``--dump-conversations`` 把 prompt+response
    实时落盘到 JSONL，避免脚本被 ctrl-c 后丢数据）。
    """

    phase.started_at = time.perf_counter()
    deadline = phase.started_at + phase.duration_s
    counter = [0]

    async def worker(wid: int) -> None:
        local_idx = wid * 100000  # 防 idx 撞车导致同一 prompt 抽到同一 case
        while time.perf_counter() < deadline:
            case = pick_case(rng)
            res = await call_chat(client, case, local_idx, timeout_s)
            phase.results.append(res)
            if dump_writer is not None:
                try:
                    dump_writer(phase.name, res)
                except Exception:
                    pass  # dump 失败不打断压测
            counter[0] += 1
            if on_progress is not None and counter[0] % 20 == 0:
                on_progress(counter[0], len(phase.results))
            local_idx += 1

    tasks = [asyncio.create_task(worker(i)) for i in range(phase.concurrency)]
    await asyncio.gather(*tasks)
    phase.ended_at = time.perf_counter()


# ============================================================
# 周期监控：worker 状态 / queue / GPU mem
# ============================================================


@dataclass
class Snapshot:
    t: float
    worker_states: Dict[str, int]              # status -> count
    worker_total: int
    queue_size: Optional[int]
    gpu_mem_used: List[int]                    # MB per visible GPU


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
                qsize = p.get("size") or p.get("queue_size") or 0
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
# 报告 / 故障判定
# ============================================================


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = max(0, min(len(ys) - 1, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


def summarize_phase(ph: PhaseStats) -> Dict[str, Any]:
    n = len(ph.results)
    ok = sum(1 for r in ph.results if r.ok)
    fail_5xx = sum(1 for r in ph.results
                   if not r.ok and 500 <= r.status < 600)
    fail_4xx = sum(1 for r in ph.results
                   if not r.ok and 400 <= r.status < 500)
    fail_net = sum(1 for r in ph.results if r.status == -1)
    fail_app = sum(1 for r in ph.results
                   if not r.ok and r.status == 200)
    lats = [r.latency_ms for r in ph.results if r.ok]
    qwaits = [r.queue_wait_ms for r in ph.results
              if r.queue_wait_ms is not None]
    elapsed = max(0.001, ph.ended_at - ph.started_at)
    by_worker: Dict[str, int] = {}
    for r in ph.results:
        if r.worker_id:
            by_worker[r.worker_id] = by_worker.get(r.worker_id, 0) + 1
    by_case: Dict[str, Dict[str, Any]] = {}
    for r in ph.results:
        d = by_case.setdefault(r.case, {"n": 0, "ok": 0, "lats": []})
        d["n"] += 1
        if r.ok:
            d["ok"] += 1
            d["lats"].append(r.latency_ms)
    case_summary = {}
    for case, d in by_case.items():
        case_summary[case] = {
            "n": d["n"],
            "ok": d["ok"],
            "p50": round(percentile(d["lats"], 50), 1),
            "p95": round(percentile(d["lats"], 95), 1),
        }
    return {
        "phase": ph.name,
        "concurrency": ph.concurrency,
        "duration_s": round(elapsed, 1),
        "total": n,
        "ok": ok,
        "fail_5xx": fail_5xx,
        "fail_4xx": fail_4xx,
        "fail_net": fail_net,
        "fail_app": fail_app,
        "rps": round(n / elapsed, 2),
        "p50_ms": round(percentile(lats, 50), 1),
        "p95_ms": round(percentile(lats, 95), 1),
        "p99_ms": round(percentile(lats, 99), 1),
        "max_ms": round(max(lats) if lats else 0.0, 1),
        "queue_p95_ms": round(percentile(qwaits, 95), 1) if qwaits else 0.0,
        "by_worker": by_worker,
        "by_case": case_summary,
    }


def summarize_snapshots(snaps: List[Snapshot]) -> Dict[str, Any]:
    if not snaps:
        return {}
    ever_error_state = False
    state_set: set = set()
    max_qsize = 0
    for s in snaps:
        for k in s.worker_states.keys():
            state_set.add(k)
            if k.lower() in ("error", "errored", "failed", "dead"):
                ever_error_state = True
        if s.queue_size is not None:
            max_qsize = max(max_qsize, s.queue_size)
    if not snaps[0].gpu_mem_used:
        gpu_drift = []
    else:
        n_gpu = len(snaps[0].gpu_mem_used)
        gpu_drift = []
        for i in range(n_gpu):
            vals = [s.gpu_mem_used[i] for s in snaps
                    if i < len(s.gpu_mem_used)]
            gpu_drift.append({
                "gpu_idx": i,
                "min": min(vals),
                "max": max(vals),
                "drift_mb": max(vals) - min(vals),
                "first": vals[0],
                "last": vals[-1],
            })
    return {
        "n_samples": len(snaps),
        "worker_states_seen": sorted(state_set),
        "ever_error_state": ever_error_state,
        "max_queue_size": max_qsize,
        "gpu": gpu_drift,
    }


def decide_pass_fail(
    phases: List[Dict[str, Any]],
    monitor: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """返回 (passed, reasons)。reasons 里区分 FAIL / WARN，FAIL 直接判失败。"""

    fails: List[str] = []
    warns: List[str] = []
    for ph in phases:
        if ph["fail_5xx"] > 0:
            fails.append(f"phase {ph['phase']}: 5xx={ph['fail_5xx']}")
        if ph["fail_net"] > 0:
            fails.append(f"phase {ph['phase']}: network/timeout={ph['fail_net']}")
        if ph["fail_app"] > 0:
            warns.append(f"phase {ph['phase']}: success=false (200) ={ph['fail_app']}")
        if ph["fail_4xx"] > 0:
            warns.append(f"phase {ph['phase']}: 4xx={ph['fail_4xx']}")
    if monitor.get("ever_error_state"):
        fails.append(
            f"worker entered error state at some point "
            f"(states seen: {monitor.get('worker_states_seen')})"
        )
    for g in monitor.get("gpu") or []:
        if g["drift_mb"] > 2000:
            warns.append(
                f"GPU{g['gpu_idx']} memory drift={g['drift_mb']}MB "
                f"(first={g['first']}, last={g['last']})"
            )
    return len(fails) == 0, fails + [f"WARN: {w}" for w in warns]


# ============================================================
# 主入口
# ============================================================


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] \
        if args.gpu_ids else []

    # ``--dump-conversations`` 把每条 (prompt, response) 对实时落到 JSONL，
    # 这样 1) 中途 ctrl-c 不丢数据，2) 可以另开 shell ``tail -f`` 看进度，
    # 3) 跑完后给人审阅"模型回答得对不对"。我们故意不在内存里 buffer 整批
    # ——压测可能成千上万条，buffer 起来 RAM 撑爆，反而干扰服务侧测量。
    dump_fp = None
    dump_lock_obj = None
    if args.dump_conversations:
        dump_path = os.path.abspath(args.dump_conversations)
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        dump_fp = open(dump_path, "w", encoding="utf-8")
        dump_lock_obj = asyncio.Lock()  # noqa: F841 (kept for future use)
        print(f"[+] dumping conversations to {dump_path}")

    def _dump(phase_name: str, res: CallResult) -> None:
        if dump_fp is None:
            return
        rec = {
            "phase":          phase_name,
            "case":           res.case,
            "request_id":     res.request_id,
            "ok":             res.ok,
            "status":         res.status,
            "latency_ms":     round(res.latency_ms, 1),
            "worker_id":      res.worker_id,
            "tokens":         res.tokens_generated,
            "prompt":         res.prompt,
            "response_text":  res.response_text,
            "err":            res.err,
        }
        dump_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        dump_fp.flush()

    transport_limits = httpx.Limits(
        max_connections=args.max_concurrency * 2,
        max_keepalive_connections=args.max_concurrency,
    )
    async with httpx.AsyncClient(
        base_url=args.base_url,
        limits=transport_limits,
    ) as client:
        try:
            r = await client.get("/v1/health", timeout=10)
            health_ok = r.status_code == 200
        except Exception as e:
            print(f"[!] gateway health check failed: {e}", file=sys.stderr)
            return 2
        if not health_ok:
            print(f"[!] /v1/health -> {r.status_code}: {r.text[:200]}",
                  file=sys.stderr)
            return 2
        try:
            r = await client.get("/v1/workers", timeout=10)
            n_workers = (r.json() or {}).get("total", 0)
        except Exception:
            n_workers = 0
        print(f"[+] gateway healthy, workers reported: {n_workers}")

        print("[+] warmup: 1 concurrency × 4 calls, sequential ...")
        warmup = PhaseStats(name="warmup", concurrency=1, duration_s=0)
        warmup.started_at = time.perf_counter()
        for i in range(4):
            wres = await call_chat(client, "greedy", i, args.timeout_s)
            warmup.results.append(wres)
            _dump("warmup", wres)
        warmup.ended_at = time.perf_counter()
        print(f"    warmup done in "
              f"{warmup.ended_at - warmup.started_at:.1f}s, "
              f"ok={sum(1 for r in warmup.results if r.ok)}/{len(warmup.results)}")

        budget = float(args.duration)
        per_phase = max(20.0, budget / 4.0)
        plan: List[PhaseStats] = []
        if args.max_concurrency >= 4:
            plan.append(PhaseStats(name="phase1_c4",
                                   concurrency=min(4, args.max_concurrency),
                                   duration_s=per_phase))
        if args.max_concurrency >= 8:
            plan.append(PhaseStats(name="phase2_c8",
                                   concurrency=min(8, args.max_concurrency),
                                   duration_s=per_phase))
        if args.max_concurrency >= 16:
            plan.append(PhaseStats(name="phase3_c16",
                                   concurrency=min(16, args.max_concurrency),
                                   duration_s=per_phase))
        plan.append(PhaseStats(name=f"phase4_c{args.max_concurrency}",
                               concurrency=args.max_concurrency,
                               duration_s=per_phase))

        snapshots: List[Snapshot] = []
        stop_evt = asyncio.Event()
        mon_task = asyncio.create_task(
            monitor_loop(client, args.monitor_interval_s,
                         stop_evt, snapshots, gpu_ids))

        for ph in plan:
            print(
                f"[+] phase {ph.name}: concurrency={ph.concurrency}, "
                f"duration={ph.duration_s:.0f}s ..."
            )
            await run_phase(client, ph, rng, args.timeout_s,
                            on_progress=lambda total, n_results, name=ph.name:
                                print(f"    [{name}] +{total} done "
                                      f"({n_results} results so far)",
                                      flush=True),
                            dump_writer=_dump)
            done_summary = summarize_phase(ph)
            print(f"    [{ph.name}] n={done_summary['total']} "
                  f"ok={done_summary['ok']} "
                  f"5xx={done_summary['fail_5xx']} "
                  f"net={done_summary['fail_net']} "
                  f"app_err={done_summary['fail_app']} "
                  f"rps={done_summary['rps']} "
                  f"p50={done_summary['p50_ms']}ms "
                  f"p95={done_summary['p95_ms']}ms")

        stop_evt.set()
        await mon_task

        phase_summaries = [summarize_phase(ph) for ph in plan]
        warmup_summary = summarize_phase(warmup)
        monitor_summary = summarize_snapshots(snapshots)
        passed, reasons = decide_pass_fail(phase_summaries, monitor_summary)

        report = {
            "base_url": args.base_url,
            "duration_requested_s": args.duration,
            "max_concurrency": args.max_concurrency,
            "n_workers_reported": n_workers,
            "warmup": warmup_summary,
            "phases": phase_summaries,
            "monitor": monitor_summary,
            "passed": passed,
            "reasons": reasons,
        }
        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".",
                        exist_ok=True)
            with open(args.report, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"[+] report written to {args.report}")

        print("\n" + "=" * 70)
        print("LOAD TEST REPORT")
        print("=" * 70)
        for ph in phase_summaries:
            print(
                f"  [{ph['phase']:<14}] c={ph['concurrency']:<2} "
                f"n={ph['total']:<5} ok={ph['ok']:<5} "
                f"5xx={ph['fail_5xx']:<2} net={ph['fail_net']:<2} "
                f"appErr={ph['fail_app']:<2} | "
                f"rps={ph['rps']:<5} "
                f"p50={ph['p50_ms']:.0f}ms p95={ph['p95_ms']:.0f}ms "
                f"p99={ph['p99_ms']:.0f}ms"
            )
            print(f"      by_worker: {ph['by_worker']}")
            cases = ", ".join(
                f"{c}=ok{d['ok']}/{d['n']} p95={d['p95']:.0f}ms"
                for c, d in ph["by_case"].items()
            )
            print(f"      by_case  : {cases}")
        print(f"  monitor: states={monitor_summary.get('worker_states_seen')} "
              f"max_qsize={monitor_summary.get('max_queue_size')}")
        for g in monitor_summary.get("gpu") or []:
            print(f"      GPU{g['gpu_idx']}: "
                  f"first={g['first']}MB last={g['last']}MB "
                  f"drift={g['drift_mb']}MB")
        print("-" * 70)
        if dump_fp is not None:
            dump_fp.close()
        if passed:
            print("RESULT: PASS")
            for w in reasons:
                print(f"  {w}")
            return 0
        print("RESULT: FAIL")
        for w in reasons:
            print(f"  {w}")
        return 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://localhost:8080",
                    help="gateway URL")
    ap.add_argument("--duration", type=float, default=240,
                    help="总时长（秒），平均分到各阶段")
    ap.add_argument("--max-concurrency", type=int, default=32,
                    help="最大并发数（最后一个阶段用）")
    ap.add_argument("--timeout-s", type=float, default=120,
                    help="单次 chat 请求超时")
    ap.add_argument("--monitor-interval-s", type=float, default=5.0)
    ap.add_argument("--gpu-ids", default="",
                    help="逗号分隔的 GPU 物理 index 列表（用于 nvidia-smi 采样）")
    ap.add_argument("--seed", type=int, default=20260521)
    ap.add_argument("--report", default="",
                    help="把 JSON 报告写到此路径")
    ap.add_argument("--dump-conversations", default="",
                    help=("把每条 (prompt, response_text) 实时落到 JSONL 文件。"
                          "每行 1 条，schema：{phase, case, request_id, ok, status, "
                          "latency_ms, worker_id, tokens, prompt, response_text, err}。"
                          "便于人工审阅模型回答质量。"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
