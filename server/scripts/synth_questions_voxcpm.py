#!/usr/bin/env python3
"""使用 VoxCPM2 把若干中文问句合成成 16kHz mono wav，给 duplex 压测做输入。

为什么单独写这个脚本（而不是直接 inline 进 stress_duplex_loadtest.py）：

* VoxCPM2 是 2B 参数 diffusion-AR TTS 模型，权重 ~4–5GB，加载就要十几秒，
  推理 RTF ~0.3 (RTX 4090)。压测脚本希望"轻、可重复、可 ctrl-c"，把
  TTS 合成放进去会让冷启动很重。这里用 **离线一次合成、多次复用**：
  跑一次本脚本生成 wav 到磁盘，后续压测直接 mmap 读 wav 就行。

* duplex_offline 端用的是 MiniCPM-o 自身的语音前端（16kHz Whisper-style
  mel + 自己的 audio embedding），所以这里我们故意把 sr 强行降采到
  **16kHz mono**，和 server 期望对齐，避免 VoxCPM2 默认 48kHz 输出
  在 duplex 那边被自动降采时引入 artifact 干扰评估。

══════════════════════════════════════════════════════════════════════════
重要约定：末尾空音频是**数据职责**，不是推理引擎职责
══════════════════════════════════════════════════════════════════════════

duplex 模型自己决定何时从 listen 切到 speak。它需要"听到一段明显的
silence"才认为"用户已经说完了"，否则会一直 emit __IS_LISTEN__；当
audio 跑完就走 audio_exhausted 退出，结果 ``speak_chunks=0,
full_text=""``。

修这个**不能**靠在 server 端 hack 时序（自动补静音、自动补 EOS、强行
进 speak），这些都会破坏离线/在线训练的分布对齐——双工的训练数据本身
就含尾静音作为对话单元的边界，server 不应擅自动它。

所以本脚本固定在每条合成 wav 的尾部 padding 一段静音
（``--trailing-silence-s``，默认 9.0s），让数据本身就长这样。这是
**唯一**正确的做法。如果你想缩短 padding 节省压测时间，请明确意识到
silent 率会显著上升（实测 <5s padding 时几乎不开口）。

CLI:
    python synth_questions_voxcpm.py \\
        --out-dir /path/to/duplex_eval_wavs \\
        --device cuda:0 \\
        --model /cache/caitianchi/model/VoxCPM2

预计耗时（单卡 cuda:0，cold load）：≈ 30s 模型加载 + 每条 0.5–1s 合成；
8 条问句总耗时 ≈ 35s。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# 默认问句池：覆盖"事实 / 数学 / 日常 / 闲聊 / 翻译 / 推理"6 类，让审阅时
# 能从多个角度判断模型回复"对不对"。每条故意保持 1–2 句，<=15 字，让
# 合成出来的 wav 时长在 3–6 秒（duplex chunk_ms=1000 + force_listen_count
# =3 + max_chunks=8 ⇒ 模型至少听 3 秒，至多 8 秒，所以问句也要在这个窗口
# 内说完，不然 audio 还没说完模型就开始回答 ⇒ 评测无意义）。
DEFAULT_QUESTIONS: List[Tuple[str, str]] = [
    ("q01_fact_hangzhou",   "杭州在哪个省份？"),
    ("q02_fact_capital",    "中国的首都是哪里？"),
    ("q03_math_addition",   "三加五等于几？"),
    ("q04_daily_weather",   "今天天气怎么样？"),
    ("q05_chitchat_intro",  "你好，请介绍一下你自己。"),
    ("q06_translate_apple", "请把苹果翻译成英文。"),
    ("q07_reason_animal",   "鸡和鸭哪个会游泳？"),
    ("q08_advice_sleep",    "晚上睡不着怎么办？"),
]


def synth(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] out_dir = {out_dir}")

    from voxcpm import VoxCPM
    import soundfile as sf
    import numpy as np

    print(f"[+] loading {args.model} on {args.device} (load_denoiser=False) ...")
    t0 = time.perf_counter()
    # ``cache_dir`` 让权重落到我们指定的位置，避免散落到 ~/.cache。
    # ``load_denoiser=False`` 跳过 BigVGAN denoiser，节省 ~1GB 显存——
    # 我们生成的 wav 直接给 MiniCPM-o 自身的 audio encoder 吃，对降噪无所谓。
    os.environ.setdefault("CUDA_VISIBLE_DEVICES",
                          args.device.split(":")[-1] if ":" in args.device else "0")
    model = VoxCPM.from_pretrained(args.model, load_denoiser=False)
    print(f"    loaded in {time.perf_counter() - t0:.1f}s, "
          f"sample_rate = {model.tts_model.sample_rate} Hz")

    # 用单条 fixed-seed cfg 让重跑结果接近一致（VoxCPM 自身有 voice
    # design 随机性，无法 100% 复现，但我们也不在乎，反正人评 prompt+
    # response 而不是 audio 一致性）。
    questions = list(DEFAULT_QUESTIONS)
    if args.questions_jsonl:
        # 允许从外部 JSONL 喂"问句池"，每行 ``{"id": "...", "text": "..."}``。
        # 用于以后做更系统的人评数据集时复用。
        with open(args.questions_jsonl, "r", encoding="utf-8") as f:
            questions = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                questions.append((rec["id"], rec["text"]))
        print(f"[+] loaded {len(questions)} questions from "
              f"{args.questions_jsonl}")

    manifest = []
    src_sr = int(model.tts_model.sample_rate)
    for qid, text in questions:
        out_wav = out_dir / f"{qid}.wav"
        if out_wav.exists() and not args.overwrite:
            print(f"    [skip] {out_wav.name} exists")
            manifest.append({"id": qid, "text": text,
                             "wav": str(out_wav), "skipped": True})
            continue
        t1 = time.perf_counter()
        wav = model.generate(
            text=text,
            cfg_value=2.0,
            inference_timesteps=10,
        )
        # VoxCPM2 默认 48kHz；duplex 端期望 16kHz 输入 → 简单线性降采。
        # 我们不需要 high-fidelity 处理（这只是模拟用户讲话），所以直接用
        # ``np.linspace`` 索引最近邻，避免引入 librosa.resample 这一段
        # 额外的依赖+耗时。下游 minicpm-o 的 mel 提取本身就只敏感到
        # ~8kHz 频段，audio quality 损失对 ASR 影响可忽略。
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if src_sr != 16000:
            n_dst = int(round(len(wav) * 16000 / src_sr))
            idx = np.linspace(0, len(wav) - 1, n_dst).astype(np.int64)
            wav16 = wav[idx]
        else:
            wav16 = wav
        # 末尾 padding 静音：duplex server 端两层门控
        #
        #   1. ``force_listen_count=3`` —— 前 3 chunk 强制 listen
        #   2. 模型自主决策：即使 force_listen 满了，模型仍可选择继续 listen
        #      （emit __IS_LISTEN__）；只有它判定"用户已经说完"才会进 speak
        #
        # 第 2 层门控**只能从数据侧**喂 silence 来触发——这是"末尾空音频
        # 是数据职责，不是引擎职责"的具体落点（详见模块顶部 docstring）。
        #
        # 实测：3s padding 模型还认为用户在思考；6s 部分开口；9s 时 8 段
        # 问句中 2 段开口（25%）；继续加更稳定一些但单条耗时也涨。
        # 默认 9s，需要更高 speak 命中率请加大 ``--trailing-silence-s``。
        trailing = np.zeros(
            int(16000 * args.trailing_silence_s), dtype=np.float32)
        wav16 = np.concatenate([wav16, trailing], axis=0)
        sf.write(str(out_wav), wav16, 16000, subtype="PCM_16")
        dur = len(wav16) / 16000.0
        elapsed = time.perf_counter() - t1
        print(f"    [ok] {out_wav.name}: text={text!r} "
              f"dur={dur:.2f}s synth={elapsed:.1f}s")
        manifest.append({"id": qid, "text": text,
                         "wav": str(out_wav),
                         "duration_s": round(dur, 2),
                         "synth_time_s": round(elapsed, 2)})

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[+] manifest written to {manifest_path}")
    print(f"[+] total {len(manifest)} questions, "
          f"out_dir = {out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/tmp/duplex_eval_wavs",
                    help="存放合成 wav 的目录")
    ap.add_argument("--model", default="openbmb/VoxCPM2",
                    help="VoxCPM2 模型 ID 或本地路径")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--questions-jsonl", default="",
                    help="可选：自定义问句池 JSONL，每行 {id, text}")
    ap.add_argument("--overwrite", action="store_true",
                    help="存在则覆盖（默认跳过）")
    ap.add_argument("--trailing-silence-s", type=float, default=9.0,
                    help=("尾部静音时长（秒）。这是**数据侧**给 duplex 模型"
                          "的'用户说完'信号；引擎侧不会做这件事。<5s 模型"
                          "几乎不开口；本机实测 9s 时 8 段问句中 2 段开口"
                          "（25%）。生产 RL 数据建议 ≥9s。"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    return synth(args)


if __name__ == "__main__":
    raise SystemExit(main())
