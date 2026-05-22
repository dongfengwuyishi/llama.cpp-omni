# MiniCPM-o Batch Inference Server — API Reference

非流式批量推理服务。一次请求对应一次完整响应，**没有 SSE / WebSocket / 长连接**，
适合评测、刷数据、离线 QA、Regression 等场景。

> 启动方式见 [`README.md`](../README.md)。本文档只讲 API。

---

## 总览

```
base = http://<host>:<port>           # 默认 :8080，可通过 config.json 改

POST  ${base}/v1/chat                 # 单工：完整 prompt → 完整 reply
POST  ${base}/v1/duplex_offline       # 双工非流式：完整音视频 → 聚合时间线

GET   ${base}/v1/health               # 集群健康
GET   ${base}/v1/workers              # worker 列表
GET   ${base}/v1/queue                # FIFO 队列状态
GET   ${base}/v1/status               # 聚合状态（busy/idle/queue/...）

GET   ${base}/health                  # 等价 /v1/health（兼容路径）
```

特性：

- **同步阻塞**：连接保持直到推理完成或超时
- **FIFO 排队**：worker 全忙时自动入队（默认 `max_queue_size = 1000`）
- **多 worker 并发**：N 个 GPU → N 路并发，没有 batching
- **超时**：默认 `request_timeout = 300s`（chat）/ `duplex_offline_timeout = 600s`
- **客户端断开自动取消**：未分配前断开 → 自动从队列移除

---

## 1. `POST /v1/chat` — 单工（turn-based）

输入完整 prompt（含可选 image/audio/video content），输出完整文本 + 可选 24kHz WAV 音频。
**KV cache 不复用**（每次从头 prefill）。

### Request

`Content-Type: application/json`，schema 见 [`core/schemas/chat.py:ChatRequest`](../core/schemas/chat.py)。

```jsonc
{
  "messages": [
    // user/system/assistant 都支持；content 可以是字符串或 ContentItem 数组
    { "role": "system", "content": "你是一个简洁的助手。" },
    {
      "role": "user",
      "content": [
        { "type": "text",  "text": "描述这张图片" },
        { "type": "image", "data": "<base64 jpg/png>" },
        { "type": "audio", "data": "<base64 16kHz mono float32>" }
        // 视频：{ "type": "video", "data": "<base64>", "stack_frames": 4 }
      ]
    }
  ],

  // 生成参数（可选；默认见 GenerationConfig）
  // 标 † 的字段 per-request 透传到 C++ ctx_sampler 重建（rebuild on every chat / half-duplex
  // turn）；标 ‡ 的字段是 server 自身处理（不进 sampler）。
  "generation": {
    "max_new_tokens":            256,    // ‡ 单轮硬上限，透传到 ctx_omni->chat_max_new_tokens
    "do_sample":                 true,   // † do_sample=false ⇒ 透传 temp=0 走 greedy
    "temperature":               0.7,    // † 仅在 do_sample=true 时生效
    "top_p":                     0.8,    // † 透传到 sampler.top_p
    "top_k":                     100,    // † 0 = HF 语义"禁用"，会在 _sampling_from_generation 端被丢弃
    "seed":                      42,     // † uint32；显式传 = trajectory 可重放（RL rollout 推荐）
    "repetition_penalty":        1.05,   // † 也接 ``repeat_penalty`` 别名
    "repetition_penalty_last_n": 64,     // † -1 = context_size，0 = 关闭
    "length_penalty":            1.1     // ‡ EOS bias，server 端在 stream/decode 处理
  },

  // TTS 输出（可选）
  "tts": {
    "enabled":         true,
    "mode":            "audio_assistant",   // 必须小写；其它合法值：'default' | 'omni' | 'audio_roleplay' | 'voice_cloning'。
                                            // 注意：mode="default" 会忽略 ref_audio_path
    "ref_audio_path":  "/abs/path/to/voice.wav",  // 可选；不填走默认音色
    "sampling":        { "temperature": 0.8 }
  },

  // 图像处理（可选）
  "image": {
    "max_slice_nums": 1
  },

  // 其它（可选）
  "use_tts_template": false,    // 输入含音频时自动启用
  "omni_mode":        false,    // 视频帧 + 音频拼接
  "enable_thinking":  false,    // 思考模式（<think>...</think>）
  "return_prompt":    false     // 调试：把组装后的 prompt 也返回
}
```

### Response 200 — `ChatResponse`

```jsonc
{
  "text":               "图中是一只柯基犬...",
  "audio_data":         "<base64 24kHz float32>",   // tts.enabled=true 时
  "audio_sample_rate":  24000,
  "tokens_generated":   42,
  "duration_ms":        2310.4,
  "token_stats": {
    "input_tokens":     128,
    "generated_tokens": 42,
    "total_tokens":     170,
    "cached_tokens":    0
  },
  "success":            true,
  "error":              null,

  // server 添加的追踪字段：
  "queue_wait_ms":           12,
  "estimated_queue_wait_s":  0.0,
  "ticket_id":               "q_a1b2c3d4e5f6",
  "worker_id":               "worker_0"
}
```

### cURL 示例（纯文本）

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "generation": {"max_new_tokens": 64}
  }'
```

### Python 示例（带图像）

```python
import base64, httpx
with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = httpx.post(
    "http://localhost:8080/v1/chat",
    json={
        "messages": [
            {"role": "user", "content": [
                {"type": "text",  "text": "描述图片"},
                {"type": "image", "data": img_b64},
            ]}
        ],
        "tts": {"enabled": False},
    },
    timeout=300,
)
print(resp.json()["text"])
```

---

## 2. `POST /v1/duplex_offline` — 双工非流式

把**完整音视频段**发给 server，server 内部按 `config.chunk_ms`（默认 1000ms）切片，
顺序 `prefill → generate → finalize`，把每个 chunk 的 listen/speak 时间线聚合后**一次性返回**。

不流式输出。调用方阻塞等结果。

### Request

Schema：[`core/schemas/duplex_batch.py:DuplexBatchRequest`](../core/schemas/duplex_batch.py)。

```jsonc
{
  // ---- system prompt ----
  "system_prompt": "你是一个友好的语音助手，简短回复。",

  // ---- 用户输入音频（path / base64 二选一） ----
  "user_audio_path":   "/data/eval/case_001.wav",
  "user_audio_base64": null,           // 16kHz mono float32 raw 或 wav 容器

  // ---- 视觉输入（可选，per-chunk 对齐） ----
  "image_paths":       null,           // 每个 chunk 对应一张
  "image_base64_list": null,
  "max_slice_nums":    1,

  // ---- TTS 音色 ----
  "ref_audio_path":   null,            // 不填走 system_prompt 语种默认音色
  "ref_audio_base64": null,
  "prompt_wav_path":  null,

  // ---- 双工模型参数 ----
  "config": {
    "force_listen_count":             3,    // 启动保护期（前 N 秒强制 listen）
    "max_new_speak_tokens_per_chunk": 20,
    "chunk_ms":                       1000,
    "temperature":                    0.7,
    "top_k":                          20,
    "top_p":                          0.8,
    "length_penalty":                 1.1,
    "decode_mode":                    "sampling"
    // 其它字段见 DuplexConfig
  },

  // ---- 批处理控制 ----
  "stop_on_end_of_turn": false,        // true=遇 end_of_turn 立刻停；false=跑完所有音频
  "max_chunks":          null,         // 兜底上限
  "leading_silence_ms":  0,            // 在用户音频前插入静音 (ms)

  // ---- 输出控制 ----
  "return_per_chunk_audio": true,      // chunks[].audio_data 是否返回
  "return_merged_audio":    true,      // 是否返回拼接的完整 24kHz 音频
  "include_text_timeline":  true,      // 是否返回 chunks 数组

  // ---- 追踪 ----
  "request_id": "case_001"
}
```

### Response 200 — `DuplexBatchResponse`

```jsonc
{
  "success":     true,
  "error":       null,

  "full_text":   "你好，我听到你说...",  // 所有 speak chunk 文本拼接

  "chunks": [
    { "chunk_idx": 0, "phase": "user",     "is_listen": true,  "text": "",  "has_audio": false, "audio_data": null, "end_of_turn": false, "elapsed_ms": 14.2 },
    { "chunk_idx": 3, "phase": "response", "is_listen": false, "text": "你好", "has_audio": true, "audio_data": "<base64>", "end_of_turn": false, "elapsed_ms": 1840.0 }
    // ...
  ],

  "merged_audio_data":        "<base64 24kHz float32 整段>",
  "merged_audio_sample_rate": 24000,

  "total_chunks":      18,
  "speak_chunks":      9,
  "listen_chunks":     9,
  "audio_duration_s":  12.3,
  "total_duration_ms": 28490.6,
  "stopped_reason":    "audio_exhausted",  // audio_exhausted | end_of_turn | max_chunks | error

  // server 添加的追踪字段：
  "request_id":              "case_001",
  "ticket_id":               "q_b2c3d4e5f6g7",
  "worker_id":               "worker_1",
  "queue_wait_ms":           42,
  "estimated_queue_wait_s":  0.0
}
```

### cURL 示例

```bash
curl -X POST http://localhost:8080/v1/duplex_offline \
  -H "Content-Type: application/json" \
  -d '{
    "system_prompt":   "请用一句话回答用户。",
    "user_audio_path": "/data/eval/q1.wav",
    "config": {"force_listen_count": 3, "chunk_ms": 1000},
    "stop_on_end_of_turn": true,
    "return_per_chunk_audio": false,
    "request_id": "case_001"
  }'
```

### Python 示例（用 base64 上传音频）

```python
import base64, httpx
with open("question.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

resp = httpx.post(
    "http://localhost:8080/v1/duplex_offline",
    json={
        "system_prompt":     "请简短回答用户。",
        "user_audio_base64": audio_b64,
        "config":            {"force_listen_count": 3},
        "stop_on_end_of_turn": True,
        "return_merged_audio": True,
        "return_per_chunk_audio": False,
        "request_id": "case_001",
    },
    timeout=600,
)
out = resp.json()
print(out["full_text"], "took", out["total_duration_ms"], "ms")

# 把合成音频落盘
import numpy as np, soundfile as sf
arr = np.frombuffer(base64.b64decode(out["merged_audio_data"]), dtype=np.float32)
sf.write("reply.wav", arr, 24000)
```

### 字段语义速查（容易踩坑）

| 字段 | 含义 | 默认 | 建议 |
|---|---|---|---|
| `config.chunk_ms` | 切片粒度 | 1000 | 别改：模型训练的就是 1s/chunk |
| `config.force_listen_count` | 前 N 个 chunk 强制 listen，给用户说话留窗口 | 3 | 短音频可设 1~2；长视频保持 3 |
| `config.max_new_speak_tokens_per_chunk` | 单 chunk 最多吐多少 token | 20 | 减少会更"碎"；加大可能爆 KV |
| `stop_on_end_of_turn` | 模型说完一轮就停 | false | 短问答评测设 true；长视频/连续对话设 false |
| `max_chunks` | 强制上限（兜底） | null | 防止视频特别长 / 模型异常时长跑 |
| `leading_silence_ms` | 用户音频前补静音 | 0 | 启动期不稳时可加 500-1000 |
| `return_per_chunk_audio` | 单 chunk 音频是否返回 | true | 大批量评测可关闭，节省 50% 响应体大小 |
| `return_merged_audio` | 整段合成音频是否返回 | true | 通常保留 |

---

## 3. 集群运维接口

### `GET /v1/health`

```jsonc
{
  "status":           "ok",
  "ready":            true,
  "workers_total":    2,
  "workers_idle":     1,
  "workers_busy":     1,
  "workers_loading":  0,
  "workers_error":    0,
  "queue_length":     0,
  "max_queue_size":   1000
}
```

### `GET /v1/workers`

```jsonc
{
  "total": 2,
  "workers": [
    {
      "worker_id":        "worker_0",
      "host":             "localhost",
      "port":             22440,
      "gpu_id":           0,
      "status":           "duplex_active",
      "current_session_id": "20260519_104233_batch_case_001",
      "total_requests":   12,
      "avg_inference_time_ms": 24500.5,
      "current_request_type":  "audio_duplex",
      "task_started_at":  "2026-05-19T10:42:33.108Z"
    }
  ]
}
```

### `GET /v1/queue`

```jsonc
{
  "queue_length":   3,
  "max_queue_size": 1000,
  "items": [
    { "ticket_id": "q_aaa", "request_type": "audio_duplex", "position": 1, "estimated_wait_s": 25.0, "enqueued_at": "...", "wait_elapsed_s": 3.2 },
    { "ticket_id": "q_bbb", "request_type": "chat",         "position": 2, "estimated_wait_s": 26.0, "enqueued_at": "...", "wait_elapsed_s": 1.1 }
  ],
  "running_tasks": [
    { "worker_id": "worker_0", "request_type": "audio_duplex", "started_at": "...", "elapsed_s": 5.4, "estimated_remaining_s": 119.6 }
  ]
}
```

### `GET /v1/status`

`/v1/health` + `/v1/queue` 的合并视图，适合做仪表盘。

---

## 4. 错误码

| HTTP | 含义 | 典型触发条件 |
|---|---|---|
| 400 | Bad Request | 请求体字段不合法（pydantic 校验失败） |
| 429 | Too Many Requests | 极少触发；worker 上一个任务还在 cleanup |
| 499 | Client Closed Request | client 在排队中主动断开 |
| 500 | Internal Server Error | worker 内部异常（model OOM / TTS 失败 / 音频解码失败等） |
| 502 | Bad Gateway | worker 返回非 200 但有 body |
| 503 | Service Unavailable | 服务未就绪 / FIFO 队列已满（`Queue full ...`） |
| 504 | Gateway Timeout | worker 在 `request_timeout` 内未返回 |

错误 body 统一格式：

```jsonc
{ "detail": "<error message>" }
```

---

## 5. 客户端代码模板

### 异步并发评测（节选自 [`scripts/eval_runner.py`](../scripts/eval_runner.py)）

```python
import asyncio, httpx, json, base64

async def run_one(client, case):
    r = await client.post("http://localhost:8080/v1/duplex_offline", json=case, timeout=600)
    return r.json()

async def main(jsonl_path, concurrency=4):
    cases = [json.loads(l) for l in open(jsonl_path)]
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        async def bounded(c):
            async with sem:
                return await run_one(client, c)
        results = await asyncio.gather(*(bounded(c) for c in cases))
    for r in results:
        print(r["request_id"], r["success"], r.get("full_text", "")[:60])

asyncio.run(main("eval.jsonl", concurrency=4))
```

### 直接调用脚本

```bash
python scripts/eval_runner.py \
    --endpoint    http://localhost:8080 \
    --input-list  eval_set.jsonl \
    --output      results.jsonl \
    --audio-dir   results_audio/ \
    --concurrency 4 \
    --timeout     600
```

`eval_set.jsonl` 行格式：

```jsonl
{"id": "c01", "task": "chat",   "messages": [{"role": "user", "content": "1+1=?"}]}
{"id": "d01", "task": "duplex", "system_prompt": "短回答", "user_audio_path": "/data/d01.wav"}
```

---

## 6. 容量 / 性能直觉

- **并发上限 = worker 数量**。一个 worker 同一时间只跑一个请求（模型不支持 batch）
- **超额请求自动入 FIFO**，按 enqueue 顺序服务。`/v1/queue` 可查实时位置和 ETA
- **`/v1/duplex_offline` 单条耗时**约等于 `音视频时长 × 1.2~3.0`，取决于 GGUF 量化和 GPU
- **HTTP 请求体上限**：FastAPI 默认无硬限制，但建议优先用 `*_path` 字段而非 base64
  （30s @ 16kHz mono float32 base64 后约 2.6MB；视频 base64 化容易上百 MB）
- **超时**：
  - `/v1/chat` → `service.request_timeout`（默认 300s）
  - `/v1/duplex_offline` → `--duplex-offline-timeout`（默认 600s）

---

## 7. 测试 / 验证

集成的自动化测试见 [`tests/`](../tests/)：

```bash
cd server

# 单元 + 集成测试（≤15s，无 GPU 无模型）
PYTHONPATH=. .venv/bin/pytest tests/unit tests/integration

# 端到端冒烟（需要 GPU + GGUF + 设置环境变量）
LLAMA_CPP_OMNI_ROOT=/abs/path/to/llama.cpp-omni \
MODEL_DIR=/abs/path/to/MiniCPM-o-4_5-gguf \
TEST_AUDIO_WAV=/abs/path/to/test.wav \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. .venv/bin/pytest tests/e2e --run-e2e -s
```

测试分层：

| 目录 | 数量 | 依赖 | 用途 |
|---|---|---|---|
| `tests/unit/`        | 36 | 纯 pydantic / dataclass | schema / config / worker_pool 行为 |
| `tests/integration/` | 20 | mock worker（无 GPU） | batch_server HTTP 路由 / 队列 / ETA |
| `tests/e2e/`         |  2 | 真实 worker + GGUF + GPU | chat / duplex_offline 冒烟 |

---

## 8. Text-only mode (skip TTS / Token2Wav)

For RL training rollouts / text-only eval where audio synthesis is dead
weight, start the worker with `--no-tts` (or `NO_TTS=1 bash start_all.sh`).
The C++ worker is then booted with `use_tts=False`:

- TTS / Token2Wav weights are **not** loaded (model boot drops from ~30-60s
  to ~7s)
- TTS / T2W threads are **not** spawned
- LLM main backbone runs without TTS contention

### Performance (1× RTX 4090, MiniCPM-o-4_5 Q8_0, 7-chunk duplex_offline)

| | with TTS | `NO_TTS=1` | Δ |
|---|---|---|---|
| `total_duration_ms` (server) | 31.3 s | **14.0 s** | **−55%** |
| client wall-clock | 49.5 s | **30.3 s** | **−39%** |
| model load time | ~30 s | ~7 s | −77% |

### Config

```jsonc
"cpp_backend": {
  ...
  "use_tts": false        // optional; default true. Overridden by CLI --no-tts.
}
```

### Known limitations (text-only mode)

1. Audio output fields (`audio_data`, `merged_audio_data`) are always `null`
   in `NO_TTS=1` mode; setting `tts.enabled=true` in the request has no
   effect — TTS / Token2Wav weights aren't loaded so audio simply can't
   be synthesized.

> Earlier revisions of this branch listed two other limitations: `/v1/chat`
> returning empty text under `NO_TTS=1`, and `logits` capture only covering
> ~5% of expected positions. Both shared the same root cause: the Python
> chat path was sending the first user message at index=0, which the C++
> ``stream_prefill`` reserves for system-prompt initialization (see the
> contract comment in ``tools/omni/omni.cpp`` near ``stream_prefill``);
> user content at index=0 was silently dropped, so the LLM either decoded
> nothing (use_tts=false) or hallucinated an unrelated reply forced by
> ``<|tts_bos|>`` (use_tts=true). The chat path now emits a dedicated
> index=0 init prefill and starts user content at index=1, which fixes
> both the empty-text and the partial-logits behavior in NO_TTS mode.

---

## 9. Logits export for RL training

设置 `logits.enabled=true` 可以让 server 把 LLM **主干**在每个位置的 next-token
分布捕获下来，配合 sampled token id 一起返回。**双工 / 单工都支持**。

> 仅 LLM 主干，不含 TTS codec head。设计文档：
> [`core/schemas/logits.py`](../core/schemas/logits.py) /
> [`tools/omni/logit_capture.cpp`](../../tools/omni/logit_capture.cpp)。

### 8.1 请求字段

`/v1/chat` 和 `/v1/duplex_offline` 都接受新的 `logits` 字段
（[`LogitsExportSpec`](../core/schemas/logits.py)）：

```jsonc
"logits": {
  "enabled":    true,
  "format":     "file",          // "file" (默认) 或 "inline"
  "output_dir": "/data/logits",  // file 模式必填；server 写盘到这里
  "include_prefill": true         // 保留字段，目前 prefill 都会被捕获
}
```

- **`format="file"`**：server 写一个 `.safetensors`，响应里只返回路径
- **`format="inline"`**：响应 JSON 里直接 base64 嵌入 token_ids + logits 字节

> **落盘路径与保留策略（`format="file"`）**
>
> server 把 `output_dir`（或环境变量 `OMNI_LOGITS_OUTPUT_DIR`，默认
> `/tmp/minicpm_logits`）当**基目录**，自动在它下面追加一层 UTC 日期子目录
> `YYYY-MM-DD/`，叶子文件名走如下统一模板（chat / duplex 共用）：
>
> ```
> {kind}_w{worker_idx}_p{pid_hex5}_{seq:08d}[_{client_request_id}].safetensors
> ```
>
> 字段语义：
>
> - `kind` ∈ `{chat, duplex}` —— 请求类型，事后批量 `grep` 友好
> - `w{worker_idx}` —— worker 进程在 batch_server pool 里的 0-based 索引
>   （由 `worker.py --worker-index` 注入），跨 worker 进程必然唯一
> - `p{pid_hex5}` —— worker 进程 PID 的低 20 bits（5 位 hex）。仅用于防
>   "同一 worker_idx + 同一日期 bucket + worker 重启 + seq 复用" 的极端撞名
> - `seq:08d` —— 进程内 atomic 单调计数器，从 0 开始
> - `_{client_request_id}` —— 可选 debug 后缀。client 给的 `request_id` 经
>   `[A-Za-z0-9_]+` sanitize、截断到 32 字符。**不参与唯一性**，client 给重了
>   也不会撞文件（前面的 `(worker_idx, pid, seq)` 三元组已经唯一）
>
> 最终落盘路径示例：
>
> ```
> /data/logits/2026-05-21/chat_w0_p1f4a_00000123.safetensors
> /data/logits/2026-05-21/chat_w0_p1f4a_00000124_e2e_001.safetensors
> /data/logits/2026-05-21/duplex_w2_p3b81_00000456_dup_42.safetensors
> /data/logits/2026-05-22/...
> ```
>
> 为什么这样命名 —— 修复一个并发缺陷：早期版本 chat 路径硬编码
> `chat_round{N}.safetensors`，且 `chat()` 入口处把内部 round 计数重置为 0，
> 导致**所有** chat 请求都写到同一个 `chat_round0.safetensors`，单 worker
> 内自相覆盖、跨 worker 共享日期 bucket 又互相覆盖。在 4 worker × 30 并发的
> 压测里 100+ 个 logits_file 调用全部 success，盘上只剩 1 个文件。
>
> Worker 进程内置一个 daemon janitor，按下面的环境变量做老化清理（默认开启）：
>
> | 变量 | 默认 | 含义 |
> |---|---|---|
> | `OMNI_LOGITS_OUTPUT_DIR`        | `/tmp/minicpm_logits` | 基目录（请求未指定 `output_dir` 时） |
> | `OMNI_LOGITS_RETENTION_DAYS`    | `7`   | 严格早于 `(today - N 天)` 的日期目录整体删除；`0` 关闭 |
> | `OMNI_LOGITS_MAX_TOTAL_BYTES`   | `0`   | 超过则按"最旧日期目录优先"逐日驱逐；`0` 关闭。永不删当天 |
> | `OMNI_LOGITS_CLEANUP_INTERVAL_S`| `600` | 扫描间隔秒，最小 60 |
>
> 多 worker 共享同一个基目录是安全的（`unlink/rmtree` 幂等，且文件名按上面
> 模板互斥）。这是 server-side 行为，client 不需要改 schema —— 老的
> `output_dir=/data/logits` 调用方拿到的 `logits.file` 字段就会变成
> `/data/logits/2026-05-21/chat_w<idx>_p<pid>_<seq>.safetensors`。

### 8.2 响应字段

`ChatResponse.logits` / `DuplexBatchResponse.logits`
（[`LogitsPayload`](../core/schemas/logits.py)，捕获关闭时为 null）：

```jsonc
"logits": {
  "success":           true,
  "n_tokens":          289,
  "n_prefill_tokens":  217,
  "vocab_size":        151748,
  "dtype":             "bf16",
  "file":              "/data/logits/2026-05-21/chat_w0_p1f4a_00000123.safetensors",  // file 模式
  // inline 模式：
  "token_ids_b64":     "<base64 int32 bytes>",
  "logits_b64":        "<base64 bf16 bytes>",
  "extra_metadata":    { "chunk_boundaries": [0, 207, 390, ..., 1139] }  // duplex 专有
}
```

- `n_tokens = n_prefill_tokens + n_decode_tokens`，前 N 个属于 prefill
- `dtype="bf16"`：每个 logit 占 2 bytes，按 `token_ids[i]` 的顺序对齐
- modality（音频/图像 embedding）位置的 `token_id` 是占位符：
  - `-1` 通用 modality（默认）
  - `-2` 音频
  - `-3` 图像

### 8.3 落盘格式（safetensors v1）

```
+--- Header ----------------------------------------------------+
| u64 LE header_size                                            |
| JSON header (UTF-8, padded to 8-byte alignment):              |
|   {                                                           |
|     "token_ids":  {dtype:"I32",  shape:[N],   data_offsets:[..]},
|     "logits":     {dtype:"BF16", shape:[N,V], data_offsets:[..]},
|     "__metadata__": {                                         |
|       "format":            "minicpm-o-omni-logits/v1",        |
|       "n_prefill_tokens":  "...",                             |
|       "vocab_size":        "151748",                          |
|       "n_tokens":          "...",                             |
|       "chunk_boundaries":  "[0, 207, 390, ...]",  // duplex   |
|       "chunk_prefill_counts": "[117, 105, 93, ...]",  // duplex
|       "request_id":        "..."                              |
|     }                                                         |
|   }                                                           |
+--- Body ------------------------------------------------------+
| raw int32 token_ids   (N * 4 bytes)                           |
| raw bf16 logits       (N * V * 2 bytes)                       |
+---------------------------------------------------------------+
```

读回示例（Python，无依赖）：

```python
import struct, json
import numpy as np

with open("chat_w0_p1f4a_00000123.safetensors", "rb") as f:
    header_size = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(header_size))
    body_off = 8 + header_size

    tspec = header["token_ids"]; lspec = header["logits"]
    f.seek(body_off + tspec["data_offsets"][0])
    token_ids = np.frombuffer(f.read(tspec["data_offsets"][1] - tspec["data_offsets"][0]), dtype=np.int32)
    f.seek(body_off + lspec["data_offsets"][0])
    raw = f.read(lspec["data_offsets"][1] - lspec["data_offsets"][0])
    logits_bf16 = np.frombuffer(raw, dtype=np.uint16).reshape(lspec["shape"])

# bf16 → fp32（位扩展）
logits_fp32 = (logits_bf16.astype(np.uint32) << 16).view(np.float32)

# 元信息（safetensors 规范：metadata 全是字符串，需要自行反序列化）
n_prefill = int(header["__metadata__"]["n_prefill_tokens"])
chunk_bounds = json.loads(header["__metadata__"].get("chunk_boundaries", "[]"))

# 切回 prefill / decode
prefill_tok = token_ids[:n_prefill]
decode_tok  = token_ids[n_prefill:]
prefill_logits = logits_fp32[:n_prefill]
decode_logits  = logits_fp32[n_prefill:]
```

### 8.4 性能影响

- **prefill** 慢约 30~50%（开了 logits 后每个位置都要跑 LM head matmul）
- **decode** 几乎无影响（本来就要算 logits）
- **响应体积**：~`N × 152K × 2 bytes`。30 tokens 大约 9 MB，建议默认走 `format="file"`
- **磁盘 IO**：file 模式下，~330 MB/s 顺序写，瓶颈通常不在这里

### 8.5 cURL 示例

```bash
curl -X POST http://localhost:8080/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages":   [{"role": "user", "content": "你好"}],
    "generation": {"max_new_tokens": 32, "do_sample": false},
    "tts":        {"enabled": false},
    "logits":     {"enabled": true, "format": "file", "output_dir": "/data/logits"}
  }'
```

返回的 `logits.file` 字段指向一个可读的 safetensors 文件。

### 8.6 验证

E2E 验证脚本：[`scripts/smoke_logits.py`](../scripts/smoke_logits.py)。
跑通后会输出三种模式（chat-inline / chat-file / duplex-file）的 PASS/FAIL 表。

---

## 10. 字段总览参考

| 文件 | 关键 schema |
|---|---|
| `core/schemas/common.py`       | `Message` / `Role` / `ContentItem` / `TTSConfig` / `GenerationConfig` |
| `core/schemas/chat.py`         | `ChatRequest` / `ChatResponse` |
| `core/schemas/duplex.py`       | `DuplexConfig` / `DuplexGenerateResult` / `DuplexChunkResult` |
| `core/schemas/duplex_batch.py` | **`DuplexBatchRequest` / `DuplexBatchResponse`** |
| `gateway_modules/models.py`    | `QueueStatus` / `WorkersResponse` / `EtaConfig` |
