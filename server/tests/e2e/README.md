# End-to-end smoke tests

These tests assume:

- `llama-server` is built and available at `${LLAMA_CPP_OMNI_ROOT}/build/bin/llama-server`
- GGUF weights are present at `${MODEL_DIR}`
- One free GPU is visible

They are **skipped by default**. Opt in with:

```bash
pytest tests/e2e --run-e2e
```

Required environment variables:

```
LLAMA_CPP_OMNI_ROOT   = absolute path to this repo root
MODEL_DIR             = absolute path to MiniCPM-o-4_5 GGUF dir
LLM_MODEL             = e.g. "MiniCPM-o-4_5-Q8_0.gguf"  (optional)
TEST_AUDIO_WAV        = absolute path to a small 16kHz mono test wav
CUDA_VISIBLE_DEVICES  = e.g. "0"
```

What each test does:

| Test | What it covers |
|---|---|
| `test_chat_smoke` | Start 1 worker + batch_server, send a single `/v1/chat` request, assert non-empty text reply and (optionally) audio. |
| `test_duplex_offline_smoke` | Send a real audio file to `/v1/duplex_offline`, assert `success=True` and at least one speak chunk. |
