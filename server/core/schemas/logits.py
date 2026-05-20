"""Schemas for LLM logits export (RL training data collection).

The non-streaming inference server can optionally record, for every position
the LLM main backbone evaluates (prefill *and* decode), the tuple

    (token_id_i32, logits_vocab_size_bf16)

and return it to the caller either inline (base64) or as a ``.safetensors``
file path. Modality (audio/image embedding) positions are recorded with
placeholder token ids that never collide with real vocabulary:

- ``-1``  generic modality (default fallback used by the prefill helpers)
- ``-2``  audio embedding
- ``-3``  image embedding

Note these are **negative** values, so int32 reinterpretation handles them
naturally and they cannot be confused with real Qwen3 vocab ids (which are
all non-negative and bounded by ``vocab_size``).

Schemas:

- ``LogitsExportSpec`` — request-side, plugged into ``ChatRequest`` and
  ``DuplexBatchRequest`` under the ``logits`` field
- ``LogitsPayload`` — response-side, attached to ``ChatResponse`` and
  ``DuplexBatchResponse`` under the ``logits`` field (nullable)

See ``docs/API.md`` -> "Logits export for RL training" for end-to-end usage.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Placeholder token ids (mirrors omni.h OMNI_LOGIT_PLACEHOLDER_*)
# =============================================================================

PLACEHOLDER_MODALITY = -1   # generic modality (audio/image, prefill helpers don't know)
PLACEHOLDER_AUDIO    = -2
PLACEHOLDER_IMAGE    = -3


# =============================================================================
# Request: LogitsExportSpec
# =============================================================================

class LogitsExportSpec(BaseModel):
    """Per-request opt-in for logits capture.

    Set ``enabled=True`` to make the LLM main backbone record every
    ``(token_id, logits[bf16])`` pair during prefill + decode. The server
    then either embeds them inline in the JSON response or writes a single
    ``.safetensors`` file (see ``format``).

    For RL/RLHF data collection prefer ``format="file"`` — a single
    ``decode_tokens × 152064 × 2 bytes`` blob will saturate JSON quickly.
    """

    enabled: bool = Field(
        default=False,
        description="If true, request the server to record logits for this call.",
    )
    format: Literal["inline", "file"] = Field(
        default="file",
        description=(
            "How logits come back. 'file' writes a .safetensors next to the "
            "running server and returns its path in the response; 'inline' "
            "returns base64-encoded bf16 bytes inside the JSON response. "
            "Default 'file' since the inline blob is large (~10MB / 32 tokens)."
        ),
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Where the server should write the .safetensors when format='file'. "
            "Must be a path the *server* (and llama-server) can write to. "
            "If null, the gateway/worker layer picks a sensible default."
        ),
    )
    include_prefill: bool = Field(
        default=True,
        description=(
            "Reserved for future use; currently always True at the C++ layer. "
            "Once set in the spec it lets callers opt out of LM-head matmul on "
            "the prefill path (recovers ~30-50% prefill throughput at the cost "
            "of losing context-side logits)."
        ),
    )


# =============================================================================
# Response: LogitsPayload
# =============================================================================

class LogitsPayload(BaseModel):
    """Logits returned to the client.

    Exactly one of (``logits_b64`` + ``token_ids_b64``) or ``file`` is set,
    depending on the request's ``logits.format``.

    Decode the inline form as follows::

        import base64, numpy as np
        token_ids = np.frombuffer(
            base64.b64decode(payload.token_ids_b64), dtype=np.int32
        )                                          # shape [n_tokens]
        logits = np.frombuffer(
            base64.b64decode(payload.logits_b64), dtype=np.uint16
        ).reshape(payload.n_tokens, payload.vocab_size)
        # bf16 -> fp32: float32_view = (logits.astype(np.uint32) << 16).view(np.float32)

    The file form is a standard safetensors file with two tensors:

        - token_ids: I32  [N]
        - logits:    BF16 [N, vocab_size]

    plus a metadata block containing at least ``n_prefill_tokens`` /
    ``vocab_size`` / ``format``, and whatever extra fields the worker
    decided to attach (e.g. ``chunk_boundaries`` for duplex requests).
    """

    success: bool = Field(default=True, description="False if export failed.")
    error: Optional[str] = Field(default=None, description="Failure reason if success=False.")

    n_tokens: int = Field(..., description="Total number of captured positions (prefill + decode).")
    n_prefill_tokens: int = Field(..., description="How many of the leading positions are prefill.")
    vocab_size: int = Field(..., description="Length of each logits row (152064 for Qwen3).")
    dtype: str = Field(default="bf16", description="Always 'bf16' in current implementation.")

    # ---- inline format ----
    token_ids_b64: Optional[str] = Field(
        default=None,
        description="Base64 of int32 little-endian token ids (length n_tokens). Inline mode only.",
    )
    logits_b64: Optional[str] = Field(
        default=None,
        description="Base64 of bf16 logits bytes (length n_tokens * vocab_size * 2). Inline mode only.",
    )

    # ---- file format ----
    file: Optional[str] = Field(
        default=None,
        description="Absolute path to a .safetensors file written by the server. File mode only.",
    )
    sha256: Optional[str] = Field(
        default=None,
        description="(Optional) sha256 hex of the .safetensors content for end-to-end integrity checks.",
    )
    extra_metadata: Optional[dict] = Field(
        default=None,
        description="Echo of any business-side metadata embedded in the safetensors header.",
    )
