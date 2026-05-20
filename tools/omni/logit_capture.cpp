// LLM logit capture for RL training.
//
// This file implements the C API declared in omni.h:
//
//     omni_logit_capture_set_enabled
//     omni_logit_capture_is_enabled
//     omni_logit_capture_reset
//     omni_logit_capture_append
//     omni_logit_capture_get
//     omni_logit_capture_export_safetensors
//
// Plus a minimal self-contained safetensors writer (no external deps).
//
// Capture is enabled per session via update_session_config; once enabled,
// stream_prefill and stream_decode call ``omni_logit_capture_append`` for
// every token position they push through the LLM (real tokens *and*
// modality embedding placeholders). The buffer accumulates within a single
// "round" and is dumped / reset by the caller between rounds.
//
// Storage layout per token i: token_ids[i] (int32), logits_bf16[i*V .. (i+1)*V).
// bf16 is the high 16 bits of float32 (truncation rounding) — common for
// neural-net storage and round-trip safe enough for RL importance sampling.

#include "omni.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>


// ----------------------------------------------------------------------------
// fp32 → bf16 (truncation; RNE adds a few ULP of accuracy but matters little
// for downstream training loss reconstruction)
// ----------------------------------------------------------------------------

static inline uint16_t omni_fp32_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    // NaN: ensure non-zero mantissa is preserved in bf16
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u) {
        return static_cast<uint16_t>((bits >> 16) | 0x0040u);
    }
    return static_cast<uint16_t>(bits >> 16);
}


// ----------------------------------------------------------------------------
// Public C API — capture lifecycle
// ----------------------------------------------------------------------------

void omni_logit_capture_set_enabled(struct omni_context * ctx_omni, bool enabled, int32_t vocab_size) {
    if (!ctx_omni) return;
    auto & buf = ctx_omni->logit_buf;
    buf.enabled = enabled;
    if (enabled && vocab_size > 0) {
        buf.vocab_size = vocab_size;
    }
    if (!enabled) {
        // Drop memory eagerly to avoid lingering ~10MB buffers when capture
        // gets switched off mid-session.
        buf.token_ids.clear();
        buf.token_ids.shrink_to_fit();
        buf.logits_bf16.clear();
        buf.logits_bf16.shrink_to_fit();
        buf.n_prefill_tokens = 0;
    }
}

bool omni_logit_capture_is_enabled(const struct omni_context * ctx_omni) {
    return ctx_omni && ctx_omni->logit_buf.enabled;
}

void omni_logit_capture_reset(struct omni_context * ctx_omni) {
    if (!ctx_omni) return;
    auto & buf = ctx_omni->logit_buf;
    buf.token_ids.clear();
    buf.logits_bf16.clear();
    buf.n_prefill_tokens = 0;
    // Keep ``enabled`` / ``vocab_size`` so a new round can keep capturing.
}

void omni_logit_capture_append(struct omni_context * ctx_omni,
                                int32_t token_id, const float * logits_fp32,
                                bool is_prefill) {
    if (!ctx_omni || !ctx_omni->logit_buf.enabled || !logits_fp32) {
        return;
    }
    auto & buf = ctx_omni->logit_buf;
    if (buf.vocab_size <= 0) {
        // No vocab_size set; nothing we can do with the row.
        return;
    }
    const size_t V = static_cast<size_t>(buf.vocab_size);
    buf.token_ids.push_back(token_id);
    const size_t old_size = buf.logits_bf16.size();
    buf.logits_bf16.resize(old_size + V);
    uint16_t * dst = buf.logits_bf16.data() + old_size;
    for (size_t k = 0; k < V; ++k) {
        dst[k] = omni_fp32_to_bf16(logits_fp32[k]);
    }
    if (is_prefill) {
        buf.n_prefill_tokens++;
    }
}

void omni_logit_capture_prepare_batch(struct omni_context * ctx_omni,
                                       struct llama_batch * batch,
                                       omni_logit_capture_batch_handle * handle) {
    if (!handle) return;
    handle->installed = false;
    handle->prev_logits = nullptr;
    handle->mask.clear();
    if (!ctx_omni || !batch || !ctx_omni->logit_buf.enabled) return;
    if (batch->n_tokens <= 0) return;

    handle->mask.assign(static_cast<size_t>(batch->n_tokens), int8_t(1));
    handle->prev_logits = batch->logits;
    batch->logits = handle->mask.data();
    handle->installed = true;
}

void omni_logit_capture_drain_batch(struct omni_context * ctx_omni,
                                     const struct llama_batch * batch,
                                     int32_t modality_placeholder,
                                     bool is_prefill) {
    if (!ctx_omni || !batch || !ctx_omni->logit_buf.enabled) return;
    if (batch->n_tokens <= 0) return;

    // vocab_size may not have been set explicitly via set_enabled; lazily
    // discover it from the model so stream_prefill callers don't have to.
    if (ctx_omni->logit_buf.vocab_size <= 0 && ctx_omni->model != nullptr) {
        const llama_vocab * vocab = llama_model_get_vocab(ctx_omni->model);
        if (vocab != nullptr) {
            ctx_omni->logit_buf.vocab_size = llama_vocab_n_tokens(vocab);
        }
    }
    if (ctx_omni->logit_buf.vocab_size <= 0) return;

    for (int32_t j = 0; j < batch->n_tokens; ++j) {
        const float * row = llama_get_logits_ith(ctx_omni->ctx_llama, j);
        if (row == nullptr) continue;
        int32_t tok_id = modality_placeholder;
        if (batch->token != nullptr) {
            tok_id = static_cast<int32_t>(batch->token[j]);
        }
        omni_logit_capture_append(ctx_omni, tok_id, row, is_prefill);
    }
}

bool omni_logit_capture_get(const struct omni_context * ctx_omni,
                             const int32_t  ** out_token_ids,
                             int32_t         * out_n_tokens,
                             const uint16_t ** out_logits_bf16,
                             int32_t         * out_vocab_size,
                             int32_t         * out_n_prefill_tokens) {
    if (!ctx_omni) return false;
    const auto & buf = ctx_omni->logit_buf;
    if (!buf.enabled || buf.token_ids.empty() || buf.vocab_size <= 0) {
        return false;
    }
    if (out_token_ids)        *out_token_ids        = buf.token_ids.data();
    if (out_n_tokens)         *out_n_tokens         = static_cast<int32_t>(buf.token_ids.size());
    if (out_logits_bf16)      *out_logits_bf16      = buf.logits_bf16.data();
    if (out_vocab_size)       *out_vocab_size       = buf.vocab_size;
    if (out_n_prefill_tokens) *out_n_prefill_tokens = buf.n_prefill_tokens;
    return true;
}


// ----------------------------------------------------------------------------
// Minimal safetensors writer (no external dependencies)
// ----------------------------------------------------------------------------
//
// File layout (https://huggingface.co/docs/safetensors):
//
//   bytes  [0..8)   uint64 little-endian: size of the header JSON in bytes
//   bytes  [8..8+H) the header JSON itself, UTF-8
//   bytes  [8+H..)  concatenated tensor data, in the order declared by the
//                   header's "data_offsets" fields (which give [start, end)
//                   ranges relative to the start of this section)
//
// We hand-build the JSON because the project doesn't bundle a JSON writer
// and the schema we emit is tiny / static.

namespace {

// Append "field" to ``out`` with a JSON-escaped string value (handles "\"\\\n\t").
// We don't need full JSON robustness — metadata values come from controlled
// places (filenames, integer-string fields, callers passing tiny key=val).
void json_append_escaped(std::string & out, const std::string & s) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c) & 0xff);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

// Parse the **top-level** object body from ``json`` and append each "key":val
// entry (verbatim, no validation) into ``out``. Used to merge user-supplied
// metadata JSON object into our metadata block.
//
// This is intentionally permissive: we just strip the outer braces and trust
// the caller to provide a well-formed object. Empty / nullptr / non-object
// inputs are silently ignored.
void json_merge_object_body(std::string & out, const char * extra_json) {
    if (!extra_json) return;
    std::string s = extra_json;
    // strip whitespace
    size_t i = 0, j = s.size();
    while (i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\n' || s[j-1] == '\r')) --j;
    if (i >= j || s[i] != '{' || s[j-1] != '}') return;
    ++i; --j;
    while (i < j && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    while (j > i && (s[j-1] == ' ' || s[j-1] == '\t' || s[j-1] == '\n' || s[j-1] == '\r')) --j;
    if (i >= j) return;
    if (!out.empty() && out.back() != ',' && out.back() != '{') {
        out.push_back(',');
    }
    out.append(s, i, j - i);
}

} // namespace


bool omni_logit_capture_export_safetensors(const struct omni_context * ctx_omni,
                                            const char * path,
                                            const char * extra_metadata_json) {
    if (!ctx_omni || !path) return false;
    const auto & buf = ctx_omni->logit_buf;
    if (!buf.enabled || buf.token_ids.empty() || buf.vocab_size <= 0) {
        return false;
    }

    const int64_t N = static_cast<int64_t>(buf.token_ids.size());
    const int64_t V = static_cast<int64_t>(buf.vocab_size);

    const int64_t tokens_bytes = N * static_cast<int64_t>(sizeof(int32_t));
    const int64_t logits_bytes = N * V * static_cast<int64_t>(sizeof(uint16_t));

    if (static_cast<int64_t>(buf.logits_bf16.size()) != N * V) {
        std::fprintf(stderr,
                     "omni_logit_capture_export_safetensors: invariant violated, "
                     "logits_bf16.size=%zu but N*V=%" PRId64 "\n",
                     buf.logits_bf16.size(), N * V);
        return false;
    }

    // --- Build header JSON ---
    std::string header;
    header.reserve(1024);
    header.push_back('{');

    // tensor "token_ids" — I32 [N], offset [0, tokens_bytes)
    header += "\"token_ids\":{\"dtype\":\"I32\",\"shape\":[";
    {
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%" PRId64, N);
        header += tmp;
    }
    header += "],\"data_offsets\":[0,";
    {
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%" PRId64, tokens_bytes);
        header += tmp;
    }
    header += "]}";

    // tensor "logits" — BF16 [N, V], offset [tokens_bytes, tokens_bytes+logits_bytes)
    header += ",\"logits\":{\"dtype\":\"BF16\",\"shape\":[";
    {
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%" PRId64, N);
        header += tmp;
    }
    header += ",";
    {
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%" PRId64, V);
        header += tmp;
    }
    header += "],\"data_offsets\":[";
    {
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%" PRId64, tokens_bytes);
        header += tmp;
    }
    header += ",";
    {
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%" PRId64, tokens_bytes + logits_bytes);
        header += tmp;
    }
    header += "]}";

    // __metadata__ block (safetensors convention: all values must be strings)
    header += ",\"__metadata__\":{";
    header += "\"format\":";          json_append_escaped(header, "minicpm-o-omni-logits/v1");
    header += ",\"n_prefill_tokens\":"; json_append_escaped(header, std::to_string(buf.n_prefill_tokens));
    header += ",\"vocab_size\":";       json_append_escaped(header, std::to_string(buf.vocab_size));
    header += ",\"n_tokens\":";         json_append_escaped(header, std::to_string(N));
    // Merge caller-supplied metadata (keys / values must already be quoted JSON strings).
    json_merge_object_body(header, extra_metadata_json);
    header += "}";

    header += '}';

    // Pad the header to 8-byte alignment so the tensor body starts at an
    // aligned offset. The safetensors spec doesn't require it, but most
    // readers (incl. ours when mmap'd later) appreciate it.
    while ((8 + header.size()) % 8 != 0) {
        header.push_back(' ');
    }
    const uint64_t header_size = static_cast<uint64_t>(header.size());

    // --- Write file ---
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) {
        std::fprintf(stderr,
                     "omni_logit_capture_export_safetensors: cannot open '%s' for writing\n",
                     path);
        return false;
    }
    // Header size (u64 little-endian).
    uint8_t h[8];
    for (int i = 0; i < 8; ++i) h[i] = static_cast<uint8_t>((header_size >> (8 * i)) & 0xff);
    f.write(reinterpret_cast<const char *>(h), 8);
    // Header JSON.
    f.write(header.data(), header.size());
    // Tensor "token_ids" raw bytes.
    f.write(reinterpret_cast<const char *>(buf.token_ids.data()), tokens_bytes);
    // Tensor "logits" raw bytes (bf16 little-endian, native uint16_t order).
    f.write(reinterpret_cast<const char *>(buf.logits_bf16.data()), logits_bytes);
    if (!f.good()) {
        std::fprintf(stderr,
                     "omni_logit_capture_export_safetensors: write failed for '%s'\n",
                     path);
        return false;
    }
    f.close();
    return true;
}
