#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>

using json = nlohmann::ordered_json;

// ============================================================================
// Metrics (attached to downstream events)
// ============================================================================

struct ProtocolMetrics {
    std::string backend = "llama.cpp-omni";
    int kv_cache_length = 0;
    double prefill_ms = 0.0;
    double generate_ms = 0.0;
    double wall_clock_ms = 0.0;
    int n_tokens = 0;

    json to_json() const;
};

// ============================================================================
// Downstream event builders
// ============================================================================

// session.created — init success, session enters active
json make_session_created(const std::string & session_id,
                          const std::string & mode,
                          const ProtocolMetrics & metrics = {});

// response.output.delta kind=text
json make_text_delta(const std::string & session_id,
                     const std::string & response_id,
                     const std::string & text,
                     const ProtocolMetrics & metrics = {});

// response.output.delta kind=audio — audio is base64 float32 PCM
json make_audio_delta(const std::string & session_id,
                      const std::string & response_id,
                      const std::string & audio_base64,
                      const ProtocolMetrics & metrics = {});

// response.output.delta kind=listen — model switches to listen
json make_listen_delta(const std::string & session_id,
                       const std::string & response_id = "",
                       const ProtocolMetrics & metrics = {});

// response.done — turn_based response complete
// full_text: complete generated text (accumulated from deltas)
// audio_base64: full audio if TTS enabled, empty/absent if not
json make_response_done(const std::string & session_id,
                        const std::string & response_id,
                        const std::string & full_text,
                        const std::string & audio_base64 = "",
                        const std::string & reason = "turn_end",
                        const ProtocolMetrics & metrics = {});

// session.closed — session ended
json make_session_closed(const std::string & session_id,
                         const std::string & reason = "client_closed",
                         const std::string & diagnostic_message = "");

// ============================================================================
// Upstream message parsers
// ============================================================================

struct ParsedSessionInit {
    bool ok = false;
    std::string error;

    std::string mode; // "full_duplex" or "turn_based"

    // voice reference audio (base64 float32 PCM)
    std::string ref_audio_b64;
    std::string tts_ref_audio_b64;

    std::string system_prompt;

    // opaque pass-through config (sampling params)
    json config;
};

struct ParsedInput {
    bool ok = false;
    std::string error;

    // Full-duplex fields
    std::string audio_b64;                         // base64 float32 PCM
    std::vector<std::string> video_frames_b64;     // base64 JPEG frames
    int max_slice_nums = -1;

    // Turn-based fields
    bool streaming = true;
    int max_new_tokens = 512;
    float length_penalty = 1.1f;
    // messages[] — currently passed as raw json for backend to interpret
    json messages;
    std::string tts_ref_audio_b64;
    bool tts_enabled = false;
};

ParsedSessionInit parse_session_init(const json & msg);
ParsedInput parse_input_append(const json & msg);

// ============================================================================
// Helpers: base64 ↔ raw bytes
// ============================================================================

// Decode base64 string to raw bytes
std::vector<uint8_t> b64_decode(const std::string & b64);

// Decode base64 float32 PCM audio (16kHz mono) to float samples
std::vector<float> b64_to_float32_pcm(const std::string & b64);

// Encode float32 PCM samples to base64
std::string float32_pcm_to_b64(const float * samples, size_t n_samples);
