#include "protocol.h"
#include "common/base64.hpp"

#include <chrono>

// ============================================================================
// ProtocolMetrics
// ============================================================================

json ProtocolMetrics::to_json() const {
    json m;
    m["backend"] = backend;
    if (kv_cache_length > 0) {
        m["kv_cache_length"] = kv_cache_length;
    }
    if (prefill_ms > 0.0) {
        m["prefill_ms"] = prefill_ms;
    }
    if (generate_ms > 0.0) {
        m["generate_ms"] = generate_ms;
    }
    if (wall_clock_ms > 0.0) {
        m["wall_clock_ms"] = wall_clock_ms;
    }
    if (n_tokens > 0) {
        m["n_tokens"] = n_tokens;
    }
    return m;
}

// ============================================================================
// Downstream event builders
// ============================================================================

static double server_timestamp() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}

json make_session_created(const std::string & session_id,
                           const std::string & mode,
                           const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "session.created";
    ev["session_id"] = session_id;
    ev["mode"] = mode;
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_text_delta(const std::string & session_id,
                      const std::string & response_id,
                      const std::string & text,
                      const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.output.delta";
    ev["kind"] = "text";
    ev["session_id"] = session_id;
    ev["response_id"] = response_id;
    ev["text"] = text;
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_audio_delta(const std::string & session_id,
                       const std::string & response_id,
                       const std::string & audio_base64,
                       const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.output.delta";
    ev["kind"] = "audio";
    ev["session_id"] = session_id;
    ev["response_id"] = response_id;
    ev["audio"] = audio_base64;
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_listen_delta(const std::string & session_id,
                        const std::string & response_id,
                        const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.output.delta";
    ev["kind"] = "listen";
    ev["session_id"] = session_id;
    if (!response_id.empty()) {
        ev["response_id"] = response_id;
    }
    ev["server_send_ts"] = server_timestamp();

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_response_done(const std::string & session_id,
                         const std::string & response_id,
                         const std::string & full_text,
                         const std::string & audio_base64,
                         const std::string & reason,
                         const ProtocolMetrics & metrics) {
    json ev;
    ev["type"] = "response.done";
    ev["session_id"] = session_id;
    ev["response_id"] = response_id;
    ev["text"] = full_text;
    ev["reason"] = reason;
    ev["server_send_ts"] = server_timestamp();

    if (!audio_base64.empty()) {
        ev["audio"] = audio_base64;
    } else {
        ev["audio"] = nullptr;
    }

    json m = metrics.to_json();
    if (!m.empty()) {
        ev["metrics"] = m;
    }
    return ev;
}

json make_session_closed(const std::string & session_id,
                          const std::string & reason,
                          const std::string & diagnostic_message) {
    json ev;
    ev["type"] = "session.closed";
    ev["session_id"] = session_id;
    ev["reason"] = reason;
    ev["server_send_ts"] = server_timestamp();

    if (!diagnostic_message.empty()) {
        json diag;
        diag["message"] = diagnostic_message;
        ev["diagnostic"] = diag;
    }
    return ev;
}

// ============================================================================
// Upstream message parsers
// ============================================================================

static std::string json_str(const json & j, const std::string & key,
                             const std::string & default_val = "") {
    if (j.contains(key) && j.at(key).is_string()) {
        return j.at(key).get<std::string>();
    }
    return default_val;
}

static bool json_bool(const json & j, const std::string & key, bool default_val = false) {
    if (j.contains(key) && j.at(key).is_boolean()) {
        return j.at(key).get<bool>();
    }
    return default_val;
}

static int json_int(const json & j, const std::string & key, int default_val = 0) {
    if (j.contains(key) && j.at(key).is_number_integer()) {
        return j.at(key).get<int>();
    }
    return default_val;
}

static float json_float(const json & j, const std::string & key, float default_val = 0.0f) {
    if (j.contains(key) && j.at(key).is_number()) {
        return j.at(key).get<float>();
    }
    return default_val;
}

ParsedSessionInit parse_session_init(const json & msg) {
    ParsedSessionInit out;

    // Validate type
    if (!msg.contains("type") || msg.at("type") != "session.init") {
        out.error = "expected type=session.init";
        return out;
    }

    if (!msg.contains("payload") || !msg.at("payload").is_object()) {
        out.error = "missing payload";
        return out;
    }

    const json & p = msg.at("payload");

    // mode
    std::string mode = json_str(p, "mode", "full_duplex");
    if (mode != "full_duplex" && mode != "turn_based") {
        out.error = "invalid mode: " + mode + " (expected full_duplex or turn_based)";
        return out;
    }
    out.mode = mode;

    // voice (reference audio)
    if (p.contains("voice") && p.at("voice").is_object()) {
        const json & v = p.at("voice");
        out.ref_audio_b64 = json_str(v, "ref_audio");
        out.tts_ref_audio_b64 = json_str(v, "tts_ref_audio");
    }

    // system_prompt
    out.system_prompt = json_str(p, "system_prompt");

    // config (opaque pass-through)
    if (p.contains("config") && p.at("config").is_object()) {
        out.config = p.at("config");
    }

    out.ok = true;
    return out;
}

ParsedInput parse_input_append(const json & msg) {
    ParsedInput out;

    if (!msg.contains("type") || msg.at("type") != "input.append") {
        out.error = "expected type=input.append";
        return out;
    }

    if (!msg.contains("input") || !msg.at("input").is_object()) {
        out.error = "missing input";
        return out;
    }

    const json & in = msg.at("input");

    // Full-duplex fields: audio (required), video_frames (optional)
    if (in.contains("audio") && in.at("audio").is_string()) {
        out.audio_b64 = in.at("audio").get<std::string>();
    }

    if (in.contains("video_frames") && in.at("video_frames").is_array()) {
        for (const auto & f : in.at("video_frames")) {
            if (f.is_string()) {
                out.video_frames_b64.push_back(f.get<std::string>());
            }
        }
    }

    out.max_slice_nums = json_int(in, "max_slice_nums", -1);

    // Turn-based fields: messages (required), streaming (required), generation
    if (in.contains("messages") && in.at("messages").is_array()) {
        out.messages = in.at("messages");
    }

    out.streaming = json_bool(in, "streaming", true);

    if (in.contains("generation") && in.at("generation").is_object()) {
        const json & gen = in.at("generation");
        out.max_new_tokens = json_int(gen, "max_new_tokens", 512);
        out.length_penalty = json_float(gen, "length_penalty", 1.1f);
    }

    if (in.contains("tts") && in.at("tts").is_object()) {
        const json & tts = in.at("tts");
        out.tts_enabled = json_bool(tts, "enabled", false);
        out.tts_ref_audio_b64 = json_str(tts, "ref_audio_data");
    }

    out.ok = true;
    return out;
}

// ============================================================================
// Helpers: base64 ↔ raw bytes
// ============================================================================

std::vector<uint8_t> b64_decode(const std::string & b64) {
    std::string raw = base64::decode(b64);
    return std::vector<uint8_t>(raw.begin(), raw.end());
}

std::vector<float> b64_to_float32_pcm(const std::string & b64) {
    std::string raw = base64::decode(b64);
    const float * ptr = reinterpret_cast<const float *>(raw.data());
    size_t n = raw.size() / sizeof(float);
    return std::vector<float>(ptr, ptr + n);
}

std::string float32_pcm_to_b64(const float * samples, size_t n_samples) {
    const char * ptr = reinterpret_cast<const char *>(samples);
    size_t byte_size = n_samples * sizeof(float);
    return base64::encode(ptr, byte_size);
}
