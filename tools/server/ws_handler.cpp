#include "ws_handler.h"
#include "session.h"
#include "protocol.h"
#include "omni.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <cpp-httplib/httplib.h>

#include <thread>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <deque>

namespace fs = std::filesystem;

// ============================================================================
// TempMediaFiles helpers
// ============================================================================

std::string TempMediaFiles::write_temp_file(const std::string & temp_dir, const std::string & prefix,
                                            const std::string & suffix, const void * data, size_t len) {
    fs::path dir(temp_dir);
    if (!fs::exists(dir)) {
        fs::create_directories(dir);
    }
    std::string path = (dir / (prefix + suffix)).string();
    std::ofstream out(path, std::ios::binary);
    if (!out) return "";
    out.write(static_cast<const char *>(data), len);
    out.close();
    if (!out) return "";
    return path;
}

std::string TempMediaFiles::write_audio_wav(const std::string & b64, const std::string & temp_dir, int counter) {
    // Decode base64 → float32 PCM samples
    auto pcm = b64_to_float32_pcm(b64);
    if (pcm.empty()) return "";

    int n_samples = static_cast<int>(pcm.size());
    int sample_rate = 16000;
    int n_channels = 1;
    int bits_per_sample = 32;
    int byte_rate = sample_rate * n_channels * bits_per_sample / 8;
    int block_align = n_channels * bits_per_sample / 8;
    int data_size = n_samples * block_align;
    int file_size = 36 + data_size;

    // Build minimal WAV header + PCM data
    std::vector<char> wav(44 + data_size);
    auto wr = [&](int offset, const char * s, int n) { memcpy(&wav[offset], s, n); };
    auto wi = [&](int offset, int32_t val) { memcpy(&wav[offset], &val, 4); };
    auto ws = [&](int offset, int16_t val) { memcpy(&wav[offset], &val, 2); };

    wr(0,  "RIFF", 4); wi(4,  file_size);
    wr(8,  "WAVE", 4);
    wr(12, "fmt ", 4); wi(16, 16);            // subchunk size = 16 for PCM
    ws(20, 3);                                 // audio format = IEEE float
    ws(22, static_cast<int16_t>(n_channels));
    wi(24, sample_rate); wi(28, byte_rate);
    ws(32, static_cast<int16_t>(block_align));
    ws(34, static_cast<int16_t>(bits_per_sample));
    wr(36, "data", 4); wi(40, data_size);
    memcpy(&wav[44], pcm.data(), data_size);

    return write_temp_file(temp_dir, "audio_", "." + std::to_string(counter) + ".wav", wav.data(), wav.size());
}

std::string TempMediaFiles::write_image_jpeg(const std::string & b64, const std::string & temp_dir, int counter) {
    auto raw = b64_decode(b64);
    if (raw.empty()) return "";
    return write_temp_file(temp_dir, "image_", "." + std::to_string(counter) + ".jpg", raw.data(), raw.size());
}

void TempMediaFiles::cleanup() {
    if (!audio_path.empty()) { fs::remove(audio_path); audio_path.clear(); }
    if (!image_path.empty()) { fs::remove(image_path); image_path.clear(); }
}

// ============================================================================
// Session-level omni init helper
// ============================================================================

static omni_context * create_session_octx(common_params & params, const ParsedSessionInit & init,
                                          llama_model * model, llama_context * ctx,
                                          const std::string & output_dir) {
    int media_type = 2; // omni
    bool duplex_mode = (init.mode == "full_duplex");

    // Build params for omni_init
    auto & p = params;
    p.n_predict = 2048;

    omni_context * octx = omni_init(&p, media_type, /*use_tts*/true, /*tts_bin_dir*/"", /*tts_gpu_layers*/99,
                                     /*token2wav_device*/"gpu:0", duplex_mode,
                                     model, ctx, output_dir);
    if (!octx) {
        LOG_ERR("create_session_octx: omni_init failed\n");
        return nullptr;
    }

    octx->async = true;
    octx->duplex_mode = duplex_mode;

    // Voice clone / system prompt
    if (!init.system_prompt.empty()) {
        octx->omni_assistant_prompt = init.system_prompt;
    }

    LOG_INF("create_session_octx: session octx created, duplex=%d, output_dir=%s\n",
            duplex_mode, output_dir.c_str());
    return octx;
}

// ============================================================================
// Main WS handler
// ============================================================================

void handle_ws_backend(httplib::ws::WebSocket & ws,
                        SessionManager & session_mgr,
                        common_params & params_base,
                        llama_model * model,
                        llama_context * ctx,
                        std::mutex & octx_mutex) {
    const std::string temp_dir = fs::temp_directory_path() / "omni_ws";
    fs::create_directories(temp_dir);
    int msg_counter = 0;

    // Helper: fail-fast — send session.closed and close WS
    auto fail_fast = [&](const std::string & session_id, const std::string & reason) {
        if (!session_id.empty()) {
            std::string ev = make_session_closed(session_id, reason).dump();
            ws.send(ev);
        }
        session_mgr.close(session_id);
        ws.close();
    };

    // Helper: send a JSON event over WS
    auto send_event = [&](const json & ev) -> bool {
        return ws.send(ev.dump());
    };

    // ================================================================
    // Step 1: Read first message — must be session.init
    // ================================================================
    std::string raw_first;
    auto read_result = ws.read(raw_first);
    if (read_result != httplib::ws::ReadResult::Text) {
        LOG_WRN("WS /backend: failed to read init message\n");
        return; // no session yet, just return
    }

    json first_msg;
    try {
        first_msg = json::parse(raw_first);
    } catch (...) {
        LOG_ERR("WS /backend: invalid JSON in init message\n");
        ws.close();
        return;
    }

    auto parsed_init = parse_session_init(first_msg);
    if (!parsed_init.ok) {
        LOG_ERR("WS /backend: session.init parse failed: %s\n", parsed_init.error.c_str());
        // Fail-fast without session — just close
        ws.close();
        return;
    }

    // ================================================================
    // Step 2: Allocate & activate session, create omni_context
    // ================================================================
    std::string session_id = session_mgr.allocate();
    if (session_id.empty()) {
        // Already an active session
        LOG_ERR("WS /backend: session.init rejected — active session exists\n");
        ws.close();
        return;
    }

    std::string session_output_dir = (fs::path(temp_dir) / session_id).string();

    omni_context * octx = create_session_octx(params_base, parsed_init, model, ctx, session_output_dir);
    if (!octx) {
        fail_fast(session_id, "omni_init_failed");
        return;
    }

    // Voice prefill if voice audio provided
    if (!parsed_init.ref_audio_b64.empty()) {
        std::string voice_wav = TempMediaFiles::write_audio_wav(parsed_init.ref_audio_b64, temp_dir, msg_counter++);
        if (voice_wav.empty()) {
            fail_fast(session_id, "voice_audio_decode_failed");
            return;
        }
        std::lock_guard<std::mutex> lock(octx_mutex);
        if (!stream_prefill(octx, voice_wav, /*img*/"", /*index*/0)) {
            LOG_ERR("WS /backend: voice prefill failed\n");
            fs::remove(voice_wav);
            fail_fast(session_id, "voice_prefill_failed");
            return;
        }
        fs::remove(voice_wav);
        if (octx->llm_thread_info) {
            octx->llm_thread_info->start = std::chrono::steady_clock::now();
        }
    }

    // Activate session
    {
        std::lock_guard<std::mutex> lock(octx_mutex);
        if (!session_mgr.activate(session_id, octx, /*owns_octx*/true)) {
            LOG_ERR("WS /backend: session activate failed for %s\n", session_id.c_str());
            omni_free(octx);
            fail_fast(session_id, "activate_failed");
            return;
        }
    }

    // Send session.created
    ProtocolMetrics metrics;
    metrics.backend = "llama.cpp-omni";
    send_event(make_session_created(session_id, parsed_init.mode, metrics));

    LOG_INF("WS /backend: session %s activated, mode=%s\n", session_id.c_str(), parsed_init.mode.c_str());

    // ================================================================
    // Step 3: Read loop — process input.append messages
    // ================================================================
    std::string raw;
    std::string response_id_counter; // per-session response counter
    int response_seq = 0;

    while (true) {
        read_result = ws.read(raw);
        if (read_result != httplib::ws::ReadResult::Text) {
            break; // WS closed or error
        }

        json msg;
        try {
            msg = json::parse(raw);
        } catch (...) {
            fail_fast(session_id, "invalid_json");
            return;
        }

        // Validate message type
        std::string msg_type;
        if (msg.contains("type") && msg.at("type").is_string()) {
            msg_type = msg.at("type").get<std::string>();
        }

        if (msg_type != "input.append") {
            // Unknown or invalid type after init → fail-fast
            LOG_ERR("WS /backend: unexpected message type '%s' from session %s\n",
                    msg_type.c_str(), session_id.c_str());
            fail_fast(session_id, "unexpected_message_type");
            return;
        }

        auto parsed_input = parse_input_append(msg);
        if (!parsed_input.ok) {
            fail_fast(session_id, "invalid_input");
            return;
        }

        // ================================================================
        // Full-duplex input processing
        // ================================================================
        TempMediaFiles tmp_files;

        // Write audio to temp WAV
        if (!parsed_input.audio_b64.empty()) {
            tmp_files.audio_path = TempMediaFiles::write_audio_wav(
                parsed_input.audio_b64, temp_dir, msg_counter++);
        }

        // Write first video frame to temp image
        if (!parsed_input.video_frames_b64.empty()) {
            tmp_files.image_path = TempMediaFiles::write_image_jpeg(
                parsed_input.video_frames_b64[0], temp_dir, msg_counter++);
        }

        // Prefill
        {
            std::lock_guard<std::mutex> lock(octx_mutex);
            if (!stream_prefill(octx, tmp_files.audio_path, tmp_files.image_path,
                                msg_counter, parsed_input.max_slice_nums)) {
                tmp_files.cleanup();
                fail_fast(session_id, "prefill_failed");
                return;
            }
        }

        tmp_files.cleanup();

        // Generate response_id
        response_seq++;
        std::string response_id = session_id + "_resp_" + std::to_string(response_seq);

        // Decode: start background thread, poll text_queue on this thread
        std::string debug_dir = session_output_dir;
        {
            // Reset text streaming state
            {
                std::lock_guard<std::mutex> lk(octx->text_mtx);
                octx->text_queue.clear();
                octx->text_done_flag = false;
                octx->text_streaming = true;
            }

            std::thread decode_thread([octx, debug_dir]() {
                stream_decode(octx, debug_dir, -1);
            });

            // Collect full text for response.done
            std::string full_text;

            // Poll text_queue
            while (true) {
                std::string frag;
                {
                    std::unique_lock<std::mutex> lk(octx->text_mtx);
                    octx->text_cv.wait_for(lk, std::chrono::milliseconds(200), [&]{
                        return !octx->text_queue.empty() || octx->text_done_flag;
                    });

                    if (!octx->text_queue.empty()) {
                        frag = std::move(octx->text_queue.front());
                        octx->text_queue.pop_front();
                    }

                    if (octx->text_done_flag && octx->text_queue.empty()) {
                        break;
                    }
                }

                if (!frag.empty()) {
                    if (frag == "__IS_LISTEN__") {
                        // Model switched to listen
                        send_event(make_listen_delta(session_id, response_id));
                        break; // Done for this input
                    } else if (frag == "__END_OF_TURN__") {
                        // Turn ended — will be handled by response.done
                    } else {
                        // Text delta
                        full_text += frag;
                        send_event(make_text_delta(session_id, response_id, frag));
                    }
                }

                // Check if session was closed externally
                if (session_mgr.get(session_id) == nullptr) {
                    // Session was closed externally (e.g. HTTP close endpoint)
                    // Join the decode thread and exit both loops
                    if (decode_thread.joinable()) {
                        decode_thread.join();
                    }
                    goto cleanup;
                }
            }

            if (decode_thread.joinable()) {
                decode_thread.join();
            }

            // Send response.done
            send_event(make_response_done(session_id, response_id, full_text, /*audio*/"", "turn_end"));
        }
    }

cleanup:

    // ================================================================
    // Step 4: WS disconnect — cleanup
    // ================================================================
    LOG_INF("WS /backend: session %s disconnected\n", session_id.c_str());

    // Send session.closed (best-effort)
    std::string close_ev = make_session_closed(session_id, "client_disconnected").dump();
    ws.send(close_ev);

    // Close session — frees omni_context
    session_mgr.close(session_id);

    ws.close();
}
