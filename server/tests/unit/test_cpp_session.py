"""Unit tests for ``core.processors.cpp_session``.

These tests focus on the **HTTP body construction** and **error semantics**
of ``_StreamHttpClient`` - the single non-trivial behavior change vs the
legacy ``CppBackendWorker._call_*`` helpers is that ``prefill`` now raises
on non-200 instead of silently logging (Phase 1 of the simplex/duplex
refactor).

We do not spawn a real ``llama-server`` here; the subprocess plumbing in
``_CppServerProc`` is exercised end-to-end via the integration tests in
``server/tests/integration/`` once Phase 4 wires the new backends into
the worker.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def stream_client():
    """Return a ``_StreamHttpClient`` whose underlying ``http`` member is a
    plain ``MagicMock`` so each test can dictate the response and inspect
    the request payloads without needing a live server."""
    from core.processors.cpp_session import _StreamHttpClient
    fake_http = MagicMock()
    return _StreamHttpClient("http://127.0.0.1:19060", fake_http), fake_http


def _ok(payload):
    """Build a fake httpx Response that mimics ``status_code=200`` + JSON."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = payload
    return resp


def _err(status, text):
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    return resp


def test_prefill_raises_on_non_200(stream_client):
    """Phase 1 invariant: ``prefill`` no longer swallows server errors.

    The legacy ``CppBackendWorker._call_prefill`` only logged on non-200
    and returned silently, which masked the D3 system-prompt re-init
    failure. The new client must surface the error so the caller can
    react (e.g. abort the duplex session instead of pushing user audio
    into an uninitialized C++ context).
    """
    client, http = stream_client
    http.post.return_value = _err(400, "at least one of ... must be non-empty")
    with pytest.raises(RuntimeError, match="prefill"):
        client.prefill(cnt=0)


def test_prefill_cnt0_all_empty_body(stream_client):
    """``cnt=0`` with all-empty content fields is the system-init slot.

    The Phase 0 server.cpp patch lets the server accept this body; this
    test asserts the client actually sends ``audio_path_prefix=""``,
    ``img_path_prefix=""``, no ``text`` key (per the legacy behavior of
    omitting empty strings) and ``cnt=0`` so the server's relaxed branch
    is exercised."""
    client, http = stream_client
    http.post.return_value = _ok({"kv_cache_length": 1234})
    client.prefill(cnt=0)
    assert http.post.called, "prefill must POST to llama-server"
    call_kwargs = http.post.call_args.kwargs
    assert call_kwargs["json"]["audio_path_prefix"] == ""
    assert call_kwargs["json"]["img_path_prefix"] == ""
    assert "text" not in call_kwargs["json"], (
        "empty text must be omitted, server treats missing key as empty"
    )
    assert call_kwargs["json"]["cnt"] == 0
    assert client.kv_cache_length == 1234


def test_prefill_user_chunk_with_audio_and_text(stream_client):
    """Mixed-content prefill body must include only the populated fields."""
    client, http = stream_client
    http.post.return_value = _ok({})
    client.prefill(audio_path="/tmp/a.wav", text="hello", cnt=3, max_slice_nums=2)
    body = http.post.call_args.kwargs["json"]
    assert body["audio_path_prefix"] == "/tmp/a.wav"
    assert body["text"] == "hello"
    assert body["cnt"] == 3
    assert body["max_slice_nums"] == 2


def test_update_session_config_body_includes_sampling_and_return_logits(stream_client):
    """update_session_config must merge the sampling dict at top-level
    (matching the C++ schema) and pass ``return_logits`` through. The
    backend layer is responsible for whitelisting the dict before
    handing it in - this client just dumps it."""
    client, http = stream_client
    # First call is the implicit /break drain; second is the actual config update.
    http.post.side_effect = [_ok({}), _ok({"kv_cache_length": 5678})]
    client.update_session_config(
        duplex_mode=True,
        voice_clone_prompt="VCP",
        assistant_prompt="AP",
        sampling={"force_listen_count": 3, "llm_sampling": {"temp": 0.0, "seed": 42}},
        return_logits=True,
    )
    assert http.post.call_count == 2
    break_call, cfg_call = http.post.call_args_list
    assert "/v1/stream/break" in break_call.args[0]
    assert "/v1/stream/update_session_config" in cfg_call.args[0]
    body = cfg_call.kwargs["json"]
    assert body["duplex_mode"] is True
    assert body["voice_clone_prompt"] == "VCP"
    assert body["assistant_prompt"] == "AP"
    assert body["force_listen_count"] == 3
    assert body["llm_sampling"] == {"temp": 0.0, "seed": 42}
    assert body["return_logits"] is True
    assert client.kv_cache_length == 5678


def test_update_session_config_raises_on_non_200(stream_client):
    client, http = stream_client
    http.post.side_effect = [_ok({}), _err(500, "boom")]
    with pytest.raises(RuntimeError, match="update_session_config failed"):
        client.update_session_config(
            duplex_mode=False,
            voice_clone_prompt="x",
            assistant_prompt="y",
        )


def test_decode_body_passes_optional_fields(stream_client):
    """force_listen / round_idx / max_new_tokens / logit_* must all flow
    through into the decode body. This exercises the D7/D8 fix surface
    that the new DuplexCppBackend will rely on."""
    client, http = stream_client
    http.post.return_value = _ok({})
    client.decode(
        stream=True,
        round_idx=5,
        length_penalty=1.05,
        max_new_tokens=42,
        force_listen=True,
        logit_format="file",
        logit_output_dir="/tmp/logits/",
        logit_filename="chat-w0-pid-seq.safetensors",
        logit_extra_metadata={"request_id": "rid-1"},
        timeout=30.0,
    )
    body = http.post.call_args.kwargs["json"]
    assert body["stream"] is True
    assert body["round_idx"] == 5
    assert body["length_penalty"] == pytest.approx(1.05)
    assert body["max_new_tokens"] == 42
    assert body["force_listen"] is True
    assert body["logit_format"] == "file"
    assert body["logit_output_dir"] == "/tmp/logits/"
    assert body["logit_filename"] == "chat-w0-pid-seq.safetensors"
    assert body["logit_extra_metadata"] == {"request_id": "rid-1"}


def test_decode_omits_unset_optionals(stream_client):
    """Decode body must not carry optional keys that were not provided -
    the C++ server applies its own defaults when a field is missing."""
    client, http = stream_client
    http.post.return_value = _ok({})
    client.decode(stream=False)
    body = http.post.call_args.kwargs["json"]
    assert body["stream"] is False
    assert "round_idx" not in body
    assert "max_new_tokens" not in body
    assert "force_listen" not in body
    assert "logit_format" not in body


def test_break_swallows_exceptions(stream_client):
    """``break_`` is best-effort: a flaky server during teardown must not
    raise, otherwise the worker shutdown path breaks. The legacy code
    had a try/except around the same call."""
    client, http = stream_client
    http.post.side_effect = RuntimeError("connection refused")
    client.break_(reason="shutdown")  # must not raise


def test_cpp_server_proc_url_format():
    """Sanity check the URL helper - the simplex/duplex backends rely on
    this string format."""
    from core.processors.cpp_session import _CppServerProc
    proc = _CppServerProc(
        llamacpp_root="/nonexistent",
        model_dir="/nonexistent",
        port=12345,
    )
    assert proc.url == "http://127.0.0.1:12345"
