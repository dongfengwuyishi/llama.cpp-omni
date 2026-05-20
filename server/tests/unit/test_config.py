"""Unit tests for ``config.ServiceConfig`` loading.

The real ``load_config()`` reads a fixed path on disk, which is not
test-friendly. We exercise ``ServiceConfig(**data)`` directly to verify
schema validation, plus call ``load_config(path=...)`` against a temp file
for the end-to-end path.
"""

from __future__ import annotations

import json

import pytest

from config import ServiceConfig, load_config

pytestmark = pytest.mark.unit


# ============================================================
# Direct schema validation
# ============================================================


class TestServiceConfig:
    def test_minimal_pytorch(self):
        cfg = ServiceConfig(model={"model_path": "/tmp/model"})
        assert cfg.backend == "pytorch"
        assert cfg.model.model_path == "/tmp/model"
        # default ports
        assert cfg.service.gateway_port == 8006
        assert cfg.service.worker_base_port == 22400
        # convenience properties
        assert cfg.gateway_port == 8006
        assert cfg.worker_port(0) == 22400
        assert cfg.worker_port(1) == 22401

    def test_full_cpp(self):
        cfg = ServiceConfig(
            backend="cpp",
            model={"model_path": "unused"},
            cpp_backend={
                "llamacpp_root": "/abs/llama.cpp-omni",
                "model_dir": "/abs/MiniCPM-o-4_5-gguf",
                "llm_model": "MiniCPM-o-4_5-Q8_0.gguf",
                "ctx_size": 32768,
                "n_gpu_layers": 99,
            },
            service={
                "gateway_port": 9090,
                "worker_base_port": 22500,
                "max_queue_size": 50,
            },
        )
        assert cfg.backend == "cpp"
        assert cfg.cpp_backend.ctx_size == 32768
        assert cfg.gateway_port == 9090
        assert cfg.worker_port(0) == 22500
        assert cfg.max_queue_size == 50

    def test_invalid_backend(self):
        with pytest.raises(Exception):
            ServiceConfig(backend="rust", model={"model_path": "/x"})

    def test_invalid_attn(self):
        with pytest.raises(Exception):
            ServiceConfig(model={"model_path": "/x", "attn_implementation": "fake"})

    def test_worker_addresses(self):
        cfg = ServiceConfig(model={"model_path": "/x"})
        addrs = cfg.worker_addresses(3)
        assert addrs == [
            "localhost:22400",
            "localhost:22401",
            "localhost:22402",
        ]


# ============================================================
# load_config() with a tmp file
# ============================================================


class TestLoadConfig:
    def test_load_minimal(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"model": {"model_path": "/tmp/x"}}))
        cfg = load_config(path=str(path))
        assert cfg.model.model_path == "/tmp/x"

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(path=str(tmp_path / "absent.json"))

    def test_pytorch_missing_model_path(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({}))
        with pytest.raises(ValueError):
            load_config(path=str(path))

    def test_cpp_missing_llamacpp_root(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"backend": "cpp"}))
        with pytest.raises(ValueError):
            load_config(path=str(path))

    def test_cpp_missing_model_dir(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps({"backend": "cpp", "cpp_backend": {"llamacpp_root": "/foo"}})
        )
        with pytest.raises(ValueError):
            load_config(path=str(path))

    def test_cpp_full(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(
            json.dumps(
                {
                    "backend": "cpp",
                    "cpp_backend": {
                        "llamacpp_root": "/foo",
                        "model_dir": "/bar",
                    },
                }
            )
        )
        cfg = load_config(path=str(path))
        assert cfg.backend == "cpp"
        assert cfg.cpp_backend.llamacpp_root == "/foo"
        # No model.model_path required -> auto-filled with placeholder
        assert "unused" in cfg.model.model_path
