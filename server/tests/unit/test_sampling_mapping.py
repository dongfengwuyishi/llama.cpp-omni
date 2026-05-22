"""Unit tests for ``cpp_backend._sampling_from_generation``.

This is the boundary between the Python ``GenerationConfig`` (HF-style)
and the C++ ``update_session_config`` body (llama.cpp-style). Bugs at
this boundary silently kill RL reproducibility, so we pin every documented
mapping rule with an assertion.
"""
from __future__ import annotations

import pytest

from core.processors.cpp_backend import _sampling_from_generation
from core.schemas.common import GenerationConfig


def test_none_input_returns_empty_dict():
    assert _sampling_from_generation(None) == {}


def test_do_sample_false_maps_to_temp_zero():
    g = GenerationConfig(do_sample=False, temperature=0.7)
    out = _sampling_from_generation(g)
    assert out["llm_sampling"]["temp"] == 0.0
    # do_sample=False overrides temperature even if temperature was given
    assert "seed" not in out["llm_sampling"]


def test_do_sample_true_passes_temperature_through():
    g = GenerationConfig(do_sample=True, temperature=0.9, top_p=0.95, top_k=50)
    out = _sampling_from_generation(g)
    s = out["llm_sampling"]
    assert s["temp"] == 0.9
    assert s["top_p"] == 0.95
    assert s["top_k"] == 50


def test_seed_passes_through_when_given():
    g = GenerationConfig(do_sample=True, seed=42, temperature=0.7)
    out = _sampling_from_generation(g)
    assert out["llm_sampling"]["seed"] == 42


def test_seed_omitted_when_none():
    g = GenerationConfig(do_sample=True, temperature=0.7)
    out = _sampling_from_generation(g)
    assert "seed" not in out["llm_sampling"]


def test_top_k_zero_means_disabled_so_dropped():
    """HF top_k=0 = disabled. llama.cpp top_k<=0 = use vocab_size (not the
    same!), so we must NOT forward 0 — _sampling_from_generation drops it."""
    g = GenerationConfig(do_sample=True, top_k=0, temperature=0.7)
    out = _sampling_from_generation(g)
    assert "top_k" not in out["llm_sampling"]


def test_repetition_penalty_aliases():
    g_dict_hf = {"repetition_penalty": 1.2, "do_sample": True, "temperature": 0.7}
    g_dict_llcpp = {"repeat_penalty": 1.3, "do_sample": True, "temperature": 0.7}
    out_hf = _sampling_from_generation(g_dict_hf)
    out_ll = _sampling_from_generation(g_dict_llcpp)
    assert out_hf["llm_sampling"]["penalty_repeat"] == 1.2
    assert out_ll["llm_sampling"]["penalty_repeat"] == 1.3


def test_repetition_penalty_last_n_passes_through():
    g = GenerationConfig(do_sample=True, temperature=0.7, repetition_penalty_last_n=128)
    out = _sampling_from_generation(g)
    assert out["llm_sampling"]["penalty_last_n"] == 128


def test_tts_temperature_lifts_to_top_level_not_into_llm_sampling():
    """tts_temperature drives ctx_tts_sampler, NOT ctx_sampler — they're
    different. It must stay top-level so server.cpp's ``tts_temperature``
    handler picks it up, NOT inside ``llm_sampling``."""
    g = {"tts_temperature": 0.6, "do_sample": True, "temperature": 0.7}
    out = _sampling_from_generation(g)
    assert out["tts_temperature"] == 0.6
    assert "tts_temperature" not in out.get("llm_sampling", {})


def test_dict_input_works_just_like_pydantic():
    g_dict = {"do_sample": False, "temperature": 0.7, "top_p": 0.8, "top_k": 100}
    out = _sampling_from_generation(g_dict)
    assert out["llm_sampling"]["temp"] == 0.0
    assert out["llm_sampling"]["top_p"] == 0.8
    assert out["llm_sampling"]["top_k"] == 100


def test_max_new_tokens_does_not_leak_into_sampling():
    """max_new_tokens has per-request semantics, not session, so it
    must NOT show up in update_session_config's sampling block. It
    travels via the /v1/stream/decode body (chat_max_new_tokens)."""
    g = GenerationConfig(max_new_tokens=42, do_sample=True, temperature=0.7)
    out = _sampling_from_generation(g)
    assert "max_new_tokens" not in out
    assert "max_new_tokens" not in out.get("llm_sampling", {})
