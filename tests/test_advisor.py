"""Tests for the advisor's config formatting, GPU feasibility, and model lookup."""

import json
from pathlib import Path

import pytest

from src.advisor import (
    apply_gpu_filter,
    format_cli,
    format_python,
    is_feasible,
    load_configs,
    resolve_model_configs,
    resolve_sm,
)

CHAT_CFG = {
    "kv_cache_dtype": "turboquant_k8v4",
    "skip_layers": [],
    "effective_skip_layers": [0, 1, 34, 35],
    "n_protected_total": 4,
    "utility": 1.19,
    "ppl": 10.34,
    "r_mem": 2.25,
    "ppl_delta_pct": -0.02,
    "evidence": {"chat_tpot_ms": 12.1},
    "alternatives": [
        {"kv_cache_dtype": "turboquant_k8v4", "utility": 1.19},
        {"kv_cache_dtype": "turboquant_4bit_nc", "utility": 1.05},
    ],
}


# ---------------------------------------------------------------------------
# GPU feasibility
# ---------------------------------------------------------------------------

def test_resolve_sm_known_gpus():
    assert resolve_sm("RTX 3090") == 8.6
    assert resolve_sm("rtx 4090") == 8.9
    assert resolve_sm("NVIDIA A100 80GB") == 8.0


def test_resolve_sm_unknown_gpu_is_none():
    assert resolve_sm("Some Future GPU 9000") is None


def test_is_feasible_k8v4_requires_ada_or_newer():
    assert is_feasible("turboquant_k8v4", 8.9) is True
    assert is_feasible("turboquant_k8v4", 8.6) is False


def test_is_feasible_nc_presets_have_no_restriction():
    assert is_feasible("turboquant_3bit_nc", 8.6) is True
    assert is_feasible("turboquant_4bit_nc", 7.0) is True


def test_is_feasible_unknown_gpu_defaults_to_feasible():
    assert is_feasible("turboquant_k8v4", None) is True


# ---------------------------------------------------------------------------
# apply_gpu_filter
# ---------------------------------------------------------------------------

def test_apply_gpu_filter_keeps_feasible_champion():
    cfg = apply_gpu_filter(CHAT_CFG, "RTX 4090", sm=8.9)
    assert cfg["kv_cache_dtype"] == "turboquant_k8v4"
    assert "gpu_fallback_note" not in cfg


def test_apply_gpu_filter_falls_back_to_feasible_alternative():
    cfg = apply_gpu_filter(CHAT_CFG, "RTX 3090", sm=8.6)
    assert cfg["kv_cache_dtype"] == "turboquant_4bit_nc"
    assert "gpu_fallback_note" in cfg


def test_apply_gpu_filter_no_feasible_alternative_marks_nonviable():
    cfg_no_alts = {**CHAT_CFG, "alternatives": [CHAT_CFG]}  # only the infeasible champion
    result = apply_gpu_filter(cfg_no_alts, "RTX 3090", sm=8.6)
    assert result["viable"] is False


def test_apply_gpu_filter_passes_through_already_nonviable_config():
    cfg = {"viable": False, "reason": "no config meets PPL constraint"}
    assert apply_gpu_filter(cfg, "RTX 3090", sm=8.6) == cfg


def test_apply_gpu_filter_unknown_sm_is_noop():
    cfg = apply_gpu_filter(CHAT_CFG, "Mystery GPU", sm=None)
    assert cfg is CHAT_CFG


# ---------------------------------------------------------------------------
# resolve_model_configs
# ---------------------------------------------------------------------------

def test_resolve_model_configs_found():
    configs = {"Qwen/Qwen3-4B": {"chat": CHAT_CFG}}
    assert resolve_model_configs(configs, "Qwen/Qwen3-4B") == {"chat": CHAT_CFG}


def test_resolve_model_configs_missing_exits():
    configs = {"Qwen/Qwen3-4B": {"chat": CHAT_CFG}}
    with pytest.raises(SystemExit):
        resolve_model_configs(configs, "meta-llama/Llama-3.1-8B")


# ---------------------------------------------------------------------------
# load_configs
# ---------------------------------------------------------------------------

def test_load_configs_missing_file_exits(tmp_path):
    with pytest.raises(SystemExit):
        load_configs(tmp_path / "does_not_exist.json")


def test_load_configs_reads_json(tmp_path):
    path = tmp_path / "optimal_configs.json"
    data = {"Qwen/Qwen3-4B": {"chat": CHAT_CFG}}
    path.write_text(json.dumps(data))
    assert load_configs(path) == data


# ---------------------------------------------------------------------------
# format_cli / format_python
# ---------------------------------------------------------------------------

def test_format_cli_includes_skip_layers_when_present():
    cfg = {"kv_cache_dtype": "turboquant_3bit_nc", "skip_layers": [5, 33]}
    out = format_cli(cfg, "Qwen/Qwen3-4B")
    assert "--kv-cache-dtype turboquant_3bit_nc" in out
    assert "--kv-cache-dtype-skip-layers 5 33" in out
    assert "Qwen/Qwen3-4B" in out


def test_format_cli_omits_skip_layers_flag_when_floor_only():
    cfg = {"kv_cache_dtype": "turboquant_k8v4", "skip_layers": []}
    out = format_cli(cfg, "Qwen/Qwen3-4B")
    assert "--kv-cache-dtype-skip-layers" not in out


def test_format_python_produces_valid_kwargs_block():
    cfg = {"kv_cache_dtype": "turboquant_4bit_nc", "skip_layers": [5]}
    out = format_python(cfg, "Qwen/Qwen3-4B")
    assert out.startswith("LLM(")
    assert 'kv_cache_dtype="turboquant_4bit_nc"' in out
    assert "kv_cache_dtype_skip_layers=['5']" in out
