"""Tests for the tuner's utility math, layer ranking, and cell aggregation."""

from pathlib import Path

from src.tuner import (
    CellMetrics,
    _classify_protection,
    average_cells,
    compute_r_mem,
    compute_utility,
    load_profiler_ranking,
    skip_layers_for_budget,
)


# ---------------------------------------------------------------------------
# compute_r_mem
# ---------------------------------------------------------------------------

def test_r_mem_matches_known_chat_result():
    # Qwen3-4B (36 layers), k8v4, floor-only protection (4 layers) ->
    # R_mem 2.25x, matching INSIGHTS #37's measured chat result.
    r_mem = compute_r_mem("k8v4", n_protect=0, n_layers=36, n_floor=4)
    assert abs(r_mem - 2.25) < 1e-9


def test_r_mem_is_1x_for_auto_style_ratio_when_all_layers_protected():
    # Protecting every layer means no bytes are actually saved.
    r_mem = compute_r_mem("3bit_nc", n_protect=32, n_layers=36, n_floor=4)
    assert abs(r_mem - 1.0) < 1e-9


def test_r_mem_decreases_as_protection_increases():
    # More protected layers -> less compression (protection costs R_mem).
    r_low_protect = compute_r_mem("3bit_nc", n_protect=0, n_layers=36, n_floor=4)
    r_high_protect = compute_r_mem("3bit_nc", n_protect=8, n_layers=36, n_floor=4)
    assert r_high_protect < r_low_protect


def test_r_mem_is_model_agnostic():
    # Same preset/protection-budget shape, different layer count (e.g. a
    # 32-layer Llama vs a 36-layer Qwen) must not silently reuse Qwen's ratio.
    r_qwen = compute_r_mem("4bit_nc", n_protect=0, n_layers=36, n_floor=4)
    r_llama = compute_r_mem("4bit_nc", n_protect=0, n_layers=32, n_floor=4)
    assert r_qwen != r_llama


# ---------------------------------------------------------------------------
# compute_utility
# ---------------------------------------------------------------------------

CHAT_BASELINE = {"ppl": 10.35, "chat_tpot_ms": 11.5, "rag_ttft_ms": 636.0, "batch_tps": 850.0}


def test_utility_zero_when_ppl_exceeds_threshold():
    metrics = {"ppl": 10.35 * 1.01, "chat_tpot_ms": 10.0}  # +1% > chat's 0.5% threshold
    u = compute_utility("chat", metrics, CHAT_BASELINE, "k8v4", 0, 36, 4)
    assert u == 0.0


def test_utility_zero_when_a_gain_is_non_positive():
    metrics = {"ppl": 10.35, "batch_tps": 0.0}
    u = compute_utility("batch", metrics, CHAT_BASELINE, "4bit_nc", 0, 36, 4)
    assert u == 0.0


def test_utility_positive_for_viable_config():
    metrics = {"ppl": 10.34, "chat_tpot_ms": 12.1}  # slightly better PPL, within threshold
    u = compute_utility("chat", metrics, CHAT_BASELINE, "k8v4", 0, 36, 4)
    assert u > 0.0


def test_utility_missing_optional_metric_defaults_gain_to_one():
    # No chat_tpot_ms measured for this cell -> s_tpot gain defaults to 1.0,
    # so utility should equal r_mem**0.3 (chat's weight on r_mem alone).
    metrics = {"ppl": 10.35}
    u = compute_utility("chat", metrics, CHAT_BASELINE, "k8v4", 0, 36, 4)
    r_mem = compute_r_mem("k8v4", 0, 36, 4)
    assert abs(u - r_mem ** 0.3) < 1e-9


# ---------------------------------------------------------------------------
# skip_layers_for_budget / load_profiler_ranking
# ---------------------------------------------------------------------------

def test_skip_layers_for_budget_truncates_and_sorts():
    ranking = [5, 33, 4, 8]
    assert skip_layers_for_budget(ranking, 0) == []
    assert skip_layers_for_budget(ranking, 2) == [5, 33]
    assert skip_layers_for_budget(ranking, 4) == [4, 5, 8, 33]


def test_ranking_fallback_is_positional_and_model_agnostic(tmp_path):
    missing = tmp_path / "no_such_exp0_result.json"
    floor = frozenset({0, 1, 30, 31})
    ranking = load_profiler_ranking(floor, 32, exp0_path=missing)
    assert ranking == [l for l in range(32) if l not in floor]
    assert 30 not in ranking and 31 not in ranking


# ---------------------------------------------------------------------------
# _classify_protection
# ---------------------------------------------------------------------------

def test_classify_protection_empty_skip_is_stats_zero():
    assert _classify_protection([], ranking=[5, 33, 4]) == ("stats", 0)


def test_classify_protection_matches_ranking_prefix():
    ranking = [5, 33, 4, 8]
    assert _classify_protection([5, 33], ranking) == ("stats", 2)


def test_classify_protection_non_ranking_subset_is_positional():
    ranking = [5, 33, 4, 8]
    assert _classify_protection([2, 3], ranking) == ("positional", 2)


# ---------------------------------------------------------------------------
# average_cells
# ---------------------------------------------------------------------------

def _cell(preset, n_protect, rep, ppl, chat_tpot_ms=None):
    return CellMetrics(
        preset=preset, n_protect=n_protect, skip_layers=[], rep=rep, ppl=ppl,
        chat_tpot_ms=chat_tpot_ms, chat_ttft_ms=None, chat_tps=None,
        rag_ttft_ms=None, rag_tps=None, needle_acc=None, batch_tps=None,
    )


def test_average_cells_averages_across_reps():
    cells = [
        _cell("4bit_nc", 0, rep=0, ppl=10.4, chat_tpot_ms=12.0),
        _cell("4bit_nc", 0, rep=1, ppl=10.6, chat_tpot_ms=14.0),
    ]
    [avg] = average_cells(cells)
    assert avg["n_reps"] == 2
    assert abs(avg["ppl"] - 10.5) < 1e-9
    assert abs(avg["chat_tpot_ms"] - 13.0) < 1e-9


def test_average_cells_ignores_missing_values():
    cells = [
        _cell("4bit_nc", 0, rep=0, ppl=10.4, chat_tpot_ms=None),
        _cell("4bit_nc", 0, rep=1, ppl=10.6, chat_tpot_ms=14.0),
    ]
    [avg] = average_cells(cells)
    assert abs(avg["chat_tpot_ms"] - 14.0) < 1e-9  # only the non-None value counted


def test_average_cells_groups_by_preset_and_protection():
    cells = [
        _cell("4bit_nc", 0, rep=0, ppl=10.4),
        _cell("4bit_nc", 2, rep=0, ppl=10.3),
    ]
    averaged = average_cells(cells)
    assert len(averaged) == 2
    assert {(a["preset"], a["n_protect"]) for a in averaged} == {("4bit_nc", 0), ("4bit_nc", 2)}
