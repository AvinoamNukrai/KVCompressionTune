"""Tests for the shared model-topology helpers (no network required)."""

from src.model_info import floor_layers_for


def test_floor_layers_matches_qwen3_4b():
    # vLLM's hard-coded rule on the primary 36-layer model: {0, 1, 34, 35}.
    assert floor_layers_for(36) == frozenset({0, 1, 34, 35})


def test_floor_layers_generalizes_to_other_layer_counts():
    assert floor_layers_for(32) == frozenset({0, 1, 30, 31})
    assert floor_layers_for(28) == frozenset({0, 1, 26, 27})


def test_floor_layers_always_has_four_members_for_large_models():
    assert len(floor_layers_for(80)) == 4
