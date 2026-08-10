"""Shared model-topology helpers: layer count and vLLM's positional floor.

Single source of truth so profiler.py and tuner.py agree on how many
layers a model has and which layers vLLM's hard-coded boundary rule
protects, instead of each hardcoding Qwen3-4B's 36-layer layout.
"""
from __future__ import annotations


def detect_n_layers(model_name: str) -> int:
    """Detect number of transformer layers from the HF model config."""
    from transformers import AutoConfig  # noqa: PLC0415

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise ValueError(f"Cannot detect layer count for {model_name}")


def floor_layers_for(n_layers: int) -> frozenset[int]:
    """vLLM's hard-coded boundary-protection rule: first 2 + last 2 layers."""
    return frozenset({0, 1, n_layers - 2, n_layers - 1})
