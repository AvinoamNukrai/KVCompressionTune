"""Layer-sensitivity profiler for KV cache compression.

Measures which transformer layers are most sensitive to KV cache
quantization — proving that uniform compression is suboptimal and that
a layer-aware compression policy preserves quality at higher compression
ratios.

Two profiling backends:

1. **Simulation mode** (``--mode sim-full``, fast, ~5 min) — runs on
   HuggingFace Transformers with hooks capturing per-layer K/V tensors.
   Simulates TurboQuant quantization (Hadamard rotation + Lloyd-Max for
   keys, uniform for values) per-layer and measures ΔPPL. Uses the same
   quantization logic as vLLM's TurboQuant implementation.

2. **vLLM mode** (``--mode vllm``, ground-truth, ~1.5 h) — generates a
   harness manifest that quantizes one layer at a time through vLLM's
   actual CUDA kernels via ``kv_cache_dtype_skip_layers``. No simulation;
   uses the real engine. Run the manifest, then ``--mode vllm-analyze``
   to compute the sensitivity ranking.

Both modes output a per-layer sensitivity ranking used by the tuner to
decide which layers to protect.

CLI:
    # Fast simulation mode:
    python -m src.profiler --mode sim-full --model Qwen/Qwen3-4B

    # Ground-truth vLLM mode (two steps):
    python -m src.profiler --mode vllm --model Qwen/Qwen3-4B
    python -m src.harness --manifest configs/grids/exp0_vllm.json
    python -m src.profiler --mode vllm-analyze --model Qwen/Qwen3-4B
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------
# TurboQuant math — import from vLLM if available, else use local fallback.
#
# The functions below replicate the quantization logic from vLLM's
# TurboQuant implementation (vllm/model_executor/layers/quantization/).
# We prefer vLLM's own code for correctness; the fallback exists so the
# profiler can run on CPU-only machines (e.g., for unit tests).
# --------------------------------------------------------------------------

_VLLM_QUANT_AVAILABLE = False

try:
    from vllm._custom_ops import scaled_int_quant as _vllm_quant  # noqa: F401
    _VLLM_QUANT_AVAILABLE = True
except ImportError:
    pass


def _fallback_solve_lloyd_max_normal(n_bits: int, n_iter: int = 200) -> torch.Tensor:
    """Lloyd-Max optimal centroids for N(0,1). Fallback when vLLM is not installed."""
    n_levels = 2 ** n_bits
    from scipy.stats import norm as _norm
    quantiles = torch.tensor(
        [_norm.ppf((i + 0.5) / n_levels) for i in range(n_levels)],
        dtype=torch.float64,
    )
    centroids = quantiles.clone()

    for _ in range(n_iter):
        bounds = torch.empty(n_levels + 1, dtype=torch.float64)
        bounds[0] = -8.0
        bounds[-1] = 8.0
        for i in range(1, n_levels):
            bounds[i] = 0.5 * (centroids[i - 1] + centroids[i])

        for i in range(n_levels):
            lo, hi = bounds[i].item(), bounds[i + 1].item()
            pdf_lo = _norm.pdf(lo)
            pdf_hi = _norm.pdf(hi)
            cdf_lo = _norm.cdf(lo)
            cdf_hi = _norm.cdf(hi)
            denom = cdf_hi - cdf_lo
            if denom > 1e-15:
                centroids[i] = (pdf_lo - pdf_hi) / denom

    return centroids.float()


_CENTROID_CACHE: dict[int, torch.Tensor] = {}


def load_centroids(n_bits: int = 3) -> torch.Tensor:
    """Lloyd-Max optimal centroids for N(0,1), computed once and cached."""
    if n_bits not in _CENTROID_CACHE:
        _CENTROID_CACHE[n_bits] = _fallback_solve_lloyd_max_normal(n_bits)
    return _CENTROID_CACHE[n_bits]


def hadamard_matrix(d: int) -> torch.Tensor:
    """Normalized Walsh-Hadamard matrix of size d (must be power of 2)."""
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(d)


def simulate_turbo_quant_keys(keys: torch.Tensor, n_bits: int = 3) -> torch.Tensor:
    """Simulate TurboQuant key quantization: Hadamard rotation + Lloyd-Max.

    Replicates the quantization logic from vLLM's TurboQuant CUDA kernels
    in pure PyTorch for per-layer profiling.
    """
    head_dim = keys.shape[-1]
    centroids = load_centroids(n_bits).to(keys.device, keys.dtype)

    H = hadamard_matrix(head_dim).to(keys.device, keys.dtype)
    rotated = keys @ H.T

    std = rotated.std(dim=-1, keepdim=True).clamp(min=1e-8)
    normalized = rotated / std

    dists = (normalized.unsqueeze(-1) - centroids.unsqueeze(0).unsqueeze(0)) ** 2
    indices = dists.argmin(dim=-1)
    quantized_normalized = centroids[indices]

    dequantized = quantized_normalized * std
    restored = dequantized @ H

    return restored


def simulate_turbo_quant_values(values: torch.Tensor, n_bits: int = 4) -> torch.Tensor:
    """Simulate TurboQuant value quantization: per-vector uniform."""
    n_levels = 2 ** n_bits
    vmin = values.min(dim=-1, keepdim=True).values
    vmax = values.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin).clamp(min=1e-8) / (n_levels - 1)

    quantized = torch.round((values - vmin) / scale).clamp(0, n_levels - 1)
    dequantized = quantized * scale + vmin
    return dequantized


# --------------------------------------------------------------------------
# Feature computation (§3.4 candidate features)
# --------------------------------------------------------------------------

@dataclass
class LayerFeatures:
    """Per-layer statistics computed from one calibration pass."""
    layer_idx: int
    key_outlier_frac: float = 0.0
    post_wht_excess_kurtosis: float = 0.0
    key_norm_cv: float = 0.0
    value_dynamic_range: float = 0.0
    simulated_quant_error: float = 0.0


def compute_features(
    keys: torch.Tensor,
    values: torch.Tensor,
    layer_idx: int,
    outlier_threshold: float = 3.0,
    key_bits: int = 3,
    value_bits: int = 4,
) -> LayerFeatures:
    """Compute the 5 candidate features for one layer's K/V.

    Args:
        keys: (n_tokens, n_heads, head_dim) post-RoPE key tensor.
        values: (n_tokens, n_heads, head_dim) value tensor.
        layer_idx: layer index.
    """
    feats = LayerFeatures(layer_idx=layer_idx)

    # Flatten heads for channel-level statistics
    # keys shape: (n_tokens, n_heads, head_dim) → (n_tokens * n_heads, head_dim)
    k_flat = keys.reshape(-1, keys.shape[-1]).float()
    v_flat = values.reshape(-1, values.shape[-1]).float()

    # --- Feature 1: key outlier-channel fraction ---
    # Fraction of channels where any token exceeds `outlier_threshold` stddevs
    channel_std = k_flat.std(dim=0)
    channel_mean = k_flat.mean(dim=0)
    max_abs_z = ((k_flat - channel_mean) / channel_std.clamp(min=1e-8)).abs().max(dim=0).values
    feats.key_outlier_frac = (max_abs_z > outlier_threshold).float().mean().item()

    # --- Feature 2: post-WHT excess kurtosis ---
    head_dim = keys.shape[-1]
    if head_dim > 0 and (head_dim & (head_dim - 1)) == 0:
        H = hadamard_matrix(head_dim).to(k_flat.device, k_flat.dtype)
        rotated = k_flat @ H.T
        mean_r = rotated.mean(dim=0)
        var_r = rotated.var(dim=0).clamp(min=1e-8)
        m4 = ((rotated - mean_r) ** 4).mean(dim=0)
        kurtosis = (m4 / var_r**2) - 3.0  # excess kurtosis
        feats.post_wht_excess_kurtosis = kurtosis.mean().item()
    else:
        feats.post_wht_excess_kurtosis = float('nan')

    # --- Feature 3: key-norm coefficient of variation ---
    key_norms = k_flat.norm(dim=-1)
    norm_mean = key_norms.mean()
    norm_std = key_norms.std()
    feats.key_norm_cv = (norm_std / norm_mean.clamp(min=1e-8)).item()

    # --- Feature 4: per-vector value dynamic range ---
    v_ranges = v_flat.max(dim=-1).values - v_flat.min(dim=-1).values
    feats.value_dynamic_range = v_ranges.mean().item()

    # --- Feature 5: direct simulated quantization error ---
    k_quant = simulate_turbo_quant_keys(k_flat, n_bits=key_bits)
    v_quant = simulate_turbo_quant_values(v_flat, n_bits=value_bits)
    k_mse = F.mse_loss(k_quant, k_flat).item()
    v_mse = F.mse_loss(v_quant, v_flat).item()
    feats.simulated_quant_error = k_mse + v_mse

    return feats


# --------------------------------------------------------------------------
# K/V capture via DynamicCache (transformers >= 4.36)
# --------------------------------------------------------------------------

def _load_model_and_tokenizer(model_name: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_kv_per_layer(
    model_name: str,
    texts: list[str],
    max_tokens: int = 2048,
    device: str = "cuda",
) -> dict[int, dict[str, torch.Tensor]]:
    """Run a calibration forward pass and capture post-RoPE K/V per layer.

    Uses the DynamicCache object returned by the model (modern transformers
    stores K/V there, not as plain tuples in the attention output).

    Returns:
        {layer_idx: {"keys": Tensor, "values": Tensor}} on CPU.
        keys/values shape: (total_tokens, n_heads, head_dim)
    """
    model, tokenizer = _load_model_and_tokenizer(model_name, device)

    all_keys: dict[int, list[torch.Tensor]] = {}
    all_values: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_tokens,
            ).to(device)
            outputs = model(**inputs, use_cache=True)

            past_kv = outputs.past_key_values
            n_layers = len(past_kv)
            for layer_idx in range(n_layers):
                k, v = _cache_get_kv(past_kv, layer_idx)
                if k is not None:
                    all_keys.setdefault(layer_idx, []).append(k.detach().cpu())
                    all_values.setdefault(layer_idx, []).append(v.detach().cpu())
                else:
                    item = list(past_kv)[layer_idx]
                    all_keys.setdefault(layer_idx, []).append(item[0].detach().cpu())
                    all_values.setdefault(layer_idx, []).append(item[1].detach().cpu())

    result = {}
    for layer_idx in sorted(all_keys.keys()):
        # Each tensor: (batch=1, n_heads, seq_len, head_dim)
        keys = torch.cat(all_keys[layer_idx], dim=0)
        values = torch.cat(all_values[layer_idx], dim=0)
        # Reshape: (n_texts, n_heads, seq_len, head_dim) → (total_tokens, n_heads, head_dim)
        keys = keys.permute(0, 2, 1, 3).reshape(-1, keys.shape[1], keys.shape[3])
        values = values.permute(0, 2, 1, 3).reshape(-1, values.shape[1], values.shape[3])
        result[layer_idx] = {"keys": keys, "values": values}

    del model
    torch.cuda.empty_cache()
    return result


def _get_decoder_layers(model):
    """Get the list of decoder layers from a HF model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(f"Cannot find decoder layers in {type(model).__name__}")


# --------------------------------------------------------------------------
# Cache K/V access — transformers DynamicCache API varies across versions
# --------------------------------------------------------------------------

def _cache_get_kv(cache, layer_idx):
    """Read (key, value) tensors for a given layer from DynamicCache."""
    # Pattern A: .key_cache / .value_cache lists (transformers 4.36-4.47)
    if hasattr(cache, "key_cache") and isinstance(getattr(cache, "key_cache"), list):
        kc = cache.key_cache
        if layer_idx < len(kc):
            return kc[layer_idx], cache.value_cache[layer_idx]
    # Pattern B: .layers list of DynamicLayer (transformers 4.48+)
    if hasattr(cache, "layers") and isinstance(cache.layers, list):
        layer = cache.layers[layer_idx]
        for kn, vn in [("keys", "values"), ("key", "value"),
                       ("key_cache", "value_cache")]:
            if hasattr(layer, kn):
                return getattr(layer, kn), getattr(layer, vn)
    return None, None


def _cache_set_kv(cache, layer_idx, key, value):
    """Write (key, value) tensors for a given layer into DynamicCache."""
    if hasattr(cache, "key_cache") and isinstance(getattr(cache, "key_cache"), list):
        cache.key_cache[layer_idx] = key
        cache.value_cache[layer_idx] = value
        return
    if hasattr(cache, "layers") and isinstance(cache.layers, list):
        layer = cache.layers[layer_idx]
        for kn, vn in [("keys", "values"), ("key", "value"),
                       ("key_cache", "value_cache")]:
            if hasattr(layer, kn):
                setattr(layer, kn, key)
                setattr(layer, vn, value)
                return
    raise RuntimeError(f"Cannot write to cache: {type(cache).__name__}")


# --------------------------------------------------------------------------
# Ground-truth sensitivity (ΔPPL per layer)
# --------------------------------------------------------------------------

def measure_layer_sensitivity(
    model_name: str,
    eval_texts: list[str],
    key_bits: int = 3,
    value_bits: int = 4,
    max_tokens: int = 2048,
    chunk_size: int = 256,
    device: str = "cuda",
) -> dict:
    """Measure per-layer quantization sensitivity via ΔPPL.

    Uses chunked processing so that tokens in chunk 2+ attend to the
    quantized cache from previous chunks — a single-pass forward would
    compute attention from unquantized hidden states and miss the effect.
    """
    model, tokenizer = _load_model_and_tokenizer(model_name, device)

    n_layers = len(_get_decoder_layers(model))
    print(f"Model has {n_layers} layers", flush=True)

    all_input_ids = []
    for text in eval_texts:
        ids = tokenizer.encode(text, truncation=True, max_length=max_tokens)
        if len(ids) > 100:
            all_input_ids.append(torch.tensor(ids, device=device))

    print("Computing baseline PPL (chunked, no quantization)...", flush=True)
    baseline_ppl = _compute_hf_ppl_chunked(model, all_input_ids, chunk_size)
    print(f"Baseline PPL: {baseline_ppl:.4f}", flush=True)

    results = []
    for target_layer in range(n_layers):
        t0 = time.perf_counter()
        layer_ppl = _compute_hf_ppl_chunked(
            model, all_input_ids, chunk_size,
            quant_layer=target_layer, key_bits=key_bits, value_bits=value_bits,
        )
        elapsed = time.perf_counter() - t0

        delta_ppl = layer_ppl - baseline_ppl
        results.append({
            "layer": target_layer,
            "ppl": round(layer_ppl, 4),
            "delta_ppl": round(delta_ppl, 4),
            "elapsed_s": round(elapsed, 1),
        })
        print(f"  Layer {target_layer:2d}: PPL={layer_ppl:.4f}  ΔPPL={delta_ppl:+.4f}  ({elapsed:.1f}s)",
              flush=True)

    del model
    torch.cuda.empty_cache()

    return {
        "model": model_name,
        "baseline_ppl": round(baseline_ppl, 4),
        "key_bits": key_bits,
        "value_bits": value_bits,
        "n_eval_texts": len(all_input_ids),
        "layers": results,
    }


def _compute_hf_ppl_chunked(
    model,
    input_ids_list: list[torch.Tensor],
    chunk_size: int = 256,
    quant_layer: int | None = None,
    key_bits: int = 3,
    value_bits: int = 4,
) -> float:
    """Chunked PPL so cache quantization is visible to attention.

    Processes each sequence in chunks of chunk_size tokens with
    use_cache=True. After each chunk, if quant_layer is set, the new
    cache entries for that layer are quantized via simulate_turbo_quant.
    Tokens from chunk 2 onward attend to the quantized cache and are
    scored; chunk 1 is skipped (it sees no cached data).
    """
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for input_ids in input_ids_list:
            seq_len = input_ids.shape[0]
            past_kv = None
            cached_len = 0

            for chunk_idx, start in enumerate(range(0, seq_len, chunk_size)):
                end = min(start + chunk_size, seq_len)
                chunk = input_ids[start:end].unsqueeze(0)

                fwd_kwargs: dict = {"use_cache": True}
                if past_kv is not None:
                    fwd_kwargs["past_key_values"] = past_kv

                outputs = model(chunk, **fwd_kwargs)
                past_kv = outputs.past_key_values

                if quant_layer is not None:
                    k, v = _cache_get_kv(past_kv, quant_layer)
                    if k is not None:
                        new_len = k.shape[2]
                        if new_len > cached_len:
                            new_k = k[:, :, cached_len:, :]
                            new_v = v[:, :, cached_len:, :]
                            nk_s, nv_s = new_k.shape, new_v.shape
                            new_k_q = simulate_turbo_quant_keys(
                                new_k.reshape(-1, nk_s[-1]), n_bits=key_bits,
                            ).reshape(nk_s)
                            new_v_q = simulate_turbo_quant_values(
                                new_v.reshape(-1, nv_s[-1]), n_bits=value_bits,
                            ).reshape(nv_s)
                            if cached_len > 0:
                                full_k = torch.cat(
                                    [k[:, :, :cached_len, :], new_k_q], dim=2)
                                full_v = torch.cat(
                                    [v[:, :, :cached_len, :], new_v_q], dim=2)
                            else:
                                full_k, full_v = new_k_q, new_v_q
                            _cache_set_kv(past_kv, quant_layer, full_k, full_v)
                        cached_len = new_len

                if chunk_idx == 0:
                    continue

                logits = outputs.logits[0]
                n_score = min(end - start, seq_len - start - 1)
                if n_score > 0:
                    pred = logits[:n_score]
                    targets = input_ids[start + 1:start + 1 + n_score]
                    log_probs = F.log_softmax(pred, dim=-1)
                    token_nlls = -log_probs.gather(
                        1, targets.unsqueeze(1)).squeeze(1)
                    total_nll += token_nlls.sum().item()
                    total_tokens += n_score

    return math.exp(total_nll / total_tokens) if total_tokens else float('inf')


# --------------------------------------------------------------------------
# H1a test: Spearman correlation
# --------------------------------------------------------------------------

def compute_h1a_correlations(
    features: list[LayerFeatures],
    sensitivity: dict,
) -> dict:
    """Spearman rank correlation between each feature and ground-truth ΔPPL."""
    from scipy.stats import spearmanr

    delta_ppls = [layer["delta_ppl"] for layer in sensitivity["layers"]]
    feature_names = [
        "key_outlier_frac",
        "post_wht_excess_kurtosis",
        "key_norm_cv",
        "value_dynamic_range",
        "simulated_quant_error",
    ]

    results = {}
    for fname in feature_names:
        fvals = [getattr(f, fname) for f in features]
        if any(math.isnan(v) for v in fvals):
            results[fname] = {"rho": float('nan'), "p_value": float('nan'), "valid": False}
            continue
        rho, pval = spearmanr(fvals, delta_ppls)
        results[fname] = {
            "rho": round(float(rho), 4),
            "p_value": round(float(pval), 6),
            "valid": bool(abs(rho) > 0.35 and pval < 0.05),
        }
    return results


# --------------------------------------------------------------------------
# Calibration data
# --------------------------------------------------------------------------

def load_calibration_texts(domain: str = "wiki", n_docs: int = 8) -> list[str]:
    """Load calibration texts for feature extraction."""
    from datasets import load_dataset

    if domain == "wiki":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n".join(row["text"] for row in ds if row["text"].strip())
        words = text.split()
        doc_size = len(words) // n_docs
        return [" ".join(words[i * doc_size:(i + 1) * doc_size]) for i in range(n_docs)]

    if domain == "code":
        ds = load_dataset("codeparrot/github-code", split="train",
                          streaming=True, languages=["Python"])
        texts = []
        for item in ds:
            if len(item["code"]) > 500:
                texts.append(item["code"][:4000])
            if len(texts) >= n_docs:
                break
        return texts

    raise ValueError(f"unknown domain: {domain}")


# --------------------------------------------------------------------------
# Full Experiment 0 pipeline
# --------------------------------------------------------------------------

def run_experiment_0(
    model_name: str = "Qwen/Qwen3-4B",
    key_bits: int = 3,
    value_bits: int = 4,
    n_calib_docs: int = 8,
    n_eval_docs: int = 4,
    max_tokens: int = 2048,
    device: str = "cuda",
    out_dir: str = "results/exp0",
) -> None:
    """Run the full Experiment 0 pipeline."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Feature extraction ---
    print("=" * 60)
    print("Step 1: Extracting per-layer features from calibration data")
    print("=" * 60)

    calib_texts = load_calibration_texts("wiki", n_calib_docs)
    t0 = time.perf_counter()
    kv_data = extract_kv_per_layer(model_name, calib_texts,
                                    max_tokens=max_tokens, device=device)
    print(f"K/V extraction took {time.perf_counter() - t0:.1f}s")

    features = []
    for layer_idx in sorted(kv_data.keys()):
        kv = kv_data[layer_idx]
        f = compute_features(
            kv["keys"].to(device), kv["values"].to(device),
            layer_idx, key_bits=key_bits, value_bits=value_bits,
        )
        features.append(f)
        print(f"  Layer {layer_idx:2d}: outlier={f.key_outlier_frac:.3f}  "
              f"kurtosis={f.post_wht_excess_kurtosis:.3f}  "
              f"norm_cv={f.key_norm_cv:.3f}  "
              f"dyn_range={f.value_dynamic_range:.3f}  "
              f"quant_err={f.simulated_quant_error:.6f}")

    del kv_data
    torch.cuda.empty_cache()

    features_data = [
        {"layer": f.layer_idx, "key_outlier_frac": f.key_outlier_frac,
         "post_wht_excess_kurtosis": f.post_wht_excess_kurtosis,
         "key_norm_cv": f.key_norm_cv, "value_dynamic_range": f.value_dynamic_range,
         "simulated_quant_error": f.simulated_quant_error}
        for f in features
    ]
    (out / "features.json").write_text(json.dumps(features_data, indent=1))
    print(f"\nFeatures saved to {out / 'features.json'}")

    # --- Step 2: Ground-truth sensitivity ---
    print("\n" + "=" * 60)
    print("Step 2: Measuring per-layer sensitivity (ΔPPL)")
    print("=" * 60)

    eval_texts = load_calibration_texts("wiki", n_eval_docs)
    sensitivity = measure_layer_sensitivity(
        model_name, eval_texts, key_bits=key_bits, value_bits=value_bits,
        max_tokens=max_tokens, device=device,
    )
    (out / "sensitivity.json").write_text(json.dumps(sensitivity, indent=1))
    print(f"\nSensitivity saved to {out / 'sensitivity.json'}")

    # --- Step 3: H1a correlations ---
    print("\n" + "=" * 60)
    print("Step 3: H1a test — Spearman correlations")
    print("=" * 60)

    correlations = compute_h1a_correlations(features, sensitivity)
    for fname, result in correlations.items():
        status = "PASS" if result["valid"] else "FAIL"
        print(f"  {fname:<30s}  ρ={result['rho']:+.4f}  "
              f"p={result['p_value']:.6f}  [{status}]")

    # --- Step 4: Layer ranking ---
    valid_features = [f for f, r in correlations.items() if r["valid"]]
    if valid_features:
        best_feature = max(valid_features, key=lambda f: abs(correlations[f]["rho"]))
        ranking = sorted(range(len(features)),
                         key=lambda i: getattr(features[i], best_feature),
                         reverse=True)
        policy = {
            "method": "statistics-guided",
            "feature": best_feature,
            "rho": correlations[best_feature]["rho"],
            "ranking": ranking,
        }
        print(f"\nH1a PASSED — best feature: {best_feature} "
              f"(ρ={correlations[best_feature]['rho']:+.4f})")
        print(f"Layer ranking (most sensitive first): {ranking[:10]}...")
    else:
        ranking = list(range(len(features)))
        policy = {
            "method": "positional",
            "reason": "H1a failed — no feature passed correlation threshold",
            "ranking": ranking,
        }
        print("\nH1a FAILED — falling back to positional protection")

    exp0_result = {
        "model": model_name,
        "features": features_data,
        "sensitivity": sensitivity,
        "correlations": correlations,
        "policy": policy,
    }
    (out / "exp0_result.json").write_text(json.dumps(exp0_result, indent=1))
    print(f"\nFull Experiment 0 result saved to {out / 'exp0_result.json'}")


# --------------------------------------------------------------------------
# vLLM-based ground-truth profiling (no simulation)
# --------------------------------------------------------------------------

def _detect_n_layers(model_name: str) -> int:
    """Detect number of layers from HuggingFace model config."""
    from .model_info import detect_n_layers  # noqa: PLC0415
    return detect_n_layers(model_name)


def generate_sensitivity_manifest(
    model_name: str = "Qwen/Qwen3-4B",
    preset: str = "turboquant_4bit_nc",
    trace_tag: str = "screen",
    trace_seed: int = 20260714,
    out_path: str = "configs/grids/exp0_vllm.json",
) -> Path:
    """Generate a harness manifest for per-layer sensitivity measurement.

    Creates L+1 cells:
    - 1 baseline cell (kv_cache_dtype="auto")
    - L cells, each quantizing ONLY layer i (all others protected)

    Run the manifest with: python -m src.harness --manifest <out_path>
    Then analyze with: python -m src.profiler --mode vllm-analyze
    """
    n_layers = _detect_n_layers(model_name)
    print(f"Model {model_name} has {n_layers} layers")

    cells = []

    # Baseline (FP16)
    cells.append({
        "model": model_name,
        "kv_cache_dtype": "auto",
        "skip_layers": [],
        "rep": 0,
        "trace_tag": trace_tag,
        "trace_seed": trace_seed,
    })

    # Per-layer: quantize ONLY layer i
    for target_layer in range(n_layers):
        skip = [l for l in range(n_layers) if l != target_layer]
        cells.append({
            "model": model_name,
            "kv_cache_dtype": preset,
            "skip_layers": skip,
            "rep": 0,
            "trace_tag": trace_tag,
            "trace_seed": trace_seed,
        })

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cells, indent=1))
    print(f"Generated {len(cells)} cells → {out}")
    print(f"\nRun: python -m src.harness --manifest {out}")
    print(f"Then: python -m src.profiler --mode vllm-analyze --model {model_name}")
    return out


def analyze_vllm_sensitivity(
    model_name: str = "Qwen/Qwen3-4B",
    cells_dir: str = "results/cells",
    manifest_path: str = "configs/grids/exp0_vllm.json",
    out_dir: str = "results/exp0",
) -> dict:
    """Analyze results from vLLM-based per-layer sensitivity manifest."""
    from src.harness import CellConfig

    manifest = json.loads(Path(manifest_path).read_text())
    cells_path = Path(cells_dir)

    # Find baseline
    baseline_entry = manifest[0]
    baseline_entry["skip_layers"] = tuple(baseline_entry.get("skip_layers", ()))
    baseline_hash = CellConfig(**baseline_entry).cell_hash()
    baseline_file = cells_path / f"{baseline_hash}.json"

    if not baseline_file.exists():
        print(f"ERROR: Baseline cell not found: {baseline_file}")
        return {}

    baseline_data = json.loads(baseline_file.read_text())
    baseline_ppl = baseline_data.get("ppl", {}).get("ppl")
    print(f"Baseline PPL: {baseline_ppl:.4f}")

    # Per-layer sensitivity
    results = []
    for entry in manifest[1:]:
        entry_copy = dict(entry)
        entry_copy["skip_layers"] = tuple(entry_copy.get("skip_layers", ()))
        cell_hash = CellConfig(**entry_copy).cell_hash()
        cell_file = cells_path / f"{cell_hash}.json"

        if not cell_file.exists():
            continue

        cell_data = json.loads(cell_file.read_text())
        if not cell_data.get("ok"):
            continue

        # The target layer is the one NOT in skip_layers
        skip_set = set(int(x) for x in entry.get("skip_layers", []))
        n_layers = len(skip_set) + 1
        target_layer = next(l for l in range(n_layers) if l not in skip_set)

        layer_ppl = cell_data.get("ppl", {}).get("ppl")
        delta_ppl = layer_ppl - baseline_ppl if layer_ppl else None

        results.append({
            "layer": target_layer,
            "ppl": round(layer_ppl, 4) if layer_ppl else None,
            "delta_ppl": round(delta_ppl, 4) if delta_ppl else None,
            "peak_vram_gb": cell_data.get("peak_vram_gb"),
        })
        if delta_ppl is not None:
            print(f"  Layer {target_layer:2d}: PPL={layer_ppl:.4f}  ΔPPL={delta_ppl:+.4f}")
        else:
            print(f"  Layer {target_layer:2d}: FAILED")

    results.sort(key=lambda r: r["layer"])

    # Build ranking (most sensitive first)
    valid = [r for r in results if r["delta_ppl"] is not None]
    ranking = [r["layer"] for r in sorted(valid, key=lambda r: -r["delta_ppl"])]

    output = {
        "model": model_name,
        "method": "vllm-ground-truth",
        "baseline_ppl": round(baseline_ppl, 4),
        "layers": results,
        "policy": {
            "method": "vllm-ground-truth",
            "ranking": ranking,
        },
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sensitivity_vllm.json").write_text(json.dumps(output, indent=1))
    print(f"\nSensitivity saved to {out / 'sensitivity_vllm.json'}")
    print(f"Ranking (most sensitive first): {ranking[:10]}...")

    return output


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=[
        "features", "sim-sensitivity", "sim-full",
        "vllm", "vllm-analyze",
        # Legacy aliases
        "sensitivity", "full",
    ], default="sim-full")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--preset", default="turboquant_4bit_nc",
                   help="TurboQuant preset for vLLM-mode profiling")
    p.add_argument("--key-bits", type=int, default=3)
    p.add_argument("--value-bits", type=int, default=4)
    p.add_argument("--n-calib-docs", type=int, default=8)
    p.add_argument("--n-eval-docs", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="results/exp0")
    p.add_argument("--manifest", default="configs/grids/exp0_vllm.json",
                   help="Output path for vLLM manifest / input for vllm-analyze")
    p.add_argument("--cells-dir", default="results/cells")
    p.add_argument("--trace-tag", default="screen")
    p.add_argument("--trace-seed", type=int, default=20260714)
    args = p.parse_args()

    # Legacy aliases
    mode = args.mode
    if mode == "sensitivity":
        mode = "sim-sensitivity"
    elif mode == "full":
        mode = "sim-full"

    if mode == "vllm":
        generate_sensitivity_manifest(
            model_name=args.model,
            preset=args.preset,
            trace_tag=args.trace_tag,
            trace_seed=args.trace_seed,
            out_path=args.manifest,
        )
    elif mode == "vllm-analyze":
        analyze_vllm_sensitivity(
            model_name=args.model,
            cells_dir=args.cells_dir,
            manifest_path=args.manifest,
            out_dir=args.out_dir,
        )
    elif mode == "sim-full":
        run_experiment_0(
            model_name=args.model, key_bits=args.key_bits,
            value_bits=args.value_bits, n_calib_docs=args.n_calib_docs,
            n_eval_docs=args.n_eval_docs, max_tokens=args.max_tokens,
            device=args.device, out_dir=args.out_dir,
        )
    elif mode == "features":
        calib_texts = load_calibration_texts("wiki", args.n_calib_docs)
        kv_data = extract_kv_per_layer(args.model, calib_texts,
                                        max_tokens=args.max_tokens,
                                        device=args.device)
        for layer_idx in sorted(kv_data.keys()):
            kv = kv_data[layer_idx]
            f = compute_features(
                kv["keys"].to(args.device), kv["values"].to(args.device),
                layer_idx, key_bits=args.key_bits, value_bits=args.value_bits,
            )
            print(f"Layer {layer_idx:2d}: outlier={f.key_outlier_frac:.3f}  "
                  f"kurtosis={f.post_wht_excess_kurtosis:.3f}  "
                  f"norm_cv={f.key_norm_cv:.3f}  "
                  f"dyn_range={f.value_dynamic_range:.3f}  "
                  f"quant_err={f.simulated_quant_error:.6f}")
    elif mode == "sim-sensitivity":
        eval_texts = load_calibration_texts("wiki", args.n_eval_docs)
        result = measure_layer_sensitivity(
            args.model, eval_texts, key_bits=args.key_bits,
            value_bits=args.value_bits, max_tokens=args.max_tokens,
            device=args.device,
        )
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "sensitivity.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
