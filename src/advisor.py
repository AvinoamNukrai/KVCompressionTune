"""Config Advisor — recommend vLLM KV cache compression settings.

Given a model, GPU, and workload profile (chat/rag/batch), outputs the
optimal TurboQuant preset and layer protection list, along with evidence
(PPL delta, utility score, latency) explaining the recommendation — and,
when a GPU is given, filters out presets that can't run on it (e.g. k8v4
requires SM >= 8.9 and fails to initialize on Ampere; INSIGHTS.md #6/#11).

Reads from ``results/optimal_configs.json`` produced by the tuner, which
is keyed by model name so multiple profiled models can coexist in one file.

CLI:
    python -m src.advisor --profile chat
    python -m src.advisor --profile rag --format python
    python -m src.advisor --all
    python -m src.advisor --all --model meta-llama/Llama-3.1-8B --gpu "RTX 3090"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_CONFIGS_PATH = Path("results/optimal_configs.json")

PROFILE_DESCRIPTIONS = {
    "chat": "Interactive chat — short input, long output, TPOT-sensitive (dPPL ≤ 0.5%)",
    "rag": "Retrieval-augmented generation — long input, short output, memory+TTFT (dPPL ≤ 1.0%)",
    "batch": "Bulk offline processing — throughput-maximizing (dPPL ≤ 2.0%)",
}

from .profiles import load_profiles

_PROFILES = load_profiles()
PPL_THRESHOLDS = {name: cfg["ppl_threshold"] * 100 for name, cfg in _PROFILES.items()}

# ---------------------------------------------------------------------------
# GPU feasibility (SPEC §3.3, INSIGHTS #6/#11: k8v4 needs native FP8 (SM>=8.9)
# and fails to initialize on Ampere; the _nc presets work everywhere tested)
# ---------------------------------------------------------------------------

GPU_SM = {
    "rtx 3090": 8.6, "rtx 4090": 8.9, "a100": 8.0, "h100": 9.0,
    "a6000": 8.6, "a10g": 8.6, "a10": 8.6, "l40s": 8.9, "l4": 8.9,
    "v100": 7.0, "t4": 7.5,
}

PRESET_MIN_SM = {"turboquant_k8v4": 8.9}


def resolve_sm(gpu_name: str) -> float | None:
    """Best-effort SM lookup by substring match; None if the GPU is unknown."""
    key = gpu_name.strip().lower()
    for name, sm in GPU_SM.items():
        if name in key:
            return sm
    return None


def is_feasible(kv_cache_dtype: str, sm: float | None) -> bool:
    """Whether a preset can run on a GPU of the given compute capability.

    An unknown GPU (sm=None) is treated as feasible — we simply have no
    basis to rule it out.
    """
    if sm is None:
        return True
    return sm >= PRESET_MIN_SM.get(kv_cache_dtype, 0.0)


def load_configs(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: {path} not found. Run the tuner first:", file=sys.stderr)
        print(f"  python -m src.tuner --optimize", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def resolve_model_configs(configs: dict, model: str) -> dict:
    """Look up the per-model entry, erroring out with the profiled-model list."""
    if model not in configs:
        available = sorted(configs)
        print(f"ERROR: no profiled config for model '{model}'.", file=sys.stderr)
        if available:
            print(f"  Profiled models: {available}", file=sys.stderr)
        print(f"  Run: python -m src.tuner --optimize --model {model!r}", file=sys.stderr)
        sys.exit(1)
    return configs[model]


def apply_gpu_filter(cfg: dict, gpu: str, sm: float | None) -> dict:
    """Swap in the best feasible alternative if the champion preset can't
    run on the given GPU; returns the (possibly substituted) config."""
    if cfg is None or cfg.get("viable") is False or sm is None:
        return cfg
    if is_feasible(cfg["kv_cache_dtype"], sm):
        return cfg

    fallback = next(
        (alt for alt in cfg.get("alternatives", []) if is_feasible(alt["kv_cache_dtype"], sm)),
        None,
    )
    min_sm = PRESET_MIN_SM.get(cfg["kv_cache_dtype"])
    if fallback is None:
        return {
            "viable": False,
            "reason": (f"{cfg['kv_cache_dtype']} requires SM>={min_sm} but {gpu} is "
                       f"SM {sm}, and no feasible alternative meets the PPL constraint"),
        }
    fallback = dict(fallback)
    fallback["gpu_fallback_note"] = (
        f"{cfg['kv_cache_dtype']} requires SM>={min_sm} but {gpu} is SM {sm} "
        f"— using next-best feasible config instead."
    )
    return fallback


def format_cli(cfg: dict, model: str) -> str:
    parts = [
        "vllm serve", model,
        f"--kv-cache-dtype {cfg['kv_cache_dtype']}",
    ]
    if cfg.get("skip_layers"):
        layers = " ".join(str(x) for x in cfg["skip_layers"])
        parts.append(f"--kv-cache-dtype-skip-layers {layers}")
    parts.append("--enable-chunked-prefill")
    return " \\\n    ".join(parts)


def format_python(cfg: dict, model: str) -> str:
    kwargs = [
        f'    model="{model}",',
        f'    kv_cache_dtype="{cfg["kv_cache_dtype"]}",',
    ]
    if cfg.get("skip_layers"):
        layers = [str(x) for x in cfg["skip_layers"]]
        kwargs.append(f"    kv_cache_dtype_skip_layers={layers},")
    kwargs.append("    enable_chunked_prefill=True,")
    return "LLM(\n" + "\n".join(kwargs) + "\n)"


def print_recommendation(profile: str, cfg: dict, model: str,
                         output_format: str) -> None:
    desc = PROFILE_DESCRIPTIONS.get(profile, profile)
    print(f"{'=' * 70}")
    print(f"RECOMMENDATION: {profile.upper()}")
    print(f"  {desc}")
    print(f"{'=' * 70}")

    if cfg.get("viable") is False:
        print(f"\n  No viable config found: {cfg.get('reason', 'unknown')}")
        print(f"  Recommendation: use FP16 baseline (--kv-cache-dtype auto)")
        return

    if cfg.get("gpu_fallback_note"):
        print(f"\n  ⚠ {cfg['gpu_fallback_note']}")

    print(f"\n  Preset:     {cfg['kv_cache_dtype']}")
    skip = cfg.get("skip_layers", [])
    eff = cfg.get("effective_skip_layers", [])
    print(f"  Extra skip: {skip if skip else '(none — floor only)'}")
    print(f"  Effective:  {eff} ({len(eff)} layers protected)")
    print(f"  R_mem:      {cfg.get('r_mem', '?')}x compression")

    print(f"\n  --- Evidence ---")
    ppl_d = cfg.get("ppl_delta_pct")
    threshold = PPL_THRESHOLDS.get(profile)
    if ppl_d is not None:
        status = "PASS" if abs(ppl_d) <= threshold else "FAIL"
        print(f"  PPL delta:  {ppl_d:+.2f}% (threshold: ≤{threshold}%) [{status}]")
    print(f"  Utility:    {cfg.get('utility', '?')}")

    ev = cfg.get("evidence", {})
    if profile == "chat" and ev.get("chat_tpot_ms"):
        print(f"  Chat TPOT:  {ev['chat_tpot_ms']:.2f} ms")
    if profile == "rag":
        if ev.get("rag_ttft_ms"):
            print(f"  RAG TTFT:   {ev['rag_ttft_ms']:.1f} ms")
        if ev.get("needle_acc") is not None:
            print(f"  Needle acc: {ev['needle_acc']:.2f}")
    if profile == "batch" and ev.get("batch_tps"):
        print(f"  Batch TPS:  {ev['batch_tps']:.1f} tok/s")

    print(f"\n  --- Launch command ---")
    if output_format == "cli":
        print(f"  {format_cli(cfg, model)}")
    else:
        print(f"  {format_python(cfg, model)}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=["chat", "rag", "batch"],
                   help="workload profile to recommend for")
    p.add_argument("--all", action="store_true",
                   help="show recommendations for all profiles")
    p.add_argument("--format", choices=["cli", "python"], default="cli",
                   dest="output_format",
                   help="output format: vLLM CLI flags or Python kwargs")
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="model to recommend for (must have been profiled by the tuner)")
    p.add_argument("--gpu", default=None,
                   help="target GPU (e.g. 'RTX 3090', 'A100') — filters out "
                        "presets that can't run on it (see GPU_SM in this module)")
    p.add_argument("--configs", default=str(DEFAULT_CONFIGS_PATH),
                   help="path to optimal_configs.json")
    p.add_argument("--json", action="store_true",
                   help="output raw JSON instead of formatted text")
    args = p.parse_args()

    if not args.profile and not args.all:
        p.error("specify --profile or --all")

    configs = load_configs(Path(args.configs))
    model_configs = resolve_model_configs(configs, args.model)
    profiles = ["chat", "rag", "batch"] if args.all else [args.profile]

    sm = None
    if args.gpu:
        sm = resolve_sm(args.gpu)
        if sm is None:
            print(f"Warning: unknown GPU '{args.gpu}' — skipping feasibility check "
                  f"(known: {sorted(GPU_SM)})", file=sys.stderr)

    resolved = {}
    for profile in profiles:
        cfg = model_configs.get(profile)
        if cfg is not None and args.gpu:
            cfg = apply_gpu_filter(cfg, args.gpu, sm)
        resolved[profile] = cfg

    if args.json:
        out = {pr: resolved[pr] for pr in profiles if pr in resolved}
        print(json.dumps(out, indent=2))
        return

    baseline = model_configs.get("baseline", {})
    if baseline:
        print(f"Model: {args.model}")
        print(f"Baseline PPL: {baseline.get('ppl', '?')}")
        print()

    for profile in profiles:
        cfg = resolved.get(profile)
        if cfg is None:
            print(f"No config found for profile '{profile}'")
            continue
        print_recommendation(profile, cfg, args.model, args.output_format)


if __name__ == "__main__":
    main()
