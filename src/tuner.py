"""Workload-aware tuner for KV cache compression configuration.

Searches the (preset × layer-protection budget) space to find the
configuration that maximizes a workload-specific utility function
while staying within a perplexity degradation threshold. The utility
functions and thresholds are loaded from ``configs/profiles.json``
and can be overridden per-deployment.

Uses Experiment 1 cells as warm-start data and Optuna TPE for
refinement search.

CLI:
    python -m src.tuner --analyze              # utilities from existing cells
    python -m src.tuner --suggest 16           # generate refinement manifest
    python -m src.tuner --optimize             # full analysis + optimal configs
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from .model_info import detect_n_layers, floor_layers_for

PRESETS = ["k8v4", "4bit_nc", "k3v4_nc", "3bit_nc"]
PRESET_FULL = {p: f"turboquant_{p}" for p in PRESETS}

BYTES_PER_DIM = {
    "auto": 4.0,
    "k8v4": 1.5,
    "4bit_nc": 1.0,
    "k3v4_nc": 0.875,
    "3bit_nc": 0.75,
}

from .profiles import load_profiles

PROFILES_CFG = load_profiles()

# ---------------------------------------------------------------------------
# Layer ranking (loaded from Exp 0 or fallback)
# ---------------------------------------------------------------------------

def load_profiler_ranking(floor_layers: frozenset[int], n_layers: int,
                          exp0_path: Path = Path("results/exp0/exp0_result.json"),
                          ) -> list[int]:
    """Layers ranked by per-layer sensitivity, most sensitive first.

    Falls back to positional order (works for any model/layer count) when
    no Exp-0 ranking has been computed yet for this model.
    """
    if exp0_path.exists():
        data = json.loads(exp0_path.read_text())
        full_ranking = data["policy"]["ranking"]
        return [l for l in full_ranking if l not in floor_layers]
    return [l for l in range(n_layers) if l not in floor_layers]


def skip_layers_for_budget(ranking: list[int], budget: int) -> list[int]:
    return sorted(ranking[:budget])


# ---------------------------------------------------------------------------
# Memory ratio (analytical)
# ---------------------------------------------------------------------------

def compute_r_mem(preset: str, n_protect: int, n_layers: int, n_floor: int) -> float:
    s = n_floor + n_protect
    base = n_layers * BYTES_PER_DIM["auto"]
    cfg = s * BYTES_PER_DIM["auto"] + (n_layers - s) * BYTES_PER_DIM[preset]
    return base / cfg


# ---------------------------------------------------------------------------
# Load cell results
# ---------------------------------------------------------------------------

@dataclass
class CellMetrics:
    preset: str
    n_protect: int
    skip_layers: list[int]
    rep: int
    ppl: float | None
    chat_tpot_ms: float | None
    chat_ttft_ms: float | None
    chat_tps: float | None
    rag_ttft_ms: float | None
    rag_tps: float | None
    needle_acc: float | None
    batch_tps: float | None


def _classify_protection(skip: list, ranking: list[int]) -> tuple[str, int]:
    skip_set = set(int(x) for x in skip)
    if not skip_set:
        return "stats", 0
    for budget in range(1, len(ranking) + 1):
        if skip_set == set(ranking[:budget]):
            return "stats", budget
    return "positional", len(skip_set)


def load_cells(cells_dir: Path, ranking: list[int],
               include_positional: bool = False) -> list[CellMetrics]:
    results = []
    for f in sorted(cells_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if not data.get("ok"):
            continue
        cfg = data["config"]
        dtype = cfg["kv_cache_dtype"]
        if dtype == "auto":
            continue
        preset = dtype.replace("turboquant_", "")
        skip = cfg.get("skip_layers", [])
        method, n_protect = _classify_protection(skip, ranking)
        if method == "positional" and not include_positional:
            continue

        chat = data.get("profiles", {}).get("chat", {})
        rag = data.get("profiles", {}).get("rag", {})
        batch = data.get("profiles", {}).get("batch", {})

        def _mean(profile_data, metric):
            v = profile_data.get(metric, {})
            if isinstance(v, dict):
                return v.get("mean")
            return v

        results.append(CellMetrics(
            preset=preset,
            n_protect=n_protect,
            skip_layers=[int(x) for x in skip],
            rep=cfg["rep"],
            ppl=data.get("ppl", {}).get("ppl"),
            chat_tpot_ms=(_mean(chat, "tpot_s") or 0) * 1000 or None,
            chat_ttft_ms=(_mean(chat, "ttft_s") or 0) * 1000 or None,
            chat_tps=chat.get("throughput_tok_s"),
            rag_ttft_ms=(_mean(rag, "ttft_s") or 0) * 1000 or None,
            rag_tps=rag.get("throughput_tok_s"),
            needle_acc=rag.get("needle_accuracy"),
            batch_tps=batch.get("throughput_tok_s"),
        ))
    return results


def load_baseline(cells_dir: Path) -> dict:
    baselines = []
    for f in sorted(cells_dir.glob("*.json")):
        data = json.loads(f.read_text())
        if not data.get("ok"):
            continue
        if data["config"]["kv_cache_dtype"] != "auto":
            continue
        chat = data.get("profiles", {}).get("chat", {})
        rag = data.get("profiles", {}).get("rag", {})
        batch = data.get("profiles", {}).get("batch", {})

        def _mean(profile_data, metric):
            v = profile_data.get(metric, {})
            if isinstance(v, dict):
                return v.get("mean")
            return v

        baselines.append({
            "ppl": data.get("ppl", {}).get("ppl"),
            "chat_tpot_ms": (_mean(chat, "tpot_s") or 0) * 1000,
            "chat_ttft_ms": (_mean(chat, "ttft_s") or 0) * 1000,
            "rag_ttft_ms": (_mean(rag, "ttft_s") or 0) * 1000,
            "batch_tps": batch.get("throughput_tok_s"),
        })
    if not baselines:
        raise RuntimeError("No baseline (auto) cells found")
    avg = {}
    for key in baselines[0]:
        vals = [b[key] for b in baselines if b[key] is not None]
        avg[key] = sum(vals) / len(vals) if vals else None
    return avg


# ---------------------------------------------------------------------------
# Utility computation
# ---------------------------------------------------------------------------

def compute_utility(profile_name: str, metrics: dict, baseline: dict,
                    preset: str, n_protect: int, n_layers: int, n_floor: int) -> float:
    pcfg = PROFILES_CFG[profile_name]
    if metrics["ppl"] is None or baseline["ppl"] is None:
        return 0.0
    dppl_rel = (metrics["ppl"] - baseline["ppl"]) / baseline["ppl"]
    if dppl_rel > pcfg["ppl_threshold"]:
        return 0.0

    r_mem = compute_r_mem(preset, n_protect, n_layers, n_floor)
    # `is not None` (not truthiness): a literal 0.0 is a real, terrible
    # measurement (e.g. zero throughput) and must not be mistaken for
    # "not measured for this cell" and default to a neutral 1.0 gain.
    s_tpot = (baseline["chat_tpot_ms"] / metrics["chat_tpot_ms"]
              if metrics.get("chat_tpot_ms") is not None else 1.0)
    s_ttft = (baseline["rag_ttft_ms"] / metrics["rag_ttft_ms"]
              if metrics.get("rag_ttft_ms") is not None else 1.0)
    s_tp = (metrics["batch_tps"] / baseline["batch_tps"]
            if metrics.get("batch_tps") is not None and baseline.get("batch_tps")
            else 1.0)

    gains = {"s_tpot": s_tpot, "s_ttft": s_ttft, "s_tp": s_tp, "r_mem": r_mem}

    utility = 1.0
    for term, weight in pcfg["terms"]:
        val = gains[term]
        if val <= 0:
            return 0.0
        utility *= val ** weight
    return utility


# ---------------------------------------------------------------------------
# Average cells across reps
# ---------------------------------------------------------------------------

def average_cells(cells: list[CellMetrics]) -> list[dict]:
    key_fn = lambda c: (c.preset, c.n_protect)
    cells_sorted = sorted(cells, key=key_fn)
    averaged = []
    for (preset, n_protect), group in groupby(cells_sorted, key=key_fn):
        grp = list(group)
        def avg(field):
            vals = [getattr(c, field) for c in grp if getattr(c, field) is not None]
            return sum(vals) / len(vals) if vals else None
        averaged.append({
            "preset": preset,
            "n_protect": n_protect,
            "skip_layers": grp[0].skip_layers,
            "n_reps": len(grp),
            "ppl": avg("ppl"),
            "chat_tpot_ms": avg("chat_tpot_ms"),
            "chat_ttft_ms": avg("chat_ttft_ms"),
            "chat_tps": avg("chat_tps"),
            "rag_ttft_ms": avg("rag_ttft_ms"),
            "rag_tps": avg("rag_tps"),
            "needle_acc": avg("needle_acc"),
            "batch_tps": avg("batch_tps"),
        })
    return averaged


# ---------------------------------------------------------------------------
# Analyze mode
# ---------------------------------------------------------------------------

def fmt(v, d=3):
    return f"{v:.{d}f}" if v is not None else "N/A"


def analyze(cells_dir: Path, ranking: list[int], n_layers: int, n_floor: int):
    cells = load_cells(cells_dir, ranking)
    baseline = load_baseline(cells_dir)
    averaged = average_cells(cells)

    print("=" * 100)
    print("EXPERIMENT 2 — UTILITY ANALYSIS")
    print("=" * 100)
    print(f"\nBaseline: PPL={fmt(baseline['ppl'])}, "
          f"chat_TPOT={fmt(baseline['chat_tpot_ms'], 2)}ms, "
          f"rag_TTFT={fmt(baseline['rag_ttft_ms'], 1)}ms, "
          f"batch_TPS={fmt(baseline['batch_tps'], 1)}")
    print(f"\nPPL thresholds — chat: {baseline['ppl']*1.005:.2f} (0.5%), "
          f"rag: {baseline['ppl']*1.01:.2f} (1%), "
          f"batch: {baseline['ppl']*1.02:.2f} (2%)")

    for profile in PROFILES_CFG:
        print(f"\n{'─' * 90}")
        print(f"  PROFILE: {profile.upper()}  "
              f"(threshold: {PROFILES_CFG[profile]['ppl_threshold']*100:.1f}% PPL, "
              f"terms: {PROFILES_CFG[profile]['terms']})")
        print(f"{'─' * 90}")
        print(f"  {'preset':<10} {'n_prot':>6} {'PPL':>7} {'dPPL%':>6} "
              f"{'R_mem':>5} {'utility':>8} {'viable':>6}")
        print(f"  {'-'*60}")

        rows = []
        for r in averaged:
            u = compute_utility(profile, r, baseline, r["preset"], r["n_protect"],
                                n_layers, n_floor)
            dppl = ((r["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100
                    if r["ppl"] and baseline["ppl"] else None)
            rmem = compute_r_mem(r["preset"], r["n_protect"], n_layers, n_floor)
            rows.append((r, u, dppl, rmem))

        rows.sort(key=lambda x: -x[1])
        for r, u, dppl, rmem in rows:
            viable = "YES" if u > 0 else "no"
            print(f"  {r['preset']:<10} {r['n_protect']:>6} {fmt(r['ppl']):>7} "
                  f"{fmt(dppl, 1):>5}% {rmem:>5.2f} {u:>8.4f} {viable:>6}")

        best = max(rows, key=lambda x: x[1])
        if best[1] > 0:
            print(f"\n  >>> BEST: {best[0]['preset']} n_protect={best[0]['n_protect']} "
                  f"utility={best[1]:.4f}")
        else:
            print(f"\n  >>> No viable config for {profile}")

    # --- Protection value analysis (ignoring PPL constraint) ---
    print(f"\n{'=' * 90}")
    print("PROTECTION VALUE ANALYSIS (no PPL constraint — isolating protection effect)")
    print(f"{'=' * 90}")
    print(f"\n  For each preset, how much does protection improve PPL and what does it cost?")
    print(f"  {'preset':<10} {'budget':>6} {'PPL':>7} {'ΔPPL_vs_floor':>14} "
          f"{'R_mem':>5} {'ΔR_mem':>7} {'needle':>6}")
    print(f"  {'-' * 70}")

    for preset in PRESETS:
        preset_rows = sorted(
            [r for r in averaged if r["preset"] == preset],
            key=lambda r: r["n_protect"],
        )
        floor_row = next((r for r in preset_rows if r["n_protect"] == 0), None)
        if not floor_row or not floor_row["ppl"]:
            continue
        for r in preset_rows:
            if not r["ppl"]:
                continue
            dppl = floor_row["ppl"] - r["ppl"]
            rmem_floor = compute_r_mem(preset, 0, n_layers, n_floor)
            rmem_this = compute_r_mem(preset, r["n_protect"], n_layers, n_floor)
            drmem = rmem_this - rmem_floor
            print(f"  {preset:<10} {r['n_protect']:>6} {fmt(r['ppl']):>7} "
                  f"{dppl:>+13.3f} {rmem_this:>5.2f} {drmem:>+6.2f}  "
                  f"{fmt(r.get('needle_acc'), 2):>6}")
        print()

    # --- Threshold proximity analysis ---
    print(f"{'=' * 90}")
    print("THRESHOLD PROXIMITY: configs closest to crossing a PPL threshold")
    print(f"{'=' * 90}")
    for profile, pcfg in PROFILES_CFG.items():
        threshold = baseline["ppl"] * (1 + pcfg["ppl_threshold"])
        near = []
        for r in averaged:
            if r["ppl"] is None:
                continue
            gap = r["ppl"] - threshold
            if 0 < gap < 0.3:
                near.append((r, gap))
        near.sort(key=lambda x: x[1])
        if near:
            print(f"\n  {profile.upper()} threshold: PPL ≤ {threshold:.2f}")
            for r, gap in near[:4]:
                print(f"    {r['preset']:<10} n_prot={r['n_protect']}  "
                      f"PPL={fmt(r['ppl'])}  gap={gap:+.3f}  "
                      f"← needs {gap:.3f} more PPL reduction to pass")
        else:
            print(f"\n  {profile.upper()}: no near-miss configs (all either pass or far away)")

    return averaged, baseline


# ---------------------------------------------------------------------------
# Suggest refinement trials
# ---------------------------------------------------------------------------

def suggest_refinements(cells_dir: Path, ranking: list[int],
                        n_trials: int = 16) -> list[dict]:
    cells = load_cells(cells_dir, ranking)
    baseline = load_baseline(cells_dir)
    averaged = average_cells(cells)
    existing = {(c.preset, c.n_protect) for c in cells}

    priority = []
    for preset in PRESETS:
        preset_rows = [r for r in averaged if r["preset"] == preset]
        floor_row = next((r for r in preset_rows if r["n_protect"] == 0), None)
        if not floor_row or not floor_row["ppl"]:
            continue
        for profile, pcfg in PROFILES_CFG.items():
            threshold = baseline["ppl"] * (1 + pcfg["ppl_threshold"])
            if floor_row["ppl"] > threshold:
                best_protected = min(
                    (r["ppl"] for r in preset_rows if r["n_protect"] > 0 and r["ppl"]),
                    default=floor_row["ppl"],
                )
                gap = best_protected - threshold
                if gap < 0.2:
                    for budget in range(1, min(9, len(ranking) + 1)):
                        if (preset, budget) not in existing:
                            priority.append((gap, preset, budget))

    priority.sort()

    all_candidates = []
    seen = set()
    for _, preset, budget in priority:
        if (preset, budget) not in seen:
            all_candidates.append((preset, budget))
            seen.add((preset, budget))

    for preset in PRESETS:
        for budget in range(0, min(9, len(ranking) + 1)):
            if (preset, budget) not in existing and (preset, budget) not in seen:
                all_candidates.append((preset, budget))
                seen.add((preset, budget))

    selected = all_candidates[:n_trials // 2]

    manifest = []
    for preset, budget in selected:
        skip = skip_layers_for_budget(ranking, budget)
        for rep in range(2):
            manifest.append({
                "kv_cache_dtype": PRESET_FULL[preset],
                "skip_layers": skip,
                "rep": rep,
            })

    print(f"Prioritized configs (threshold-crossing candidates first):")
    for preset, budget in selected:
        tag = " ← near threshold" if any(
            p == preset and b == budget for _, p, b in priority
        ) else ""
        print(f"  {preset} budget={budget} skip={skip_layers_for_budget(ranking, budget)}{tag}")

    return manifest


# ---------------------------------------------------------------------------
# Optimize: Optuna study + optimal configs
# ---------------------------------------------------------------------------

def optimize(cells_dir: Path, ranking: list[int], n_layers: int,
            floor_layers: frozenset[int], model_name: str):
    import optuna
    from optuna.distributions import CategoricalDistribution, IntDistribution

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_floor = len(floor_layers)

    cells = load_cells(cells_dir, ranking)
    baseline = load_baseline(cells_dir)
    averaged = average_cells(cells)

    for profile in PROFILES_CFG:
        print(f"\n{'=' * 70}")
        print(f"OPTUNA STUDY: {profile.upper()}")
        print(f"{'=' * 70}")

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"kv_tune_{profile}",
        )

        distributions = {
            "preset": CategoricalDistribution(PRESETS),
            "n_protect": IntDistribution(0, min(8, len(ranking))),
        }

        for r in averaged:
            u = compute_utility(profile, r, baseline, r["preset"], r["n_protect"],
                                n_layers, n_floor)
            trial = optuna.trial.create_trial(
                params={"preset": r["preset"], "n_protect": r["n_protect"]},
                distributions=distributions,
                values=[u],
            )
            study.add_trial(trial)

        print(f"  Warm-started with {len(study.trials)} trials")
        print(f"  Best so far: utility={study.best_value:.4f}, "
              f"params={study.best_params}")

        try:
            importance = optuna.importance.get_param_importances(study)
            print(f"\n  fANOVA parameter importance:")
            for param, imp in importance.items():
                print(f"    {param:<12} {imp:.3f} ({imp*100:.1f}%)")
        except Exception:
            print("  (fANOVA requires more trials)")

        # Find optimal
        best = study.best_trial
        print(f"\n  OPTIMAL for {profile}: preset={best.params['preset']}, "
              f"n_protect={best.params['n_protect']}, "
              f"utility={best.value:.4f}")

    # Build optimal_configs.json (keyed by model, so multiple models can
    # coexist in the same file — each `--model` run only touches its own key)
    def _config_entry(r: dict, u: float) -> dict:
        skip = skip_layers_for_budget(ranking, r["n_protect"])
        return {
            "kv_cache_dtype": PRESET_FULL[r["preset"]],
            "skip_layers": skip,
            "effective_skip_layers": sorted(set(skip) | floor_layers),
            "n_protected_total": n_floor + r["n_protect"],
            "utility": round(u, 4),
            "ppl": r["ppl"],
            "r_mem": round(compute_r_mem(r["preset"], r["n_protect"],
                                         n_layers, n_floor), 3),
            "ppl_delta_pct": round(
                (r["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100, 2
            ) if r["ppl"] else None,
            "evidence": {
                "chat_tpot_ms": r.get("chat_tpot_ms"),
                "rag_ttft_ms": r.get("rag_ttft_ms"),
                "batch_tps": r.get("batch_tps"),
                "needle_acc": r.get("needle_acc"),
            },
        }

    model_entry = {}
    for profile in PROFILES_CFG:
        scored = [(r, compute_utility(profile, r, baseline, r["preset"], r["n_protect"],
                                      n_layers, n_floor))
                  for r in averaged]
        viable = sorted((pair for pair in scored if pair[1] > 0), key=lambda x: -x[1])
        if viable:
            # alternatives[0] is the champion; the rest are ranked fallbacks
            # (e.g. for when the champion's preset is infeasible on some GPU).
            alternatives = [_config_entry(r, u) for r, u in viable[:5]]
            model_entry[profile] = {**alternatives[0], "alternatives": alternatives}
        else:
            model_entry[profile] = {"viable": False, "reason": "no config meets PPL constraint"}

    model_entry["baseline"] = {
        "ppl": baseline["ppl"],
        "chat_tpot_ms": baseline["chat_tpot_ms"],
        "rag_ttft_ms": baseline["rag_ttft_ms"],
        "batch_tps": baseline["batch_tps"],
    }

    out = Path("results/optimal_configs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    all_configs = json.loads(out.read_text()) if out.exists() else {}
    all_configs[model_name] = model_entry
    out.write_text(json.dumps(all_configs, indent=2))
    print(f"\nOptimal configs for {model_name} saved to {out}")

    configs = model_entry  # for the summary print below

    # Summary
    print(f"\n{'=' * 70}")
    print("OPTIMAL CONFIGURATION SUMMARY")
    print(f"{'=' * 70}")
    for profile in PROFILES_CFG:
        c = configs[profile]
        if c.get("viable") is False:
            print(f"  {profile:>6}: NO VIABLE CONFIG")
        else:
            print(f"  {profile:>6}: {c['kv_cache_dtype']:<25s} "
                  f"protect={c['n_protected_total']} layers  "
                  f"R_mem={c['r_mem']:.2f}x  "
                  f"dPPL={c['ppl_delta_pct']:+.2f}%  "
                  f"U={c['utility']:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells-dir", default="results/cells")
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="model whose layer count/floor to use (HF config lookup)")
    p.add_argument("--n-layers", type=int, default=None,
                   help="override detected layer count (skips the HF config lookup)")
    p.add_argument("--analyze", action="store_true",
                   help="Compute utilities from existing cells")
    p.add_argument("--suggest", type=int, metavar="N",
                   help="Generate N refinement trial configs as manifest")
    p.add_argument("--optimize", action="store_true",
                   help="Full Optuna run + output optimal_configs.json")
    args = p.parse_args()

    n_layers = args.n_layers if args.n_layers is not None else detect_n_layers(args.model)
    floor_layers = floor_layers_for(n_layers)
    n_floor = len(floor_layers)

    ranking = load_profiler_ranking(floor_layers, n_layers)
    cells_dir = Path(args.cells_dir)

    if args.analyze:
        analyze(cells_dir, ranking, n_layers, n_floor)
    elif args.suggest:
        manifest = suggest_refinements(cells_dir, ranking, args.suggest)
        out = Path("configs/grids/exp2_refinement.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2))
        print(f"Refinement manifest: {len(manifest)} cells → {out}")
        print(f"\nRun: python -m src.harness --manifest {out}")
    elif args.optimize:
        optimize(cells_dir, ranking, n_layers, floor_layers, args.model)
    else:
        analyze(cells_dir, ranking, n_layers, n_floor)
        print("\n\nUse --suggest N to generate refinement trials, "
              "or --optimize for full Optuna analysis.")


if __name__ == "__main__":
    main()
