"""Experiment 3 — Validation: head-to-head statistical comparison.

Runs the full statistical protocol from SPEC §5.1:
- Mean ± 95% CI (Student's t, n-1 df) per metric per config
- Per-profile utility with paired comparisons
- Wilcoxon signed-rank + paired t-test on run-level scalars
- Cohen's d effect sizes
- Holm-Bonferroni correction across all tests

Usage:
    python -m analysis.exp3_validation [--cells-dir results/cells]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

MANIFEST = Path("configs/grids/exp3.json")

N_LAYERS = 36
FLOOR_LAYERS = {0, 1, 34, 35}

BYTES_PER_KV = {
    "auto": 4.0,
    "turboquant_k8v4": 1.5,
    "turboquant_4bit_nc": 1.0,
    "turboquant_k3v4_nc": 0.875,
    "turboquant_3bit_nc": 0.75,
}

import sys
sys.path.insert(0, ".")
from src.profiles import load_profiles

_PROFILES = load_profiles()
PPL_THRESHOLDS = {name: cfg["ppl_threshold"] for name, cfg in _PROFILES.items()}

PROFILE_NAMES = ["chat", "rag", "batch"]

ARM_LABELS = {
    "chat": {
        ("turboquant_k8v4", ()): "Tuned pick (k8v4)",
        ("turboquant_4bit_nc", ()): "vLLM default (4bit)",
        ("turboquant_3bit_nc", ()): "Aggressive (3bit)",
        ("auto", ()): "Baseline (FP16)",
    },
    "rag": {
        ("auto", ()): "Baseline (FP16)",
        ("turboquant_4bit_nc", ()): "vLLM default (4bit)",
        ("turboquant_4bit_nc", (2,)): "Tuned, positional (4bit+L2)",
        ("turboquant_4bit_nc", (5,)): "Tuned, stats-guided (4bit+L5)",
        ("turboquant_k8v4", ()): "k8v4 ref",
        ("turboquant_3bit_nc", ()): "Aggressive (3bit)",
    },
    "batch": {
        ("auto", ()): "Baseline (FP16)",
        ("turboquant_4bit_nc", ()): "Tuned pick (4bit)",
        ("turboquant_k8v4", ()): "k8v4 ref",
        ("turboquant_3bit_nc", ()): "Aggressive (3bit)",
    },
}

KEY_COMPARISONS = [
    ("chat", "auto_vs_tuned",
     ("auto", ()), ("turboquant_k8v4", ()),
     "Chat: baseline vs tuned pick (k8v4)"),
    ("chat", "auto_vs_naive",
     ("auto", ()), ("turboquant_4bit_nc", ()),
     "Chat: baseline vs naive 4bit"),
    ("chat", "naive_vs_tuned",
     ("turboquant_4bit_nc", ()), ("turboquant_k8v4", ()),
     "Chat: naive 4bit vs tuned k8v4"),

    ("rag", "auto_vs_default",
     ("auto", ()), ("turboquant_4bit_nc", ()),
     "RAG: baseline vs vLLM default (4bit)"),
    ("rag", "auto_vs_tuned",
     ("auto", ()), ("turboquant_4bit_nc", (5,)),
     "RAG: baseline vs tuned stat (4bit+L5)"),
    ("rag", "default_vs_stat",
     ("turboquant_4bit_nc", ()), ("turboquant_4bit_nc", (5,)),
     "RAG: vLLM default vs stats-guided protection"),
    ("rag", "default_vs_pos",
     ("turboquant_4bit_nc", ()), ("turboquant_4bit_nc", (2,)),
     "RAG: vLLM default vs positional protection"),
    ("rag", "pos_vs_stat",
     ("turboquant_4bit_nc", (2,)), ("turboquant_4bit_nc", (5,)),
     "RAG: H1b — positional vs stats-guided"),

    ("batch", "auto_vs_tuned",
     ("auto", ()), ("turboquant_4bit_nc", ()),
     "Batch: baseline vs tuned pick (4bit)"),
    ("batch", "auto_vs_aggressive",
     ("auto", ()), ("turboquant_3bit_nc", ()),
     "Batch: baseline vs aggressive 3bit"),
]


def compute_r_mem(dtype: str, skip_layers: tuple) -> float:
    if dtype == "auto":
        return 1.0
    b = BYTES_PER_KV[dtype]
    n_protect = len(FLOOR_LAYERS | set(int(x) for x in skip_layers))
    return (N_LAYERS * 4.0) / (n_protect * 4.0 + (N_LAYERS - n_protect) * b)


def config_key(cfg: dict) -> tuple:
    return (cfg["kv_cache_dtype"],
            tuple(sorted(int(x) for x in cfg.get("skip_layers", []))))


def load_cells(cells_dir: Path) -> list[dict]:
    manifest = json.loads(MANIFEST.read_text())
    from src.harness import CellConfig
    hashes = set()
    for entry in manifest:
        e = dict(entry)
        e["skip_layers"] = tuple(e.get("skip_layers", ()))
        hashes.add(CellConfig(**e).cell_hash())

    results = []
    for f in sorted(cells_dir.glob("*.json")):
        if f.stem in hashes:
            data = json.loads(f.read_text())
            if data.get("ok"):
                results.append(data)
    return results


def extract_metrics(cell: dict) -> dict:
    cfg = cell["config"]
    chat = cell.get("profiles", {}).get("chat", {})
    rag = cell.get("profiles", {}).get("rag", {})
    batch = cell.get("profiles", {}).get("batch", {})
    ttft_chat = chat.get("ttft_s", {}).get("mean")
    tpot_chat = chat.get("tpot_s", {}).get("mean")
    ttft_rag = rag.get("ttft_s", {}).get("mean")
    return {
        "ppl": cell.get("ppl", {}).get("ppl"),
        "chat_tpot_ms": tpot_chat * 1000 if tpot_chat else None,
        "chat_ttft_ms": ttft_chat * 1000 if ttft_chat else None,
        "chat_tps": chat.get("throughput_tok_s"),
        "rag_ttft_ms": ttft_rag * 1000 if ttft_rag else None,
        "needle_acc": rag.get("needle_accuracy"),
        "batch_tps": batch.get("throughput_tok_s"),
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

T_CRIT_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
             7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def mean_ci(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n < 2:
        return (values[0] if values else float("nan"), 0.0)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    t = T_CRIT_95.get(n, 2.0)
    return (m, t * se)


def _t_sf(t_val: float, df: int) -> float:
    try:
        from scipy.stats import t as t_dist
        return float(t_dist.sf(abs(t_val), df))
    except ImportError:
        x = abs(t_val) * (1 - 1 / (4 * df)) / math.sqrt(1 + t_val ** 2 / (2 * df))
        return 0.5 * math.erfc(x / math.sqrt(2))


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    d_mean = sum(diffs) / n
    d_var = sum((d - d_mean) ** 2 for d in diffs) / (n - 1)
    d_se = math.sqrt(d_var / n) if d_var > 0 else 0
    if d_se == 0:
        return (0.0, 1.0) if d_mean == 0 else (float("inf"), 0.0)
    t_stat = d_mean / d_se
    p = 2 * _t_sf(t_stat, n - 1)
    return (t_stat, min(p, 1.0))


def wilcoxon_test(a: list[float], b: list[float]) -> tuple[float, float]:
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, alternative="two-sided", method="exact")
        return (float(stat), float(p))
    except ImportError:
        diffs = [(abs(ai - bi), 1 if ai > bi else -1)
                 for ai, bi in zip(a, b) if ai != bi]
        if not diffs:
            return (0.0, 1.0)
        diffs.sort(key=lambda x: x[0])
        w_plus = sum(r + 1 for r, (_, s) in enumerate(diffs) if s > 0)
        w_minus = sum(r + 1 for r, (_, s) in enumerate(diffs) if s < 0)
        w = min(w_plus, w_minus)
        n = len(diffs)
        exact = {5: {0: 0.0625, 1: 0.0625, 2: 0.125, 3: 0.1875,
                     4: 0.3125, 5: 0.4375, 6: 0.5625, 7: 0.8125},
                 4: {0: 0.125, 1: 0.125, 2: 0.25, 3: 0.375, 4: 0.625},
                 3: {0: 0.25, 1: 0.25, 2: 0.5, 3: 0.75}}
        p = exact.get(n, {}).get(int(w), 1.0)
        return (w, p)


def cohens_d(a: list[float], b: list[float]) -> float:
    diffs = [ai - bi for ai, bi in zip(a, b)]
    n = len(diffs)
    if n < 2:
        return 0.0
    d_mean = sum(diffs) / n
    d_var = sum((d - d_mean) ** 2 for d in diffs) / (n - 1)
    return d_mean / math.sqrt(d_var) if d_var > 0 else 0.0


def holm_bonferroni(tests: list[dict]) -> list[dict]:
    n = len(tests)
    indexed = sorted(enumerate(tests), key=lambda x: x[1]["t_p"])
    max_adj = 0.0
    for rank, (i, t) in enumerate(indexed):
        adj = min(t["t_p"] * (n - rank), 1.0)
        adj = max(adj, max_adj)
        max_adj = adj
        tests[i]["t_p_adj"] = adj
        tests[i]["t_sig"] = adj < 0.05

    indexed_w = sorted(enumerate(tests), key=lambda x: x[1]["w_p"])
    max_adj = 0.0
    for rank, (i, t) in enumerate(indexed_w):
        adj = min(t["w_p"] * (n - rank), 1.0)
        adj = max(adj, max_adj)
        max_adj = adj
        tests[i]["w_p_adj"] = adj
        tests[i]["w_sig"] = adj < 0.05
    return tests


def compute_utility(profile: str, m: dict, base: dict,
                    dtype: str, skip: tuple) -> float | None:
    ppl = m.get("ppl")
    base_ppl = base.get("ppl")
    if ppl is None or base_ppl is None or base_ppl == 0:
        return None
    delta = (ppl - base_ppl) / base_ppl
    if delta > PPL_THRESHOLDS[profile]:
        return 0.0
    r_mem = compute_r_mem(dtype, skip)

    if profile == "chat":
        bt = base.get("chat_tpot_ms")
        ct = m.get("chat_tpot_ms")
        if not bt or not ct or ct == 0:
            return None
        return (bt / ct) ** 0.7 * r_mem ** 0.3

    if profile == "rag":
        bt = base.get("rag_ttft_ms")
        ct = m.get("rag_ttft_ms")
        if not bt or not ct or ct == 0:
            return None
        return r_mem ** 0.5 * (bt / ct) ** 0.5

    if profile == "batch":
        bt = base.get("batch_tps")
        ct = m.get("batch_tps")
        if not bt or bt == 0:
            return None
        return (ct / bt) ** 0.8 * r_mem ** 0.2 if ct else None

    return None


def fmt(v, digits=2):
    return f"{v:.{digits}f}" if v is not None else "N/A"


def fmt_ci(mean, hw, digits=2):
    return f"{mean:.{digits}f} ± {hw:.{digits}f}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells-dir", default="results/cells")
    args = p.parse_args()

    cells = load_cells(Path(args.cells_dir))
    if not cells:
        print("No completed validation cells found.")
        return

    print(f"Loaded {len(cells)} completed cells\n")

    # Group cells by config key → list of per-rep metric dicts
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        key = config_key(c["config"])
        m = extract_metrics(c)
        m["rep"] = c["config"]["rep"]
        grouped[key].append(m)

    for key in grouped:
        grouped[key].sort(key=lambda x: x["rep"])

    # -----------------------------------------------------------------------
    # Part 1: Per-config summary
    # -----------------------------------------------------------------------
    print("=" * 120)
    print("EXPERIMENT 3 — VALIDATION: PER-CONFIG SUMMARY (mean ± 95% CI, n=5)")
    print("=" * 120)
    header = (f"{'Config':<28} {'n':>2}  {'PPL':>14}  {'chat_TPOT':>14}  "
              f"{'chat_TPS':>14}  {'needle':>8}  {'batch_TPS':>14}  {'R_mem':>5}")
    print(header)
    print("-" * 120)

    config_summary = {}
    for key in sorted(grouped.keys()):
        dtype, skip = key
        reps = grouped[key]
        n = len(reps)
        label = dtype.replace("turboquant_", "")
        if skip:
            label += f"+L{','.join(str(s) for s in skip)}"

        fields = {}
        for field in ["ppl", "chat_tpot_ms", "chat_tps", "needle_acc", "batch_tps"]:
            vals = [r[field] for r in reps if r[field] is not None]
            if vals:
                m, hw = mean_ci(vals)
                fields[field] = (m, hw, vals)
            else:
                fields[field] = (None, None, [])

        r_mem = compute_r_mem(dtype, skip)
        config_summary[key] = {"fields": fields, "r_mem": r_mem, "label": label,
                               "reps": reps, "n": n}

        print(f"{label:<28} {n:>2}  "
              f"{fmt_ci(*fields['ppl'][:2], 3) if fields['ppl'][0] is not None else 'N/A':>14}  "
              f"{fmt_ci(*fields['chat_tpot_ms'][:2]) if fields['chat_tpot_ms'][0] is not None else 'N/A':>14}  "
              f"{fmt_ci(*fields['chat_tps'][:2], 1) if fields['chat_tps'][0] is not None else 'N/A':>14}  "
              f"{fmt_ci(*fields['needle_acc'][:2]) if fields['needle_acc'][0] is not None else 'N/A':>8}  "
              f"{fmt_ci(*fields['batch_tps'][:2], 1) if fields['batch_tps'][0] is not None else 'N/A':>14}  "
              f"{r_mem:>5.2f}x")
    print()

    # -----------------------------------------------------------------------
    # Part 2: Per-profile utility
    # -----------------------------------------------------------------------
    baseline_key = ("auto", ())
    if baseline_key not in grouped:
        print("ERROR: No baseline (auto) config found.")
        return

    baseline_reps = grouped[baseline_key]
    base_metrics_by_rep = {r["rep"]: r for r in baseline_reps}

    profile_utilities: dict[str, dict[tuple, list[float]]] = {}

    for profile in PROFILE_NAMES:
        print("=" * 100)
        print(f"UTILITY — {profile.upper()} (SPEC §4: threshold dPPL ≤ {PPL_THRESHOLDS[profile]*100:.1f}%)")
        print("=" * 100)
        print(f"{'Config':<28} {'Utility':>14}  {'dPPL%':>8}  {'R_mem':>6}  {'Pass':>4}")
        print("-" * 100)

        util_data: dict[tuple, list[float]] = {}
        for key in sorted(grouped.keys()):
            dtype, skip = key
            reps = grouped[key]
            label = config_summary[key]["label"]
            r_mem = config_summary[key]["r_mem"]

            utilities = []
            ppls = []
            for rep_m in reps:
                base_m = base_metrics_by_rep.get(rep_m["rep"])
                if base_m is None:
                    continue
                u = compute_utility(profile, rep_m, base_m, dtype, skip)
                if u is not None:
                    utilities.append(u)
                if rep_m["ppl"] is not None and base_m["ppl"] is not None:
                    ppls.append((rep_m["ppl"] - base_m["ppl"]) / base_m["ppl"] * 100)

            util_data[key] = utilities

            if utilities:
                u_mean, u_hw = mean_ci(utilities)
                ppl_mean = sum(ppls) / len(ppls) if ppls else None
                passes = all(u > 0 for u in utilities)
                print(f"{label:<28} {fmt_ci(u_mean, u_hw, 4):>14}  "
                      f"{fmt(ppl_mean, 2) + '%':>8}  {r_mem:>5.2f}x  "
                      f"{'YES' if passes else 'NO':>4}")
            else:
                print(f"{label:<28} {'N/A':>14}  {'N/A':>8}  {r_mem:>5.2f}x  {'N/A':>4}")

        profile_utilities[profile] = util_data
        print()

    # -----------------------------------------------------------------------
    # Part 3: Pairwise comparisons
    # -----------------------------------------------------------------------
    print("=" * 130)
    print("PAIRWISE COMPARISONS — paired t-test + Wilcoxon signed-rank, n=5")
    print("=" * 130)
    print(f"{'Comparison':<50} {'Metric':<12} {'Delta':>10}  "
          f"{'t-stat':>7} {'t-p':>8}  {'W-stat':>6} {'W-p':>8}  {'Cohen d':>8}")
    print("-" * 130)

    all_tests = []

    for profile, comp_id, key_a, key_b, desc in KEY_COMPARISONS:
        if key_a not in grouped or key_b not in grouped:
            continue

        reps_a = grouped[key_a]
        reps_b = grouped[key_b]
        rep_set = sorted(set(r["rep"] for r in reps_a) & set(r["rep"] for r in reps_b))
        if len(rep_set) < 3:
            continue

        a_by_rep = {r["rep"]: r for r in reps_a}
        b_by_rep = {r["rep"]: r for r in reps_b}

        metrics_to_test = {
            "chat": [("ppl", "PPL"), ("chat_tpot_ms", "TPOT_ms")],
            "rag": [("ppl", "PPL"), ("needle_acc", "needle")],
            "batch": [("ppl", "PPL"), ("batch_tps", "batch_TPS")],
        }

        # Also test utility
        u_a = profile_utilities.get(profile, {}).get(key_a, [])
        u_b = profile_utilities.get(profile, {}).get(key_b, [])

        for field, label in metrics_to_test.get(profile, []):
            vals_a = [a_by_rep[r][field] for r in rep_set if a_by_rep[r][field] is not None]
            vals_b = [b_by_rep[r][field] for r in rep_set if b_by_rep[r][field] is not None]
            if len(vals_a) < 3 or len(vals_a) != len(vals_b):
                continue

            delta_mean = sum(a - b for a, b in zip(vals_a, vals_b)) / len(vals_a)
            t_stat, t_p = paired_ttest(vals_a, vals_b)
            w_stat, w_p = wilcoxon_test(vals_a, vals_b)
            d = cohens_d(vals_a, vals_b)

            test_entry = {
                "comparison": desc, "metric": label, "profile": profile,
                "delta": delta_mean, "t_stat": t_stat, "t_p": t_p,
                "w_stat": w_stat, "w_p": w_p, "d": d,
            }
            all_tests.append(test_entry)
            print(f"{desc:<50} {label:<12} {delta_mean:>+10.4f}  "
                  f"{t_stat:>7.2f} {t_p:>8.4f}  "
                  f"{w_stat:>6.1f} {w_p:>8.4f}  {d:>+8.2f}")

        if len(u_a) >= 3 and len(u_a) == len(u_b):
            delta_mean = sum(a - b for a, b in zip(u_a, u_b)) / len(u_a)
            t_stat, t_p = paired_ttest(u_a, u_b)
            w_stat, w_p = wilcoxon_test(u_a, u_b)
            d = cohens_d(u_a, u_b)
            test_entry = {
                "comparison": desc, "metric": "utility", "profile": profile,
                "delta": delta_mean, "t_stat": t_stat, "t_p": t_p,
                "w_stat": w_stat, "w_p": w_p, "d": d,
            }
            all_tests.append(test_entry)
            print(f"{desc:<50} {'utility':<12} {delta_mean:>+10.4f}  "
                  f"{t_stat:>7.2f} {t_p:>8.4f}  "
                  f"{w_stat:>6.1f} {w_p:>8.4f}  {d:>+8.2f}")

    print()

    # -----------------------------------------------------------------------
    # Part 4: Holm-Bonferroni correction
    # -----------------------------------------------------------------------
    if all_tests:
        corrected = holm_bonferroni(all_tests)

        print("=" * 130)
        print("HOLM-BONFERRONI CORRECTED P-VALUES")
        print("=" * 130)
        print(f"{'Comparison':<50} {'Metric':<12} {'raw t-p':>8} {'adj t-p':>8} {'t-sig':>5}  "
              f"{'raw W-p':>8} {'adj W-p':>8} {'W-sig':>5}")
        print("-" * 130)

        for t in sorted(corrected, key=lambda x: x["t_p"]):
            sig_t = "*" if t.get("t_sig") else ""
            sig_w = "*" if t.get("w_sig") else ""
            print(f"{t['comparison']:<50} {t['metric']:<12} "
                  f"{t['t_p']:>8.4f} {t['t_p_adj']:>8.4f} {sig_t:>5}  "
                  f"{t['w_p']:>8.4f} {t['w_p_adj']:>8.4f} {sig_w:>5}")
        print()

        sig_count = sum(1 for t in corrected if t.get("t_sig"))
        print(f"  Total tests: {len(corrected)}")
        print(f"  Significant (t-test, corrected): {sig_count}")
        print(f"  Note: Wilcoxon with n=5 has minimum two-tailed p=0.0625 > 0.05;")
        print(f"        significance via Wilcoxon requires all 5 differences same sign.")

    # -----------------------------------------------------------------------
    # Part 5: Headline claims
    # -----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("HEADLINE CLAIMS — Experiment 3 Validation")
    print("=" * 100)

    for profile in PROFILE_NAMES:
        tuned_keys = {
            "chat": ("turboquant_k8v4", ()),
            "rag": ("turboquant_4bit_nc", (5,)),
            "batch": ("turboquant_4bit_nc", ()),
        }
        at_key = tuned_keys[profile]
        if at_key not in config_summary:
            continue

        at = config_summary[at_key]
        base = config_summary.get(baseline_key, {})
        at_ppl = at["fields"]["ppl"]
        base_ppl = base["fields"]["ppl"]

        print(f"\n  {profile.upper()}:")
        print(f"    Tuned pick: {at['label']}, R_mem={at['r_mem']:.2f}x")
        if at_ppl[0] is not None and base_ppl[0] is not None:
            dppl = at_ppl[0] - base_ppl[0]
            dppl_pct = dppl / base_ppl[0] * 100
            print(f"    PPL: {at_ppl[0]:.3f} vs baseline {base_ppl[0]:.3f} "
                  f"(delta: {dppl:+.3f}, {dppl_pct:+.2f}%)")

        utils = profile_utilities.get(profile, {})
        at_u = utils.get(at_key, [])
        base_u = utils.get(baseline_key, [])
        if at_u:
            u_m, u_hw = mean_ci(at_u)
            print(f"    Utility: {u_m:.4f} ± {u_hw:.4f}")
        if base_u:
            bu_m, _ = mean_ci(base_u)
            if at_u:
                gain = (u_m - bu_m) / bu_m * 100 if bu_m else 0
                print(f"    Utility gain over baseline: {gain:+.1f}%")

    # -----------------------------------------------------------------------
    # Save structured results
    # -----------------------------------------------------------------------
    out = {
        "n_cells": len(cells),
        "configs": {},
        "comparisons": all_tests if all_tests else [],
    }
    for key, info in config_summary.items():
        dtype, skip = key
        out["configs"][f"{dtype}_{list(skip)}"] = {
            "label": info["label"],
            "r_mem": info["r_mem"],
            "n_reps": info["n"],
            "ppl_mean": info["fields"]["ppl"][0],
        }

    out_path = Path("results/exp3_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
