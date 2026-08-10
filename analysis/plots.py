"""Report figures, regenerated from committed/collected raw data.

Every figure here reads directly from the harness's per-cell JSON
(``results/cells/``) or the profiler's Exp-0 output
(``results/exp0/exp0_result.json``) — nothing is hand-transcribed, so
regenerating a figure after a rerun is just re-running this script.

Reuses ``src.tuner``'s cell loading/utility math instead of re-deriving
it, so the figures and the tuner's own numbers can never drift apart.

CLI:
    python -m analysis.plots --cells-dir results/cells --out-dir results/figures
    python -m analysis.plots --exp0-path results/exp0/exp0_result.json --out-dir results/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed to render PNGs
import matplotlib.pyplot as plt

from src.model_info import detect_n_layers, floor_layers_for
from src.tuner import (
    PRESETS,
    average_cells,
    compute_r_mem,
    compute_utility,
    load_baseline,
    load_cells,
    load_profiler_ranking,
)

# ---------------------------------------------------------------------------
# Style — fixed categorical order (never re-cycled per chart), one hue per
# preset, mild-to-aggressive; a muted gray for the FP16 baseline reference
# since it's a reference line, not a competing series.
# ---------------------------------------------------------------------------

PRESET_COLOR = {
    "k8v4": "#2a78d6",      # slot 1 blue   — mildest
    "4bit_nc": "#eb6834",   # slot 2 orange
    "k3v4_nc": "#1baf7a",   # slot 3 aqua
    "3bit_nc": "#eda100",   # slot 4 yellow — most aggressive
}
BASELINE_COLOR = "#898781"      # muted ink — reference, not a series
GRID_COLOR = "#e1e0d9"
TEXT_COLOR = "#0b0b0b"
SECONDARY_TEXT = "#52514e"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]

PROFILE_THRESHOLDS_PCT = {"chat": 0.5, "rag": 1.0, "batch": 2.0}


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=SECONDARY_TEXT, labelsize=9)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(SECONDARY_TEXT)
    ax.yaxis.label.set_color(SECONDARY_TEXT)
    ax.title.set_color(TEXT_COLOR)


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figure 1: PPL degradation per preset (the characterization headline)
# ---------------------------------------------------------------------------

def plot_preset_ppl_tradeoff(cells_dir: Path, ranking: list[int], out_dir: Path) -> None:
    cells = load_cells(cells_dir, ranking, include_positional=True)
    if not cells:
        print(f"skip plot_preset_ppl_tradeoff: no cells in {cells_dir}")
        return
    try:
        baseline = load_baseline(cells_dir)
    except RuntimeError as exc:
        print(f"skip plot_preset_ppl_tradeoff: {exc}")
        return

    floor_only = [c for c in cells if c.n_protect == 0]
    averaged = average_cells(floor_only)
    rows = {r["preset"]: r for r in averaged if r["ppl"] is not None}
    if not rows:
        print("skip plot_preset_ppl_tradeoff: no PPL data in floor-only cells")
        return

    presets = [p for p in PRESETS if p in rows]
    dppl_pct = [(rows[p]["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100 for p in presets]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(presets, dppl_pct, color=[PRESET_COLOR[p] for p in presets],
                  width=0.6, zorder=2)
    for bar, val in zip(bars, dppl_pct):
        ax.annotate(f"{val:+.2f}%", (bar.get_x() + bar.get_width() / 2, val),
                   ha="center", va="bottom" if val >= 0 else "top",
                   fontsize=9, color=TEXT_COLOR)

    for threshold in PROFILE_THRESHOLDS_PCT.values():
        ax.axhline(threshold, color=BASELINE_COLOR, linestyle="--", linewidth=1, zorder=1)

    legend_lines = "\n".join(f"{p} threshold: {t}%" for p, t in PROFILE_THRESHOLDS_PCT.items())
    ax.text(0.02, 0.98, legend_lines, transform=ax.transAxes, fontsize=8,
           color=SECONDARY_TEXT, va="top", ha="left")

    ax.set_ylabel("ΔPPL vs. FP16 baseline (%)")
    ax.set_title("PPL degradation by preset (floor-only protection)")
    _style_axes(ax)
    _save(fig, out_dir, "preset_ppl_tradeoff")


# ---------------------------------------------------------------------------
# Figure 2: per-workload utility comparison (the "no single default" figure)
# ---------------------------------------------------------------------------

def plot_workload_utility(cells_dir: Path, ranking: list[int], n_layers: int,
                          n_floor: int, out_dir: Path) -> None:
    cells = load_cells(cells_dir, ranking, include_positional=True)
    if not cells:
        print(f"skip plot_workload_utility: no cells in {cells_dir}")
        return
    try:
        baseline = load_baseline(cells_dir)
    except RuntimeError as exc:
        print(f"skip plot_workload_utility: {exc}")
        return

    floor_only = [c for c in cells if c.n_protect == 0]
    averaged = average_cells(floor_only)
    rows = {r["preset"]: r for r in averaged}
    presets = [p for p in PRESETS if p in rows]
    profiles = list(PROFILE_THRESHOLDS_PCT)

    fig, ax = plt.subplots(figsize=(7, 4))
    n_presets = len(presets)
    width = 0.8 / max(n_presets, 1)
    x = range(len(profiles))
    for i, preset in enumerate(presets):
        utilities = [
            compute_utility(profile, rows[preset], baseline, preset, 0, n_layers, n_floor)
            for profile in profiles
        ]
        offsets = [xi + (i - n_presets / 2 + 0.5) * width for xi in x]
        ax.bar(offsets, utilities, width=width * 0.9, color=PRESET_COLOR[preset],
              label=preset, zorder=2)

    ax.axhline(1.0, color=BASELINE_COLOR, linestyle="--", linewidth=1, zorder=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([p.upper() for p in profiles])
    ax.set_ylabel("Utility (1.0 = same as FP16 baseline; 0 = violates PPL constraint)")
    ax.set_title("Workload-specific utility per preset (floor-only protection)")
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_TEXT)
    _style_axes(ax)
    _save(fig, out_dir, "workload_utility")


# ---------------------------------------------------------------------------
# Figure 3: per-layer sensitivity curve (Exp 0)
# ---------------------------------------------------------------------------

def plot_layer_sensitivity(exp0_path: Path, out_dir: Path) -> None:
    if not exp0_path.exists():
        print(f"skip plot_layer_sensitivity: {exp0_path} not found")
        return
    data = json.loads(exp0_path.read_text())
    layers = data["sensitivity"]["layers"]
    floor = floor_layers_for(len(layers))

    idx = [l["layer"] for l in layers]
    delta = [l["delta_ppl"] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    for i in floor:
        ax.axvspan(i - 0.5, i + 0.5, color=SEQUENTIAL_BLUE[0], zorder=0)
    ax.plot(idx, delta, color=PRESET_COLOR["3bit_nc"], linewidth=1.5, zorder=2)
    ax.scatter(idx, delta, color=PRESET_COLOR["3bit_nc"], s=14, zorder=3)
    ax.axhline(0, color=BASELINE_COLOR, linewidth=1, zorder=1)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("ΔPPL (single-layer quantization)")
    ax.set_title("Per-layer sensitivity — shaded = vLLM's positional floor")
    _style_axes(ax)
    _save(fig, out_dir, "layer_sensitivity")


# ---------------------------------------------------------------------------
# Figure 4: model-dependent sensitivity (Exp 4 generalization)
# ---------------------------------------------------------------------------

def plot_model_generalization(models_cells: dict[str, Path], out_dir: Path) -> None:
    """models_cells: {model_name: cells_dir} — one dir per profiled model."""
    per_model_dppl = {}
    for model_name, cells_dir in models_cells.items():
        try:
            n_layers = detect_n_layers(model_name)
        except Exception as exc:  # noqa: BLE001 — best-effort, network/HF dependent
            print(f"skip {model_name}: cannot detect layer count ({exc})")
            continue
        floor = floor_layers_for(n_layers)
        ranking = load_profiler_ranking(floor, n_layers)
        cells = load_cells(cells_dir, ranking, include_positional=True)
        if not cells:
            print(f"skip {model_name}: no cells in {cells_dir}")
            continue
        try:
            baseline = load_baseline(cells_dir)
        except RuntimeError:
            print(f"skip {model_name}: no baseline cell in {cells_dir}")
            continue
        floor_only = [c for c in cells if c.n_protect == 0]
        averaged = {r["preset"]: r for r in average_cells(floor_only) if r["ppl"]}
        per_model_dppl[model_name] = {
            preset: (row["ppl"] - baseline["ppl"]) / baseline["ppl"] * 100
            for preset, row in averaged.items()
        }

    if not per_model_dppl:
        print("skip plot_model_generalization: no model data available")
        return

    presets = [p for p in PRESETS if any(p in v for v in per_model_dppl.values())]
    models = list(per_model_dppl)

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.8 / max(len(presets), 1)
    x = range(len(models))
    for i, preset in enumerate(presets):
        vals = [per_model_dppl[m].get(preset) for m in models]
        offsets = [xi + (i - len(presets) / 2 + 0.5) * width for xi in x]
        vals_plot = [v if v is not None else 0 for v in vals]
        ax.bar(offsets, vals_plot, width=width * 0.9, color=PRESET_COLOR[preset],
              label=preset, zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("ΔPPL vs. FP16 baseline (%)")
    ax.set_title("Quantization sensitivity by model (same preset, floor-only)")
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_TEXT)
    _style_axes(ax)
    _save(fig, out_dir, "model_generalization")


# ---------------------------------------------------------------------------
# Figure 5: chat TPOT latency distribution per preset
# ---------------------------------------------------------------------------

def plot_latency_distribution(cells_dir: Path, ranking: list[int], out_dir: Path) -> None:
    cells = load_cells(cells_dir, ranking, include_positional=True)
    floor_only = [c for c in cells if c.n_protect == 0 and c.chat_tpot_ms is not None]
    if not floor_only:
        print(f"skip plot_latency_distribution: no chat TPOT data in {cells_dir}")
        return

    by_preset: dict[str, list[float]] = {}
    for c in floor_only:
        by_preset.setdefault(c.preset, []).append(c.chat_tpot_ms)

    presets = [p for p in PRESETS if p in by_preset]
    data = [by_preset[p] for p in presets]

    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, tick_labels=presets, patch_artist=True, widths=0.5, zorder=2)
    for patch, preset in zip(bp["boxes"], presets):
        patch.set_facecolor(PRESET_COLOR[preset])
        patch.set_alpha(0.7)
        patch.set_edgecolor(TEXT_COLOR)
    for element in ("whiskers", "caps", "medians"):
        for line in bp[element]:
            line.set_color(TEXT_COLOR)

    ax.set_ylabel("Chat TPOT (ms) — lower is better")
    ax.set_title("Per-token decode latency distribution by preset")
    _style_axes(ax)
    _save(fig, out_dir, "latency_distribution")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells-dir", default="results/cells")
    p.add_argument("--exp0-path", default="results/exp0/exp0_result.json")
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="primary model (for layer topology in figures 1/2/5)")
    p.add_argument("--n-layers", type=int, default=None,
                   help="override detected layer count (skips the HF config lookup)")
    p.add_argument("--out-dir", default="results/figures")
    args = p.parse_args()

    cells_dir = Path(args.cells_dir)
    out_dir = Path(args.out_dir)
    n_layers = args.n_layers if args.n_layers is not None else detect_n_layers(args.model)
    floor_layers = floor_layers_for(n_layers)
    ranking = load_profiler_ranking(floor_layers, n_layers)

    plot_preset_ppl_tradeoff(cells_dir, ranking, out_dir)
    plot_workload_utility(cells_dir, ranking, n_layers, len(floor_layers), out_dir)
    plot_layer_sensitivity(Path(args.exp0_path), out_dir)
    plot_latency_distribution(cells_dir, ranking, out_dir)
    print(f"\nFigures written to {out_dir}/ "
          f"(plot_model_generalization needs multiple models' cell dirs — "
          f"call it directly once Exp 4 has results for more than one model)")


if __name__ == "__main__":
    main()
