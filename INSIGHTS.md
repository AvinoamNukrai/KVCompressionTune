# Insights Log — material for the final report

Short, dated findings collected during the project. Each entry: what we found,
how we know, and why it matters for the report.

## Baseline & configuration space (from vLLM source reading, 2026-07-11)

1. **TurboQuant is upstream in vLLM** (merged 2026-04-15,
   [vllm#38479](https://github.com/vllm-project/vllm/pull/38479)) but exposed
   only as **4 fixed presets** (`k8v4`, `4bit_nc`, `k3v4_nc`, `3bit_nc`).
   Key-bits / value-bits / norm-correction are **confounded inside the preset**
   — the full factorial exists in the code (`TurboQuantConfig` dataclass) but is
   not reachable from the CLI. → Motivation + PR opportunity.
2. **Layer protection is hard-coded**: vLLM force-protects the first 2 and last
   2 layers for every `turboquant_*` dtype, validated by its authors on Qwen3-4B
   (their comment: removing it costs ~30 GSM8K points at aggressive presets).
3. **Protection is expensive on small models**: protecting s of L layers cuts
   effective compression to `L/(s+(L−s)/r)`. Qwen3-4B (36L) at `3bit_nc`
   (4.9×): the default n=2 leaves only 3.5× — ~29% of the compression paid for.
   On an 80-layer model the same rule is nearly free → optimal protection is
   model-size-dependent.
4. **Community context (Red Hat / vLLM study, May 2026)**: no single config
   wins — FP8 KV is the best default on Hopper+, 3-bit collapses on reasoning
   at long context, Qwen-class models need more K bits than V bits, and layer-0
   has ~20% outlier channels vs 4–6% mid-stack. Nobody automated the choice.

## Smoke test on RTX 3090, Qwen3-4B, vLLM 0.21.0 (2026-07-14)

5. **The n=2 protection floor cannot be disabled on stock vLLM** (measured):
   passing an empty `--kv-cache-dtype-skip-layers` still yields effective
   protection `[0, 1, 34, 35]`; user-supplied layers are *unioned* with the
   floor (`[10,11]` → `[0,1,10,11,34,35]`). You can protect **more, never
   fewer**. → Shapes the experiment grid (floor + additions); strong PR
   motivation; report-worthy limitation of the shipped implementation.
6. **`turboquant_k8v4` fails to initialize on RTX 3090** (Ampere, SM 8.6 — no
   native FP8), while all three `_nc` presets work and generate coherent
   output. Pending root-cause confirmation, the **feasible preset space itself
   is hardware-dependent** — hardware-awareness demonstrated before any tuning.
7. **All `_nc` presets pass sanity on 3090**: 8/8 distinct coherent outputs,
   correct 4K-token retrieval answer, ~460–470 tok/s short-gen (vs 498 tok/s
   FP16 baseline — ~6% decode overhead at trivial scale; real differences are
   expected under long context / concurrency, not here).
8. **Per-cell cost measured**: ~85 s TQ engine load (226 s only on first cold
   weight read), ~103 s per config with mini-workloads. Confirms the
   ~30–40 GPU-h budget for the full experimental program, with margin.

## Smoke test on RTX 4090, Qwen3-4B, vLLM 0.21.0 (2026-07-14)

11. **7/7 configs pass on RTX 4090 (Ada, SM 8.9), including `k8v4`** — the
    preset that fails to initialize on the 3090. Combined with #6: **the
    feasible preset space is hardware-dependent**, measured from both sides
    (same model, same vLLM, same scripts; only the GPU differs). AutoTune must
    treat feasibility, not just optimality, as GPU-specific.
12. **`k8v4` is the slowest TQ preset on the 4090 at small scale** (562 tok/s
    vs ~612 for all `_nc` presets, 642 FP16 baseline) — the *lightest*
    quantization costs the most decode speed. Single mini-run → hypothesis for
    Experiment 1, not a claim yet.
13. **The n=2 protection floor reproduces identically on the 4090**
    (empty skip list → `[0,1,34,35]`) — finding #5 generalizes across GPUs.

## Harness validation — baseline vs. 3bit_nc, RTX 4090, Qwen3-4B (2026-07-24)

Source: Phase 2 harness (`src/harness.py`), two cells — `auto` (FP16 baseline)
vs. `turboquant_3bit_nc`, both rep 0 on RTX 4090.

14. **TTFT paradox: 3-bit quantization is 8× faster for chat TTFT** (78 ms vs.
    636 ms baseline). Compressed KV means more tokens fit per chunked-prefill
    batch → less queuing → faster time-to-first-token under concurrent
    scheduling. This is a workload-dependent trade-off: TTFT improves while
    per-token decode slows down. → Core motivation for workload-aware tuning.
15. **TPOT is 19% slower with 3-bit** (13.6 ms vs. 11.5 ms baseline). Each
    generated token pays quantization/dequantization overhead. Chat workloads
    (decode-heavy) are hurt; RAG workloads (prefill-heavy, short output) are
    less affected.
16. **PPL degrades 3.4% under 3bit_nc** (10.70 vs. 10.35 baseline). Exceeds
    the SPEC's 0.5% chat tolerance but within the 2% batch threshold. →
    Confirms that aggressive quantization is acceptable for some workloads but
    not others — the utility function per-profile design is necessary.
17. **Peak VRAM is identical between configs** (21.64 GB 3-bit vs. 21.55 GB
    baseline). vLLM pre-allocates a fixed KV-cache pool based on
    `gpu_memory_utilization` (0.85 × 24 GB). Quantization does not reduce
    *peak memory used* — it increases *capacity* (more tokens fit in the same
    pool). **Peak VRAM is the wrong metric for measuring compression benefit**;
    the correct metric is the number of KV-cache blocks allocated or max
    concurrent sequences. → Must add block-count extraction to harness before
    Experiment 1.
18. **Needle accuracy is 1.0 (20/20) under 3bit_nc** — the model retrieves the
    correct embedded code from 4K–8K token contexts even at aggressive
    quantization. Combined with finding #16: 3-bit hurts PPL (global quality)
    but not simple retrieval (local attention still works).
19. **Per-cell wall time is ~70–100 s on RTX 4090** (67 s for 3bit_nc, 99 s for
    auto). FP16 baseline is slower due to longer engine init (59 s vs. 27 s) —
    larger per-token KV means more CUDA graph compilations. Budget for Exp 1
    (28 cells × 2 reps × ~85 s) ≈ 1.3 GPU-hours — well within the 8–12 h
    allocation.

## Experiment 0 — Layer sensitivity profiler, Qwen3-4B (2026-07-25)

Source: `src/profiler.py --mode full`, HuggingFace Transformers (not vLLM),
8 WikiText calibration docs, 4 eval docs, 3-bit keys / 4-bit values,
chunked PPL with 256-token chunks. Results in `results/exp0/`.

22. **Layer 0 is ~19× more sensitive than any other layer** (ΔPPL=+4.59 vs.
    next-worst layer 34 at +0.24). quant_error=6.84 vs ~0.15 median. Layer 0
    must always be protected regardless of budget.
23. **`simulated_quant_error` is the only feature that predicts sensitivity**
    (Spearman ρ=+0.40, p=0.016). The other four candidates — key outlier
    fraction, post-WHT excess kurtosis, key-norm CV, value dynamic range — all
    fail (p>0.10). This is because TurboQuant's Hadamard rotation specifically
    neutralizes the outlier/kurtosis properties those features measure.
24. **vLLM's positional protection is suboptimal on Qwen3-4B.** The fixed
    "first 2 + last 2" rule protects {0,1,34,35}. Layer 1 is insensitive
    (ΔPPL=+0.01), wasting budget. Layers 30-33 are sensitive (ΔPPL=+0.07
    to +0.15) but unprotected. A statistics-guided policy protecting
    {0,32,33,34} at the same 4-layer budget covers 3× more ΔPPL.
25. **Layer 5 breaks the correlation** — second-highest quant_error (0.85) but
    zero sensitivity (ΔPPL=-0.007). The quantization damage doesn't propagate
    downstream, likely because early layers have enough residual capacity to
    compensate. This limits single-feature prediction to ρ≈0.4.
26. **Correct Lloyd-Max centroids matter.** Initial hardcoded centroids (wrong
    3-bit values, only 12/16 for 4-bit) gave ρ=0.50. Switching to dynamically
    computed centroids via the iterative Lloyd-Max algorithm for N(0,1) dropped
    it to ρ=0.40 — the earlier "better" result was an artifact of wrong math.
27. **Value dynamic range increases monotonically with depth** — from 0.24
    (layer 0) to 26.3 (layer 34), dropping to 18.0 at layer 35. Despite this
    100× spread, it doesn't correlate with sensitivity (ρ=+0.24, p=0.15),
    because TurboQuant uses per-vector min-max scaling that absorbs range.
28. **Some mid-layers show negative ΔPPL** (layers 18, 20: ΔPPL=-0.08, -0.04).
    Quantization acts as regularization — the added noise slightly improves
    generalization on the eval set. Not reproducible across seeds; noise floor.
29. **Modern transformers (4.48+) uses `DynamicLayer` cache API**, not the older
    `DynamicCache.key_cache`/`value_cache` lists. Cache entries are accessed via
    `past_kv.layers[i].keys` / `.values`. Iteration yields 3-tuples, not
    2-tuples.

## Experiment 1 — Screening grid, Qwen3-4B, RTX 4090 (2026-07-25)

Source: `src/harness.py --manifest configs/grids/exp1.json`, vLLM 0.21.0
engine, 34 cells = 5 kv_cache_dtype × 4 protection sets × 2 reps.
Protection sets: floor (vLLM default {0,1,34,35}), pos_n4 (+{2,3,32,33}),
stat_b8 (+{4,5,8,33} from Exp 0 ranking), stat_b6 (+{5,33}).
Results in `results/cells/`.

30. **Preset selection dominates layer protection.** The PPL gap between best
    TQ preset (k8v4, 10.34 — matching baseline 10.35) and worst (3bit_nc/floor,
    10.70 — +3.4%) is 0.36 PPL. Layer protection at best recovers 0.09 PPL
    (3bit_nc stat_b8 vs floor). AutoTune's main practical value is automated
    preset selection, not layer protection fine-tuning.
31. **k8v4 achieves zero PPL degradation** (10.34 vs 10.35 baseline, within
    noise) at +9% TPOT overhead. FP8 keys are effectively lossless; 4-bit
    uniform values cause negligible damage. However, k8v4 requires Ada/Hopper
    (fails on Ampere — finding #6), so hardware-aware selection is essential.
32. **Layer protection scales with quantization aggressiveness.** On 3bit_nc
    and k3v4_nc, protecting 4 extra layers (stat_b8) recovers ~0.09 PPL
    (~25% of the degradation). On 4bit_nc and k8v4, protection adds <0.01 PPL
    because per-layer damage is already negligible. Optimal protection budget
    is preset-dependent.
33. **Stats-guided ≈ positional protection on PPL** at the same 8-layer budget.
    Differences are 0.003–0.031 PPL, within noise at 2 reps. However,
    stats-guided fixes k8v4/floor's needle accuracy drop (0.95 → 1.00),
    suggesting it protects the right layers for retrieval tasks even when
    PPL differences are marginal.
34. **TPOT overhead is monotonic with compression aggressiveness**: k8v4 +9%,
    4bit_nc +13–15%, k3v4_nc +16–18%, 3bit_nc +16–19% vs FP16 baseline.
    More compressed cache = more dequantization work per decode step.
35. **TTFT is too noisy for comparison at 2 reps.** Cold-start effects
    (CUDA graph compilation, first engine load) cause 10× variance between
    reps (e.g., 817ms vs 76ms). Needs more reps or warm-up runs to be
    reliable.
36. **Needle accuracy is robust across all configs** — 1.00 (20/20) everywhere
    except k8v4/floor (0.95) and one 3bit_nc/stat_b6 rep (0.95). KV-cache
    quantization does not break simple retrieval even at aggressive settings.

## Experiment 2 — Auto-tuner, Qwen3-4B, RTX 4090 (2026-07-26)

Source: `src/tuner.py --optimize`, Optuna TPE over (preset, protection budget)
with utility functions from SPEC §4. 20 unique (preset, budget) configs from
Exp 1 + 16 refinement cells (k3v4_nc budgets 1-8, 4bit_nc budgets 1,3).
PPL hard constraints: chat ≤0.5%, RAG ≤1%, batch ≤2%.

37. **Each workload profile selects a different optimal config.** Chat → k8v4
    (zero PPL loss, R_mem 2.25×); RAG → 4bit_nc + protect layer 5 (PPL +0.9%,
    R_mem 2.82×); Batch → 4bit_nc floor-only (PPL +1.0%, R_mem 3.00×). A
    single default config cannot serve all three — AutoTune's core premise.
38. **RAG selects extra protection (n_protect=1)** — the only profile where the
    tuner adds protection beyond the vLLM floor. Protecting layer 5 improves
    4bit_nc PPL from 10.446 → 10.442, pushing utility from 1.52 to 1.54.
    Marginal but measurable.
39. **Layer 33 is the critical non-floor layer for aggressive presets.**
    k3v4_nc PPL jumps from 10.662 → 10.570 when layer 33 is added (budget 1 →
    budget 2). This single layer accounts for ~90% of all protection value.
    After layer 33, diminishing returns plateau.
40. **Protection has a sweet spot at budget 2-3 for k3v4_nc.** Budget 3
    achieves the best PPL (10.561, recovering +0.106 from floor). Higher
    budgets (4-8) show no improvement or slight regression — more protection
    can hurt. Optimal budget is preset-dependent, not "protect as many as
    possible."
41. **k3v4_nc misses batch viability by 0.007 PPL.** Budget 3 gives PPL 10.561
    vs threshold 10.554. Protection brought it 94% of the way (from gap=0.116
    to gap=0.007). On a slightly less sensitive model, protection would flip
    k3v4_nc to viable — demonstrating the threshold-crossing value of tuning.
42. **Protection fixes k8v4 needle accuracy for free.** Floor-only k8v4 has
    needle accuracy 0.95; budget 2 (protecting layers 5, 33) restores it to
    1.00 with identical PPL (10.343). A quality improvement invisible to PPL.
43. **Preset × protection interaction confirmed.** Aggressive presets (k3v4_nc,
    3bit_nc) benefit from protection (+0.08-0.10 PPL recovery). Mild presets
    (k8v4, 4bit_nc) show no measurable PPL effect (±0.007, noise). The optimal
    protection budget is zero for mild presets and 2-3 for aggressive ones.

## Experiment 3 — Validation, Qwen3-4B, RTX 4090 (2026-07-26)

Source: `analysis/exp3_validation.py`, 30 cells = 6 configs × 5 reps on
held-out traces (tag=valid, seed=20260726). Full SPEC §5.1 statistical
protocol: paired t-test, Wilcoxon signed-rank, Cohen's d, Holm-Bonferroni
correction across 30 tests.

44. **AutoTune's primary value is workload-aware preset selection, not layer
    protection.** Chat: k8v4 gives 2.25× compression at –0.02% PPL (utility
    +19.4%). RAG: 4bit_nc gives 3.00× at +0.98% PPL (utility +57.6%).
    Batch: 4bit_nc gives 3.00× at +0.98% PPL (utility +14.8%). All validated
    on held-out traces not seen during screening or tuning.
45. **Naive preset choice silently violates quality thresholds.** A user
    applying 4bit_nc to chat gets dPPL=+0.98%, exceeding the 0.5% chat
    threshold → utility drops to zero. AutoTune selects k8v4 for chat
    (dPPL=–0.02%), avoiding the quality regression. This is the strongest
    argument for per-profile tuning.
46. **Layer protection does not improve utility on held-out data.** 4bit_nc
    with floor-only protection (R_mem=3.00×, utility 1.62 RAG) beats
    4bit_nc+L5 (R_mem=2.82×, utility 1.58). Protection improves PPL
    (10.449→10.442) but costs R_mem, and since both configs pass the PPL
    threshold, the PPL gain has zero utility value. Protection only matters
    at threshold-crossing boundaries.
47. **H1b is a negative result: stats-guided does not beat positional.**
    Positional protection (L2, PPL 10.436) gives better PPL than
    stats-guided (L5, PPL 10.442), significant on paired t-test (p<0.001).
    Utility difference is not significant (p=0.33). Layer 5 has high
    quant_error but zero sensitivity (finding #25) — the profiler's ranking
    does not predict end-to-end protection value in vLLM.
48. **PPL is deterministic across reps** — same model, same eval data,
    teacher-forcing → identical PPL values in all 5 reps (CI ±0.000).
    t-statistics are infinite for PPL comparisons. Latency metrics have
    very low variance (TPOT ±0.01ms), confirming high system determinism
    under offline single-GPU inference.
49. **Wilcoxon is structurally underpowered at n=5 with Holm-Bonferroni.**
    Minimum two-tailed p=0.0625; with 30 tests, minimum adjusted p=1.0.
    Zero Wilcoxon tests survive correction. All 23 significant results
    rely on the parametric paired t-test.
50. **3bit_nc fails all profile thresholds on held-out data.** dPPL=+3.43%
    exceeds even the batch threshold (2.0%). Utility=0 for all profiles.
    Without AutoTune, a user might try 3bit_nc for maximum compression
    and silently degrade quality beyond any acceptable limit.

## Experiment 4 — Generalization spot-check, Qwen3-1.7B vs 4B, RTX 4090 (2026-07-26)

Source: `analysis/exp4_generalization.py`, 10 cells = 5 configs × 2 reps on
Qwen3-1.7B (28 layers). Same configs as Exp 3 primary model (Qwen3-4B, 36L).

51. **Smaller models are far more sensitive to KV-cache quantization.**
    dPPL on Qwen3-1.7B: 3bit_nc +54.6% (vs +3.4% on 4B — 16× worse),
    4bit_nc +5.9% (vs +0.98% — 6× worse). The 1.7B model has less
    redundancy per layer; quantization noise is amplified without spare
    capacity to absorb it. 3bit_nc essentially collapses the model.
52. **The optimal config is model-dependent.** RAG: 4bit_nc on 4B
    (utility 1.62) vs k8v4 on 1.7B (utility 1.10) — 4bit_nc fails the
    1% PPL threshold on 1.7B (dPPL=5.9%). Batch: 4bit_nc on 4B
    (utility 1.15) vs baseline on 1.7B — no TQ preset is viable for batch
    on 1.7B. A config tuned for one model would silently degrade quality
    on another.
53. **k8v4 shows quantization-as-regularization on the small model.**
    dPPL = -2.03% on 1.7B (PPL improves from 13.43 to 13.16). FP8 keys +
    4-bit uniform values act as noise injection that improves
    generalization on the smaller, more overfitted model. On 4B the
    effect is negligible (-0.02%).
54. **Protection is more valuable on sensitive models.** 4bit_nc+L5 vs
    4bit_nc: PPL recovery = 0.77 percentage points on 1.7B (5.90→5.13%)
    vs 0.07 on 4B (0.98→0.91%). Protecting layer 5 recovers 11× more PPL
    on the smaller model, consistent with finding #32 (protection scales
    with quantization aggressiveness / model sensitivity).
55. **Floor protection costs more on small models.** R_mem for 4bit_nc:
    2.80× on 1.7B (4/28 = 14.3% floor) vs 3.00× on 4B (4/36 = 11.1%).
    The fixed 4-layer floor is a larger fraction of the smaller model,
    reducing effective compression — another reason per-model tuning
    matters.

## Profiler methodology — refactoring insights (2026-07-28)

56. **Simulation mode reimplements TurboQuant math in pure PyTorch** (~100 lines:
    Hadamard rotation + Lloyd-Max centroids for keys, uniform quantization for
    values). This was originally the only profiling method. The reimplementation
    replicates vLLM's CUDA kernel logic in Python for per-layer hook-based
    profiling via HuggingFace Transformers. vLLM's quantization CUDA kernels
    cannot be called from HuggingFace hooks — they operate inside vLLM's engine,
    not as standalone Python functions. The simulation is documented as a
    fallback, not a replacement for vLLM.
57. **vLLM ground-truth mode added** (`--mode vllm`): generates a harness manifest
    where each cell quantizes ONLY one layer through vLLM's actual engine via
    `kv_cache_dtype_skip_layers` (all other layers protected as FP16). No
    simulation — uses real CUDA kernels. Slower (~1.5h for 36 layers) but
    provides ground truth to validate the simulation results from Experiment 0.
    Two-step workflow: `--mode vllm` generates the manifest, run through
    harness, then `--mode vllm-analyze` computes the sensitivity ranking.
58. **Utility parameters extracted to config** (`configs/profiles.json`). PPL
    thresholds and utility function exponents were hard-coded identically in
    4 files (`tuner.py`, `advisor.py`, `exp3_validation.py`,
    `exp4_generalization.py`). Now loaded from a single JSON file with a shared
    loader module (`src/profiles.py`) — defaults baked in as fallback.
59. **Legacy CLI modes renamed for clarity.** `--mode sensitivity` → `--mode
    sim-sensitivity`, `--mode full` → `--mode sim-full` (old names kept as
    aliases). New modes: `--mode vllm`, `--mode vllm-analyze`. Makes explicit
    which backend (simulation vs vLLM engine) is being used.

## Methodology / infrastructure lessons

9. **vLLM v1 is multi-process**: `torch.cuda.max_memory_allocated()` in the
   client process reads 0 — VRAM must be measured device-level via NVML.
   (Smoke-test `peak_vram_gb: 0.0` is this bug, not a real number.)
10. **Reproducibility trap**: `vllm==0.20.2` ships sdist-only → pip silently
    attempts an hours-long CUDA source build. Pin `0.21.0` (prebuilt wheel)
    and install with `--only-binary :all:`.
20. **vLLM V1 offline `LLM.generate()` requires `disable_log_stats=False`** to
    populate `RequestOutput.metrics`. With `True` (which we initially set to
    reduce noise), all per-request latencies are `None`. The V1 metrics object
    is `RequestStateStats`, not the V0 `RequestMetrics` — field names differ
    (`first_token_latency` vs. `first_token_time`; monotonic timestamps vs.
    wall-clock). Source: Phase 2 harness debugging.
21. **HuggingFace `datasets` namespace change**: `load_dataset("wikitext", ...)`
    fails on recent `huggingface_hub` versions — must use
    `"Salesforce/wikitext"`. Silent breakage if not caught. Source: harness
    PPL computation failure on cluster.
