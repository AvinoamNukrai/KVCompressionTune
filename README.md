# KVCompressionTune

**Course:** LLM Caching — Final Project  
**Author:** Avinoam Nukrai  
**Baseline:** Stock vLLM (>= 0.20) with TurboQuant KV-cache quantization

---

## What This Project Investigates

vLLM ships TurboQuant KV-cache quantization with **4 fixed presets** and a **hard-coded layer-protection rule** (first 2 + last 2 layers stay FP16). These defaults were validated on a handful of Qwen models. Nobody has asked: *does compression policy actually matter?*

This project proves that it does. We show three things:

1. **Layer sensitivity is non-uniform.** Layer 0 of Qwen3-4B is 10x more sensitive to quantization than any other layer. Layer 1 (protected by default) is essentially insensitive. The fixed protection rule wastes budget on safe layers and leaves sensitive layers exposed.

2. **Sensitivity is model-dependent.** 3-bit quantization costs +3.4% PPL on Qwen3-4B but +54.6% on Qwen3-1.7B — a 16x difference. The same preset that works on one model destroys another. There is no universal safe choice.

3. **The optimal config depends on the workload.** Chat (latency-sensitive, strict quality) selects k8v4. RAG (memory-sensitive) selects 4bit_nc. Batch (throughput-maximizing) selects 4bit_nc with fewer protections. A single default cannot serve all three.

## How It Works

The system has four components, all running on **unmodified stock vLLM**:

- **Layer Sensitivity Profiler** (`src/profiler.py`) — measures per-layer sensitivity to quantization. Two backends: a fast simulation mode (~5 min, pure PyTorch reimplementation of TurboQuant math) and a vLLM ground-truth mode (~1.5h, runs the actual engine per-layer via `kv_cache_dtype_skip_layers`).

- **Benchmark Harness** (`src/harness.py`) — launches vLLM with a given config, replays frozen workload traces, collects PPL/latency/VRAM. Content-addressed checkpointing — rerunning skips completed cells.

- **Workload-Aware Tuner** (`src/tuner.py`) — searches the (preset x layer-protection budget) space using Optuna, guided by workload-specific utility functions that balance speed, memory, and quality differently per use case (chat/rag/batch).

- **Config Advisor** (`src/advisor.py`) — given a workload profile, outputs the recommended vLLM launch flags with evidence trail explaining why.

## Key Findings

| Workload | Optimal Preset | Compression | PPL Delta | vs Default |
|----------|---------------|-------------|-----------|------------|
| Chat     | k8v4          | 2.25x       | -0.02%    | Prevents quality violation (+0.98% with naive 4bit) |
| RAG      | 4bit_nc+L5    | 2.82x       | +0.91%    | +57.6% utility over baseline |
| Batch    | 4bit_nc       | 3.00x       | +0.98%    | +14.8% utility over baseline |

TurboQuant is the experimental vehicle. The contribution is the finding (compression policy matters), the methodology (per-layer sensitivity + workload-aware utility), and the practical tool.

## Technical Details

See [SPEC.md](SPEC.md) for the full specification, experimental protocol, and statistical methodology. See [INSIGHTS.md](INSIGHTS.md) for the chronological findings log.
