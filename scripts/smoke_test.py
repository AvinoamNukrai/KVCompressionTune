#!/usr/bin/env python3
"""Phase-1 smoke test for KVCompressionTune.

Verifies, on the actual cluster GPU, that:
  1. Every TurboQuant KV-cache preset in stock vLLM launches and generates
     coherent (non-empty, non-degenerate) output.
  2. The --kv-cache-dtype-skip-layers flag plumbs through, and we learn the
     *effective* skip list (checking whether vLLM unions our list with the
     hard-coded n=2 boundary layers — a suspected floor we cannot go below).
  3. Real per-cell cost numbers (engine load time, generation throughput,
     peak VRAM) to replace the spec's estimates in the experiment budget.

Each configuration runs in a fresh subprocess so VRAM is fully released
between engines. Results are written to results/smoke_test.json.

Usage (on a GPU node):
    python scripts/smoke_test.py                      # full sweep
    python scripts/smoke_test.py --model Qwen/Qwen3-4B
    python scripts/smoke_test.py --configs auto turboquant_3bit_nc
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

RESULT_MARKER = "===SMOKE_RESULT_JSON==="

# Each entry: name -> (kv_cache_dtype, extra skip layers requested by us)
CONFIGS: dict[str, dict] = {
    "auto": {"kv_cache_dtype": "auto", "skip_layers": None},
    "turboquant_k8v4": {"kv_cache_dtype": "turboquant_k8v4", "skip_layers": None},
    "turboquant_4bit_nc": {"kv_cache_dtype": "turboquant_4bit_nc", "skip_layers": None},
    "turboquant_k3v4_nc": {"kv_cache_dtype": "turboquant_k3v4_nc", "skip_layers": None},
    "turboquant_3bit_nc": {"kv_cache_dtype": "turboquant_3bit_nc", "skip_layers": None},
    # Checks that user-supplied skip layers plumb through, and reveals the
    # effective list after vLLM's boundary-union logic.
    "3bit_nc_extra_skip": {
        "kv_cache_dtype": "turboquant_3bit_nc",
        "skip_layers": ["10", "11"],
    },
    # Attempts an empty override: if the effective list still contains the
    # boundary layers, the n=2 floor is confirmed un-disableable on stock vLLM.
    "3bit_nc_try_no_skip": {
        "kv_cache_dtype": "turboquant_3bit_nc",
        "skip_layers": [],
    },
}

SHORT_PROMPTS = [
    "Explain in one paragraph why the sky is blue.",
    "Write a haiku about GPU memory.",
    "What is 17 * 23? Show your reasoning briefly.",
    "Name three uses of a hash table and explain one of them.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What does the acronym CPU stand for, and what does it do?",
    "Give a one-line definition of entropy.",
    "Translate 'good morning' into French and Spanish.",
]

# ~4k-token prefill probe (RAG-shaped): long context, short answer.
LONG_CONTEXT_FILLER = (
    "The quick brown fox jumps over the lazy dog near the riverbank while "
    "engineers benchmark key-value caches under heavy load. "
)


def run_single(args: argparse.Namespace) -> None:
    """Runs inside the subprocess: one engine, one config, prints JSON."""
    import torch  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    cfg = CONFIGS[args.single]
    result: dict = {"name": args.single, "config": cfg, "ok": False}

    engine_kwargs: dict = dict(
        model=args.model,
        kv_cache_dtype=cfg["kv_cache_dtype"],
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=True,
    )
    if cfg["skip_layers"] is not None:
        engine_kwargs["kv_cache_dtype_skip_layers"] = cfg["skip_layers"]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    llm = LLM(**engine_kwargs)
    result["engine_load_s"] = round(time.perf_counter() - t0, 2)

    # Effective skip list after vLLM's boundary-union logic (best effort —
    # attribute paths differ across vLLM versions).
    try:
        cache_config = llm.llm_engine.vllm_config.cache_config
        result["effective_skip_layers"] = list(cache_config.kv_cache_dtype_skip_layers)
        result["effective_cache_dtype"] = str(cache_config.cache_dtype)
    except AttributeError as exc:
        result["effective_skip_layers"] = f"unreadable: {exc}"

    # --- Short-prompt generation (chat-shaped) ---
    sp = SamplingParams(temperature=0.0, max_tokens=128)
    t0 = time.perf_counter()
    outputs = llm.generate(SHORT_PROMPTS, sp)
    gen_s = time.perf_counter() - t0
    texts = [o.outputs[0].text for o in outputs]
    n_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    result["short_gen_s"] = round(gen_s, 2)
    result["short_gen_tok_per_s"] = round(n_tokens / gen_s, 1)
    result["sample_output"] = texts[0][:200]

    # Sanity: no empty outputs, no immediate-EOS degeneration, and outputs
    # are not all identical (a classic symptom of a corrupted cache).
    empty = sum(1 for t in texts if len(t.strip()) < 3)
    result["empty_outputs"] = empty
    result["distinct_outputs"] = len(set(texts))

    # --- Long-context prefill probe (RAG-shaped, ~4k tokens) ---
    long_prompt = (
        LONG_CONTEXT_FILLER * 180
        + "\n\nQuestion: what animal appears in the text above? Answer in one word."
    )
    sp_long = SamplingParams(temperature=0.0, max_tokens=16)
    t0 = time.perf_counter()
    long_out = llm.generate([long_prompt], sp_long)
    result["long_prefill_gen_s"] = round(time.perf_counter() - t0, 2)
    result["long_answer"] = long_out[0].outputs[0].text.strip()[:80]
    result["long_answer_mentions_fox"] = "fox" in result["long_answer"].lower()

    result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    result["ok"] = empty == 0 and result["distinct_outputs"] > 1

    print(RESULT_MARKER)
    print(json.dumps(result))


def run_sweep(args: argparse.Namespace) -> None:
    """Parent mode: run each config in a fresh subprocess, aggregate results."""
    # Preflight: fail fast with a clear message instead of 7 identical failures.
    try:
        import vllm  # noqa: PLC0415
        import torch  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        sys.exit(
            f"PREFLIGHT FAILED: {exc}.\n"
            f"Interpreter: {sys.executable}\n"
            "Activate the project venv and install deps first:\n"
            "  source .venv/bin/activate && pip install -r requirements.txt"
        )
    if not torch.cuda.is_available():
        sys.exit("PREFLIGHT FAILED: torch sees no CUDA GPU on this node "
                 "(check nvidia-smi / CUDA_VISIBLE_DEVICES / srun --gres).")
    print(f"Preflight OK: vllm {vllm.__version__}, torch {torch.__version__}, "
          f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    names = args.configs or list(CONFIGS)
    results = []
    for name in names:
        print(f"\n=== [{name}] launching subprocess ===", flush=True)
        cmd = [
            sys.executable, __file__,
            "--single", name,
            "--model", args.model,
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        wall_s = round(time.perf_counter() - t0, 1)

        if RESULT_MARKER in proc.stdout:
            payload = proc.stdout.split(RESULT_MARKER, 1)[1].strip()
            res = json.loads(payload.splitlines()[0])
        else:
            res = {
                "name": name,
                "ok": False,
                "error": (proc.stderr or proc.stdout)[-2000:],
            }
        res["subprocess_wall_s"] = wall_s
        results.append(res)
        status = "OK" if res.get("ok") else "FAILED"
        print(f"=== [{name}] {status} in {wall_s}s ===", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "model": args.model,
        "gpu": _gpu_name(),
        "results": results,
    }
    out_path.write_text(json.dumps(report, indent=2))

    print(f"\n{'name':<24} {'ok':<4} {'load_s':>7} {'tok/s':>8} "
          f"{'vram_gb':>8} {'skip_layers'}")
    for r in results:
        print(f"{r['name']:<24} {str(r.get('ok')):<4} "
              f"{r.get('engine_load_s', '-'):>7} "
              f"{r.get('short_gen_tok_per_s', '-'):>8} "
              f"{r.get('peak_vram_gb', '-'):>8} "
              f"{r.get('effective_skip_layers', '-')}")
    print(f"\nFull report: {out_path}")

    n_failed = sum(1 for r in results if not r.get("ok"))
    sys.exit(1 if n_failed else 0)


def _gpu_name() -> str:
    try:
        import torch  # noqa: PLC0415
        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 — informational only
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--single", choices=list(CONFIGS), default=None,
                   help="internal: run one config in-process")
    p.add_argument("--configs", nargs="*", choices=list(CONFIGS), default=None,
                   help="subset of configs to sweep (default: all)")
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--output", default="results/smoke_test.json")
    args = p.parse_args()

    if args.single:
        run_single(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()
