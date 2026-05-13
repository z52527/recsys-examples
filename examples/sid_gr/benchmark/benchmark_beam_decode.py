# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""
Benchmark: SIDGRModel.generate (arbitrary-mask FlashAttention) vs
           SIDGRModel.generate_beam_decode (CuTe beam_decode_attn kernel).

Both paths share the same model weights (use_jagged_flash_attn=True). The
difference is the attention path during generation:
  - generate(): re-runs full transformer over [history + all_generated]
    every hierarchy step, with arbitrary_func mask isolating beams.
  - generate_beam_decode(): prefill once → KV cache; per-step decode reuses
    cached context KV and accumulates beam KV via topk_indices.

Run inside the recsys-examples Docker container (commons + dynamicemb +
megatron + torchrec must import cleanly). The CuTe kernel is vendored at
corelib/gr_decode_atten/; the Dockerfile adds it to PYTHONPATH
automatically.

Example:
  cd examples/sid_gr
  torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
    --max_hist_len 128 --beam_width 10 --num_layers 4
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Tuple

import torch

# Make tests/ importable so we can reuse the model factory, and the local
# benchmark directory so _validate is reachable.
_SID_GR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SID_GR_ROOT not in sys.path:
    sys.path.insert(0, _SID_GR_ROOT)
_BENCH_DIR = os.path.dirname(__file__)
if _BENCH_DIR not in sys.path:
    sys.path.insert(0, _BENCH_DIR)

# Lightweight (no benchmark runtime deps) — imported before the heavy stack
# so the test suite can also pull validate_compare_outputs from _validate.
from _validate import validate_compare_outputs, validate_pair_outputs  # noqa: E402

# Heavy imports — require the full Docker stack
import commons.utils as init  # noqa: E402
from commons.checkpoint import get_unwrapped_module  # noqa: E402
from commons.datasets.gpt_sid_batch import FeatureConfig, GPTSIDBatch  # noqa: E402
from commons.modules.embedding import ShardedEmbeddingConfig  # noqa: E402
from commons.ops.length_to_offsets import length_to_complete_offsets  # noqa: E402
from tests.test_utils import create_sid_gr_model_and_optimizer  # noqa: E402


def time_fn(fn: Callable[[], None], num_warmup: int, num_iter: int) -> Dict[str, float]:
    """Time a callable with warmup; return latency stats in milliseconds."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    samples: List[float] = []
    for _ in range(num_iter):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    samples_sorted = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "p95_ms": samples_sorted[max(0, int(0.95 * len(samples)) - 1)],
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def build_random_batch(
    batch_size: int,
    max_history_length: int,
    codebook_sizes: List[int],
    hist_name: str,
    cand_name: str,
) -> GPTSIDBatch:
    num_hierarchies = len(codebook_sizes)
    cum_sum = length_to_complete_offsets(torch.tensor(codebook_sizes))
    raw_hist_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
    raw_cand_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]
    feature_configs = [
        FeatureConfig(
            feature_names=raw_hist_names,
            max_item_ids=cum_sum[1:],
            min_item_ids=cum_sum[:-1],
            max_sequence_length=max_history_length,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=raw_cand_names,
            max_item_ids=cum_sum[1:],
            min_item_ids=cum_sum[:-1],
            max_sequence_length=1,
            is_jagged=False,
        ),
    ]
    batch = GPTSIDBatch.random(
        batch_size=batch_size,
        feature_configs=feature_configs,
        raw_hist_sid_names=raw_hist_names,
        raw_cand_sid_names=raw_cand_names,
        combined_history_feature_name=hist_name,
        combined_candidate_feature_name=cand_name,
        contextual_feature_names=[],
        device=torch.cuda.current_device(),
    )
    batch.to(torch.cuda.current_device())
    return batch


_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Choose from {list(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[name]


def build_model(args, dtype: torch.dtype):
    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_sizes = [args.codebook_size] * args.num_hierarchies
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=args.hidden_size,
        sharding_type="data_parallel",
    )
    model, optimizer = create_sid_gr_model_and_optimizer(
        dtype=dtype,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_heads,
        kv_channels=args.kv_channels,
        num_layers=args.num_layers,
        num_hierarchies=args.num_hierarchies,
        codebook_embedding_config=codebook_embedding_config,
        codebook_sizes=codebook_sizes,
        use_jagged_flash_attn=True,
    )
    optimizer.reload_model_params()
    model_unwrapped = get_unwrapped_module(model)
    model_unwrapped.eval()
    model_unwrapped.beam_search.beam_widths = [args.beam_width] * args.num_hierarchies
    batch = build_random_batch(
        batch_size=args.batch_size,
        max_history_length=args.max_hist_len,
        codebook_sizes=codebook_sizes,
        hist_name=hist_name,
        cand_name=cand_name,
    )
    return model, optimizer, model_unwrapped, batch


def measure_phase_breakdown(
    model_unwrapped, batch, num_iter: int,
    backend: str = "3kernel", use_jagged_kv: bool = False,
) -> Dict[str, float]:
    """Average prefill_ms and decode_loop_ms across num_iter calls."""
    prefill = []
    decode = []
    for _ in range(num_iter):
        phase: Dict[str, float] = {}
        with torch.no_grad():
            model_unwrapped.generate_beam_decode(
                batch, backend=backend, use_jagged_kv=use_jagged_kv,
                phase_times=phase,
            )
        prefill.append(phase["prefill_ms"])
        decode.append(phase["decode_loop_ms"])
    return {
        "prefill_ms_median": statistics.median(prefill),
        "decode_loop_ms_median": statistics.median(decode),
    }


def run_one_config(args) -> None:
    init.set_random_seed(args.seed)
    dtype = _resolve_dtype(args.dtype)

    model, optimizer, model_unwrapped, batch = build_model(args, dtype)

    # Functions to time
    @torch.no_grad()
    def run_original():
        sids, _ = model_unwrapped.generate(batch)
        return sids

    @torch.no_grad()
    def run_beam_decode():
        sids, _ = model_unwrapped.generate_beam_decode(
            batch, backend=args.backend, use_jagged_kv=args.use_jagged_kv,
        )
        return sids

    # Sanity: both produce valid outputs
    sids_a = run_original()
    sids_b = run_beam_decode()
    assert sids_a.shape == sids_b.shape, (
        f"shape mismatch: orig={sids_a.shape}, decode={sids_b.shape}"
    )

    print("=" * 80)
    print(
        f"Config: dtype={args.dtype}, batch={args.batch_size}, "
        f"hist_len={args.max_hist_len}, beam_w={args.beam_width}, "
        f"hierarchies={args.num_hierarchies}, codebook={args.codebook_size}, "
        f"hidden={args.hidden_size}, heads={args.num_heads}, "
        f"layers={args.num_layers}"
    )
    print(f"warmup={args.num_warmup}, iter={args.num_iter}")
    print("-" * 80)

    print("[1/2] Timing generate() (arbitrary-mask FlashAttention)...")
    stats_orig = time_fn(run_original, args.num_warmup, args.num_iter)
    print(f"  median={stats_orig['median_ms']:.3f} ms, "
          f"mean={stats_orig['mean_ms']:.3f} ms, "
          f"p95={stats_orig['p95_ms']:.3f} ms, "
          f"stdev={stats_orig['stdev_ms']:.3f} ms")

    kv_label = "jagged+cu_seqlens_k" if args.use_jagged_kv else "dense+seqused_k"
    print(
        f"[2/2] Timing generate_beam_decode() "
        f"(backend={args.backend}, {kv_label})..."
    )
    stats_decode = time_fn(run_beam_decode, args.num_warmup, args.num_iter)
    print(f"  median={stats_decode['median_ms']:.3f} ms, "
          f"mean={stats_decode['mean_ms']:.3f} ms, "
          f"p95={stats_decode['p95_ms']:.3f} ms, "
          f"stdev={stats_decode['stdev_ms']:.3f} ms")

    # Phase breakdown for the beam_decode path
    phase = measure_phase_breakdown(
        model_unwrapped, batch, num_iter=args.num_iter,
        backend=args.backend, use_jagged_kv=args.use_jagged_kv,
    )
    print(f"  phase breakdown: prefill={phase['prefill_ms_median']:.3f} ms, "
          f"decode_loop={phase['decode_loop_ms_median']:.3f} ms")

    speedup = stats_orig["median_ms"] / stats_decode["median_ms"]
    print("-" * 80)
    print(f"Speedup (median orig / median decode) = {speedup:.2f}x")
    print("=" * 80)


def run_sweep(base_args) -> None:
    """Run a sweep over (max_hist_len, beam_width, dtype) and print a markdown table.

    Pass ``--validate_outputs`` to add an untimed A-vs-B correctness pass
    per config (same thresholds as ``--compare_kv_modes``: top-1 exact,
    |lp delta| < 0.15, top-K overlap >= 70%). Off by default to keep the
    headline numbers from doubling in wall time.
    """
    hist_lens = [int(x) for x in base_args.sweep_hist.split(",")]
    beam_widths = [int(x) for x in base_args.sweep_beam.split(",")]
    dtypes = [s.strip() for s in base_args.sweep_dtype.split(",")]
    validate = getattr(base_args, "validate_outputs", False)

    rows = []
    val_passes = 0
    val_fails: List[str] = []
    for dtype_name in dtypes:
        dtype = _resolve_dtype(dtype_name)
        for hl in hist_lens:
            for bw in beam_widths:
                cfg = argparse.Namespace(**vars(base_args))
                cfg.max_hist_len = hl
                cfg.beam_width = bw
                cfg.dtype = dtype_name

                init.set_random_seed(cfg.seed)
                model, optimizer, model_unwrapped, batch = build_model(cfg, dtype)

                if validate:
                    with torch.no_grad():
                        sids_a, lp_a = model_unwrapped.generate(batch)
                        sids_b, lp_b = model_unwrapped.generate_beam_decode(
                            batch, backend=cfg.backend,
                            use_jagged_kv=cfg.use_jagged_kv,
                        )
                    val_ok, val_msg = validate_pair_outputs(
                        sids_a, lp_a, sids_b, lp_b,
                    )
                    if val_ok:
                        val_passes += 1
                    else:
                        val_fails.append(
                            f"[{dtype_name} hist={hl} bw={bw}] {val_msg}"
                        )

                @torch.no_grad()
                def run_orig():
                    model_unwrapped.generate(batch)

                @torch.no_grad()
                def run_decode():
                    model_unwrapped.generate_beam_decode(
                        batch, backend=cfg.backend, use_jagged_kv=cfg.use_jagged_kv,
                    )

                stats_o = time_fn(run_orig, cfg.num_warmup, cfg.num_iter)
                stats_d = time_fn(run_decode, cfg.num_warmup, cfg.num_iter)
                phase = measure_phase_breakdown(
                    model_unwrapped, batch, num_iter=cfg.num_iter,
                    backend=cfg.backend, use_jagged_kv=cfg.use_jagged_kv,
                )
                rows.append(
                    {
                        "dtype": dtype_name,
                        "hist_len": hl,
                        "beam_w": bw,
                        "orig_med": stats_o["median_ms"],
                        "decode_med": stats_d["median_ms"],
                        "prefill_med": phase["prefill_ms_median"],
                        "decode_loop_med": phase["decode_loop_ms_median"],
                        "speedup": stats_o["median_ms"] / stats_d["median_ms"],
                    }
                )
                line = (
                    f"[dtype={dtype_name} hl={hl} bw={bw}]  "
                    f"orig={stats_o['median_ms']:.2f} ms, "
                    f"decode={stats_d['median_ms']:.2f} ms "
                    f"(prefill={phase['prefill_ms_median']:.2f} + "
                    f"decode_loop={phase['decode_loop_ms_median']:.2f}), "
                    f"speedup={rows[-1]['speedup']:.2f}x"
                )
                if validate:
                    line += f"  [validate: {'PASS' if val_ok else 'FAIL'}]"
                print(line)

                # Free memory between configs
                del model, optimizer, model_unwrapped, batch
                torch.cuda.empty_cache()

    # Markdown table
    print()
    print("## Sweep results")
    print()
    print(
        f"Fixed: batch={base_args.batch_size}, hierarchies={base_args.num_hierarchies}, "
        f"hidden={base_args.hidden_size}, heads={base_args.num_heads}, "
        f"layers={base_args.num_layers}, codebook={base_args.codebook_size}"
    )
    print()
    print(
        "| dtype | hist_len | beam_w | generate (ms) | "
        "decode total (ms) | prefill (ms) | decode_loop (ms) | speedup |"
    )
    print(
        "|-------|---------:|-------:|--------------:|"
        "------------------:|-------------:|-----------------:|--------:|"
    )
    for r in rows:
        print(
            f"| {r['dtype']:>5} | {r['hist_len']:>8} | {r['beam_w']:>6} | "
            f"{r['orig_med']:>13.2f} | {r['decode_med']:>17.2f} | "
            f"{r['prefill_med']:>12.2f} | {r['decode_loop_med']:>16.2f} | "
            f"{r['speedup']:>6.2f}x |"
        )

    if validate:
        total = val_passes + len(val_fails)
        print()
        if val_fails:
            print(f"## Validation: {val_passes}/{total} configs PASS, "
                  f"{len(val_fails)} FAIL")
            for line in val_fails:
                print(f"  - {line}")
            if not getattr(base_args, "allow_validation_fail", False):
                raise RuntimeError(
                    f"--validate_outputs detected {len(val_fails)}/{total} "
                    f"configs that disagree between generate() and "
                    f"generate_beam_decode(). Pass --allow_validation_fail "
                    f"to continue anyway."
                )
        else:
            print(f"## Validation: {val_passes}/{total} configs PASS "
                  f"(top-1 exact match, |lp delta| < 0.15, "
                  f"top-K overlap >= 70%)")


def run_compare_kv_modes(base_args) -> None:
    """3-way sweep: generate() vs generate_beam_decode(use_jagged_kv=False)
    vs generate_beam_decode(use_jagged_kv=True). Reports per-config
    timings and a markdown table that maps to the "Jagged-native vs
    dense" section in RESULTS.md.

    Each config is validated for output equivalence before timing
    (top-1 exact match, |lp delta| < 0.15, top-K overlap >= 70%). Any
    failure raises ``RuntimeError`` after the table is printed; pass
    ``--allow_validation_fail`` to continue on failure for diagnostic
    sweeps.
    """
    hist_lens = [int(x) for x in base_args.sweep_hist.split(",")]
    beam_widths = [int(x) for x in base_args.sweep_beam.split(",")]
    dtypes = [s.strip() for s in base_args.sweep_dtype.split(",")]

    rows = []
    val_passes = 0
    val_fails: List[str] = []
    for dtype_name in dtypes:
        dtype = _resolve_dtype(dtype_name)
        for hl in hist_lens:
            for bw in beam_widths:
                cfg = argparse.Namespace(**vars(base_args))
                cfg.max_hist_len = hl
                cfg.beam_width = bw
                cfg.dtype = dtype_name

                init.set_random_seed(cfg.seed)
                model, optimizer, model_unwrapped, batch = build_model(cfg, dtype)

                # 1. Untimed correctness pass first. If thresholds fail,
                # we still time and report — but flag the validation.
                with torch.no_grad():
                    sids_a, lp_a = model_unwrapped.generate(batch)
                    sids_b, lp_b = model_unwrapped.generate_beam_decode(
                        batch, backend="3kernel", use_jagged_kv=False,
                    )
                    sids_c, lp_c = model_unwrapped.generate_beam_decode(
                        batch, backend="3kernel", use_jagged_kv=True,
                    )
                val_ok, val_msg = validate_compare_outputs(
                    sids_a, lp_a, sids_b, lp_b, sids_c, lp_c,
                )
                if val_ok:
                    val_passes += 1
                else:
                    val_fails.append(
                        f"[{dtype_name} hist={hl} bw={bw}] {val_msg}"
                    )

                @torch.no_grad()
                def run_a():  # generate()
                    model_unwrapped.generate(batch)

                @torch.no_grad()
                def run_b():  # dense + seqused_k
                    model_unwrapped.generate_beam_decode(
                        batch, backend="3kernel", use_jagged_kv=False,
                    )

                @torch.no_grad()
                def run_c():  # jagged + cu_seqlens_k
                    model_unwrapped.generate_beam_decode(
                        batch, backend="3kernel", use_jagged_kv=True,
                    )

                ms_a = time_fn(run_a, cfg.num_warmup, cfg.num_iter)["median_ms"]
                ms_b = time_fn(run_b, cfg.num_warmup, cfg.num_iter)["median_ms"]
                ms_c = time_fn(run_c, cfg.num_warmup, cfg.num_iter)["median_ms"]
                phase_b = measure_phase_breakdown(
                    model_unwrapped, batch, num_iter=cfg.num_iter,
                    backend="3kernel", use_jagged_kv=False,
                )
                phase_c = measure_phase_breakdown(
                    model_unwrapped, batch, num_iter=cfg.num_iter,
                    backend="3kernel", use_jagged_kv=True,
                )
                rows.append({
                    "dtype": dtype_name, "hist_len": hl, "beam_w": bw,
                    "ms_a": ms_a, "ms_b": ms_b, "ms_c": ms_c,
                    "pre_b": phase_b["prefill_ms_median"],
                    "dec_b": phase_b["decode_loop_ms_median"],
                    "pre_c": phase_c["prefill_ms_median"],
                    "dec_c": phase_c["decode_loop_ms_median"],
                })
                val_tag = "PASS" if val_ok else "FAIL"
                print(
                    f"[{dtype_name} hist={hl:>3} bw={bw:>2}] "
                    f"A(generate)={ms_a:5.2f}  "
                    f"B(dense)={ms_b:5.2f}(pre={phase_b['prefill_ms_median']:.2f},dec={phase_b['decode_loop_ms_median']:.2f})  "
                    f"C(jagged)={ms_c:5.2f}(pre={phase_c['prefill_ms_median']:.2f},dec={phase_c['decode_loop_ms_median']:.2f})  "
                    f"B_ms/C_ms={ms_b/ms_c:.3f}x  "  # >1: C faster, <1: B faster
                    f"[validate: {val_tag}]"
                )

                del model, optimizer, model_unwrapped, batch
                torch.cuda.empty_cache()

    print()
    print("## 3-way comparison")
    print()
    print(
        "| dtype | hist | bw | A generate | B dense | B pre | B dec "
        "| C jagged | C pre | C dec | B_ms / C_ms |"
    )
    print(
        "|-------|-----:|---:|-----------:|--------:|------:|------:"
        "|---------:|------:|------:|------------:|"
    )
    for r in rows:
        print(
            f"| {r['dtype']:>5} | {r['hist_len']:>4} | {r['beam_w']:>2} | "
            f"{r['ms_a']:>10.2f} | {r['ms_b']:>7.2f} | "
            f"{r['pre_b']:>5.2f} | {r['dec_b']:>5.2f} | "
            f"{r['ms_c']:>8.2f} | {r['pre_c']:>5.2f} | {r['dec_c']:>5.2f} | "
            f"{r['ms_b']/r['ms_c']:>.3f}x |"
        )

    total = val_passes + len(val_fails)
    print()
    if val_fails:
        print(f"## Validation: {val_passes}/{total} configs PASS, "
              f"{len(val_fails)} FAIL")
        for line in val_fails:
            print(f"  - {line}")
        if not getattr(base_args, "allow_validation_fail", False):
            raise RuntimeError(
                f"--compare_kv_modes validation failed on "
                f"{len(val_fails)}/{total} configs. Pass "
                f"--allow_validation_fail to continue anyway. See the "
                f"validation block above for per-config diagnostics."
            )
    else:
        print(f"## Validation: {val_passes}/{total} configs PASS "
              f"(top-1 exact match, |lp delta| < 0.15, "
              f"top-K overlap >= 70% on all 3 pairs)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # Workload
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_hist_len", type=int, default=128)
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--num_hierarchies", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=256)
    # Model
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--kv_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    # Timing
    parser.add_argument("--num_warmup", type=int, default=5)
    parser.add_argument("--num_iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    # Sweep
    parser.add_argument("--sweep", action="store_true",
                        help="Run a sweep over (hist_len, beam_w, dtype) and print a markdown table.")
    parser.add_argument("--sweep_hist", default="32,64,128,256",
                        help="Comma-separated hist_len values for sweep.")
    parser.add_argument("--sweep_beam", default="4,10,20",
                        help="Comma-separated beam_width values for sweep.")
    parser.add_argument("--sweep_dtype", default="bf16,fp16",
                        help="Comma-separated dtype names (bf16/fp16) for sweep.")
    parser.add_argument("--dtype", default="bf16", choices=list(_DTYPE_MAP.keys()),
                        help="Model + activation dtype (bf16 or fp16).")
    # Backend / KV-mode
    parser.add_argument("--backend", default="3kernel", choices=["3kernel", "dsl"],
                        help="beam_decode_attn backend. '3kernel' supports "
                             "variable-length history; 'dsl' requires uniform.")
    parser.add_argument("--use_jagged_kv", action="store_true",
                        help="Use the jagged-native prefill + cu_seqlens_k path. "
                             "Only valid with backend='3kernel'. Needs the "
                             "cu_seqlens_k kernel entry point (already present "
                             "in the vendored corelib/gr_decode_atten). "
                             "See RESULTS.md for the perf trade-off.")
    parser.add_argument("--compare_kv_modes", action="store_true",
                        help="Time generate(), generate_beam_decode(use_jagged_kv=False) "
                             "and generate_beam_decode(use_jagged_kv=True) side by side. "
                             "Implies sweep semantics; uses --sweep_hist/--sweep_beam.")
    parser.add_argument("--validate_outputs", action="store_true",
                        help="In --sweep mode, also run an untimed A-vs-B "
                             "output-equivalence check per config (top-1 exact, "
                             "|lp delta| < 0.15, top-K overlap >= 70%). "
                             "Off by default to keep the headline timings fast.")
    parser.add_argument("--allow_validation_fail", action="store_true",
                        help="Allow --compare_kv_modes / --validate_outputs to "
                             "exit successfully even when validation fails. "
                             "Default: validation failures raise RuntimeError.")
    args = parser.parse_args()

    init.initialize_distributed()
    init.initialize_model_parallel(1)

    with init.auto_destroy_global_state():
        if args.compare_kv_modes:
            run_compare_kv_modes(args)
        elif args.sweep:
            run_sweep(args)
        else:
            run_one_config(args)


if __name__ == "__main__":
    main()
