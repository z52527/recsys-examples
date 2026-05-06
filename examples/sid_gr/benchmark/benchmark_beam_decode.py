# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""
Benchmark: SIDGRModel.generate (jiayus FA + arbitrary mask) vs
           SIDGRModel.generate_beam_decode (Jerry's beam_decode_attn kernel).

Both paths share the same model weights (use_jagged_flash_attn=True). The
difference is the attention path during generation:
  - generate(): re-runs full transformer over [history + all_generated]
    every hierarchy step, with arbitrary_func mask isolating beams.
  - generate_beam_decode(): prefill once → KV cache; per-step decode reuses
    cached context KV and accumulates beam KV via topk_indices.

Run inside the recsys-examples Docker container (commons + dynamicemb +
megatron + torchrec must import cleanly). Required PYTHONPATH must include
the gr-decode_atten clone for the CuTe kernel.

Example:
  cd examples/sid_gr
  PYTHONPATH=/path/to/gr-decode_atten:$PYTHONPATH \
    torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
    --max_hist_len 128 --beam_width 10 --num_layers 4
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Callable, Dict, List

import torch

# Make tests/ importable so we can reuse the model factory
_SID_GR_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SID_GR_ROOT not in sys.path:
    sys.path.insert(0, _SID_GR_ROOT)

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


def run_one_config(args) -> None:
    init.set_random_seed(args.seed)

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
        dtype=torch.bfloat16,
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
    # Override beam width via the BeamSearch instance
    model_unwrapped.beam_search.beam_widths = [args.beam_width] * args.num_hierarchies

    batch = build_random_batch(
        batch_size=args.batch_size,
        max_history_length=args.max_hist_len,
        codebook_sizes=codebook_sizes,
        hist_name=hist_name,
        cand_name=cand_name,
    )

    # Functions to time
    @torch.no_grad()
    def run_original():
        sids, _ = model_unwrapped.generate(batch)
        return sids

    @torch.no_grad()
    def run_beam_decode():
        sids, _ = model_unwrapped.generate_beam_decode(batch)
        return sids

    # Sanity: both produce valid outputs
    sids_a = run_original()
    sids_b = run_beam_decode()
    assert sids_a.shape == sids_b.shape, (
        f"shape mismatch: orig={sids_a.shape}, decode={sids_b.shape}"
    )

    print("=" * 80)
    print(
        f"Config: batch={args.batch_size}, hist_len={args.max_hist_len}, "
        f"beam_w={args.beam_width}, hierarchies={args.num_hierarchies}, "
        f"codebook={args.codebook_size}, hidden={args.hidden_size}, "
        f"heads={args.num_heads}, layers={args.num_layers}"
    )
    print(f"warmup={args.num_warmup}, iter={args.num_iter}")
    print("-" * 80)

    print("[1/2] Timing generate() (jiayus FA + arbitrary mask)...")
    stats_orig = time_fn(run_original, args.num_warmup, args.num_iter)
    print(f"  median={stats_orig['median_ms']:.3f} ms, "
          f"mean={stats_orig['mean_ms']:.3f} ms, "
          f"p95={stats_orig['p95_ms']:.3f} ms, "
          f"stdev={stats_orig['stdev_ms']:.3f} ms")

    print("[2/2] Timing generate_beam_decode() (Jerry kernel, 3-kernel backend)...")
    stats_decode = time_fn(run_beam_decode, args.num_warmup, args.num_iter)
    print(f"  median={stats_decode['median_ms']:.3f} ms, "
          f"mean={stats_decode['mean_ms']:.3f} ms, "
          f"p95={stats_decode['p95_ms']:.3f} ms, "
          f"stdev={stats_decode['stdev_ms']:.3f} ms")

    speedup = stats_orig["median_ms"] / stats_decode["median_ms"]
    print("-" * 80)
    print(f"Speedup (median orig / median decode) = {speedup:.2f}x")
    print("=" * 80)


def run_sweep(base_args) -> None:
    """Run a sweep over (max_hist_len, beam_width) and print a markdown table."""
    hist_lens = [int(x) for x in base_args.sweep_hist.split(",")]
    beam_widths = [int(x) for x in base_args.sweep_beam.split(",")]

    rows = []
    for hl in hist_lens:
        for bw in beam_widths:
            cfg = argparse.Namespace(**vars(base_args))
            cfg.max_hist_len = hl
            cfg.beam_width = bw

            init.set_random_seed(cfg.seed)
            hist_name = "hist_sids"
            cand_name = "cand_sids"
            codebook_sizes = [cfg.codebook_size] * cfg.num_hierarchies
            codebook_embedding_config = ShardedEmbeddingConfig(
                feature_names=[hist_name, cand_name],
                table_name="codebook",
                vocab_size=sum(codebook_sizes),
                dim=cfg.hidden_size,
                sharding_type="data_parallel",
            )
            model, optimizer = create_sid_gr_model_and_optimizer(
                dtype=torch.bfloat16,
                hidden_size=cfg.hidden_size,
                num_attention_heads=cfg.num_heads,
                kv_channels=cfg.kv_channels,
                num_layers=cfg.num_layers,
                num_hierarchies=cfg.num_hierarchies,
                codebook_embedding_config=codebook_embedding_config,
                codebook_sizes=codebook_sizes,
                use_jagged_flash_attn=True,
            )
            optimizer.reload_model_params()
            model_unwrapped = get_unwrapped_module(model)
            model_unwrapped.eval()
            model_unwrapped.beam_search.beam_widths = [bw] * cfg.num_hierarchies
            batch = build_random_batch(
                batch_size=cfg.batch_size,
                max_history_length=hl,
                codebook_sizes=codebook_sizes,
                hist_name=hist_name,
                cand_name=cand_name,
            )

            @torch.no_grad()
            def run_orig():
                model_unwrapped.generate(batch)

            @torch.no_grad()
            def run_decode():
                model_unwrapped.generate_beam_decode(batch)

            stats_o = time_fn(run_orig, cfg.num_warmup, cfg.num_iter)
            stats_d = time_fn(run_decode, cfg.num_warmup, cfg.num_iter)
            rows.append(
                {
                    "hist_len": hl,
                    "beam_w": bw,
                    "orig_med": stats_o["median_ms"],
                    "decode_med": stats_d["median_ms"],
                    "speedup": stats_o["median_ms"] / stats_d["median_ms"],
                }
            )
            print(
                f"[hl={hl}, bw={bw}]  orig={stats_o['median_ms']:.2f} ms, "
                f"decode={stats_d['median_ms']:.2f} ms, "
                f"speedup={rows[-1]['speedup']:.2f}x"
            )

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
    print("| hist_len | beam_w | generate (ms) | generate_beam_decode (ms) | speedup |")
    print("|---------:|-------:|--------------:|--------------------------:|--------:|")
    for r in rows:
        print(
            f"| {r['hist_len']:>8} | {r['beam_w']:>6} | "
            f"{r['orig_med']:>13.2f} | {r['decode_med']:>25.2f} | "
            f"{r['speedup']:>6.2f}x |"
        )


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
                        help="Run a sweep over (hist_len, beam_w) and print a markdown table.")
    parser.add_argument("--sweep_hist", default="32,64,128,256",
                        help="Comma-separated hist_len values for sweep.")
    parser.add_argument("--sweep_beam", default="4,10,20",
                        help="Comma-separated beam_width values for sweep.")
    args = parser.parse_args()

    init.initialize_distributed()
    init.initialize_model_parallel(1)

    with init.auto_destroy_global_state():
        if args.sweep:
            run_sweep(args)
        else:
            run_one_config(args)


if __name__ == "__main__":
    main()
