# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Beam decode attention benchmark and profile.

Usage:
    # Benchmark: 3-kernel vs fused vs fused+split-KV
    PYTHONPATH=. python tests/benchmark.py --mode benchmark

    # Profile (1 run, for ncu / nsys)
    PYTHONPATH=. python tests/benchmark.py --mode profile --decode_nums 1

    # Custom config
    PYTHONPATH=. python tests/benchmark.py --mode benchmark --decode_nums 1 2 3
"""

import argparse
import math
import time

import torch

from tests.reference import generate_test_data
from interface import (
    _context_attention,
    _beam_sparse_attention,
    _combine,
    beam_decode_attn,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH = 8
SEQLEN_Q = 1
BEAM_WIDTH = 256
HEAD_Q = 4
HEAD_KV = 4  # MHA
DIM = 128
SEQLEN_CONTEXT = 4096
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def benchmark_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(mode, decode_nums_list):
    device = "cuda"
    softmax_scale = 1.0 / math.sqrt(DIM)
    cc = torch.cuda.get_device_capability()[0]
    gpu_name = torch.cuda.get_device_name()

    print(f"GPU: {gpu_name} (SM{cc}0)")
    print(f"Config: bs={BATCH} bw={BEAM_WIDTH} ctx={SEQLEN_CONTEXT} "
          f"hq=hkv={HEAD_Q} d={DIM}")

    if mode == "benchmark":
        # Per-kernel timing (dn=1)
        print("\nPer-kernel timing (dn=1):")
        print("-" * 40)
        torch.manual_seed(42)
        q0, k_ctx0, v_ctx0, k_beam0, v_beam0, topk0, _ = generate_test_data(
            batch=BATCH, seqlen_q=SEQLEN_Q, beam_width=BEAM_WIDTH,
            head_q=HEAD_Q, head_kv=HEAD_KV, dim=DIM,
            seqlen_context=SEQLEN_CONTEXT, decode_nums=1, max_decode_nums=1,
            dtype=DTYPE, device=device,
        )
        B_, W_, Hq_, D__ = BATCH, BEAM_WIDTH, HEAD_Q, DIM
        op0 = torch.empty(2, B_, W_, Hq_, D__, device=device, dtype=torch.float32)
        lr0 = torch.empty(2, B_, Hq_, W_, device=device, dtype=torch.float32)
        oo0 = torch.empty(B_, W_, Hq_, D__, device=device, dtype=torch.bfloat16)
        lo0 = torch.empty(B_, Hq_, W_, device=device, dtype=torch.float32).transpose(-1, -2)
        lp0 = lr0.transpose(-1, -2)

        t_k1 = benchmark_fn(lambda: _context_attention(q0, k_ctx0, v_ctx0, softmax_scale,
                                                        out=op0[0], lse=lr0[0]))
        t_k2 = benchmark_fn(lambda: _beam_sparse_attention(q0, k_beam0, v_beam0, topk0, 1,
                                                            softmax_scale, out=op0[1], lse=lr0[1]))
        t_k3 = benchmark_fn(lambda: _combine(op0, lp0, oo0, lo0))
        print(f"  K1 Context:  {t_k1:7.3f} ms")
        print(f"  K2 Sparse:   {t_k2:7.3f} ms")
        print(f"  K3 Combine:  {t_k3:7.3f} ms")

        # Comparison table header
        print(f"\n{'dn':>4s} | {'3-Kernel':>10s} | {'Fused(dsl)':>10s} | {'Speedup':>8s}")
        print("-" * 44)

    for dn in decode_nums_list:
        torch.manual_seed(42)
        q, k_ctx, v_ctx, k_beam, v_beam, topk_idx, _ = generate_test_data(
            batch=BATCH, seqlen_q=SEQLEN_Q, beam_width=BEAM_WIDTH,
            head_q=HEAD_Q, head_kv=HEAD_KV, dim=DIM,
            seqlen_context=SEQLEN_CONTEXT, decode_nums=dn,
            max_decode_nums=dn, dtype=DTYPE, device=device,
        )

        def run_3kernel():
            beam_decode_attn(q, k_ctx, v_ctx, k_beam, v_beam, topk_idx, dn,
                             softmax_scale, return_lse=True, backend="3kernel")

        def run_fused():
            beam_decode_attn(q, k_ctx, v_ctx, k_beam, v_beam, topk_idx, dn,
                             softmax_scale, return_lse=True, backend="dsl")

        if mode == "profile":
            print(f"\ndn={dn}:")
            with torch.cuda.nvtx.range("3kernel"):
                run_3kernel()
            with torch.cuda.nvtx.range("fused_dsl"):
                run_fused()
            torch.cuda.synchronize()
            print("  Profile run complete (use ncu or nsys to capture)")

        elif mode == "benchmark":
            t_3k = benchmark_fn(run_3kernel)
            t_fused = benchmark_fn(run_fused)
            speedup = t_3k / t_fused if t_fused > 0 else 0
            print(f"{dn:4d} | {t_3k:8.3f} ms | {t_fused:8.3f} ms | {speedup:6.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Beam decode attention benchmark")
    parser.add_argument("--mode", choices=["profile", "benchmark"], default="benchmark")
    parser.add_argument("--decode_nums", type=int, nargs="+",
                        default=list(range(1, 17)))
    args = parser.parse_args()

    run(args.mode, args.decode_nums)
