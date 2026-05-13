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

"""Kernel 1 (Context Attention): correctness tests via interface.

Validates that _context_attention (with architecture auto-dispatch)
produces results matching beam_context_attention_ref.

Usage:
    pytest tests/test_context.py -v                  # full suite (144 cases)
    pytest tests/test_context.py -v -k "d128"        # d=128 only
    pytest tests/test_context.py -v -k "bw512"       # beam_width=512 only
    python tests/test_context.py                      # quick smoke test
"""

import math

import pytest
import torch

from tests.reference import beam_context_attention_ref, generate_test_data
from interface import _context_attention


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------

def _test_single(batch, beam_width, seqlen_context, head_q, head_kv, dim,
                 dtype=torch.bfloat16):
    """Run a single correctness test through the interface."""
    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    softmax_scale = 1.0 / math.sqrt(dim)

    q, k_context, v_context, _, _, _, _ = generate_test_data(
        batch=batch, seqlen_q=1, beam_width=beam_width, head_q=head_q,
        head_kv=head_kv, dim=dim, seqlen_context=seqlen_context,
        decode_nums=0, dtype=dtype, device=device,
    )

    # fp32 reference (ground truth)
    out_ref, lse_ref = beam_context_attention_ref(q, k_context, v_context, softmax_scale)

    # Kernel under test: through interface (auto arch dispatch)
    B, _, W, Hq, D = q.shape
    out = torch.empty(B, W, Hq, D, device=device, dtype=torch.float32)
    lse = torch.empty(B, Hq, W, device=device, dtype=torch.float32)
    _context_attention(q, k_context, v_context, softmax_scale, out=out, lse=lse)
    out = out.unsqueeze(1)  # → [B, 1, W, Hq, D]
    lse = lse.transpose(-1, -2).unsqueeze(1)  # → [B, 1, W, Hq]

    # FA-style tolerance
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2
    kernel_diff = (out.float() - out_ref).abs().max().item()
    # Native dtype baseline for pt_diff
    q_f, k_f, v_f = q.float(), k_context.float(), v_context.float()
    ngroups = head_q // head_kv
    if ngroups > 1:
        k_f = k_f.repeat_interleave(ngroups, dim=2)
        v_f = v_f.repeat_interleave(ngroups, dim=2)
    scores_pt = torch.einsum('bqwhd,bshd->bqwhs', q.float() * softmax_scale, k_f)
    attn_pt = torch.softmax(scores_pt, dim=-1).to(dtype).float()
    out_pt = torch.einsum('bqwhs,bshd->bqwhd', attn_pt, v_f)
    pt_diff = (out_pt - out_ref).abs().max().item()
    fwd_tol = rtol * pt_diff + fwd_atol

    # LSE check
    finite_mask = lse.isfinite() & lse_ref.isfinite()
    lse_diff = (
        (lse - lse_ref).abs()[finite_mask].max().item()
        if finite_mask.any() else 0.0
    )
    lse_tol = 1e-3

    gqa_str = "mha" if head_q == head_kv else f"gqa({head_q}/{head_kv})"
    passed = (kernel_diff <= fwd_tol) and (lse_diff <= lse_tol)
    tag = "PASS" if passed else "FAIL"
    print(
        f"  {tag} bs={batch} bw={beam_width} ctx={seqlen_context} {gqa_str} d={dim}: "
        f"kernel={kernel_diff:.6f} pt={pt_diff:.6f} tol={fwd_tol:.6f} "
        f"lse_diff={lse_diff:.6f}"
    )
    assert passed, (
        f"kernel_diff={kernel_diff:.6f} > fwd_tol={fwd_tol:.6f}, "
        f"lse_diff={lse_diff:.6f} > lse_tol={lse_tol}"
    )


# ---------------------------------------------------------------------------
# Pytest parametrized tests (144 cases: 2 dim × 3 hkv × 4 bw × 6 ctx)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [64, 128], ids=lambda d: f"d{d}")
@pytest.mark.parametrize("head_kv", [1, 4, 16], ids=lambda h: f"hkv{h}")
@pytest.mark.parametrize("beam_width", [128, 256, 512, 1024], ids=lambda w: f"bw{w}")
@pytest.mark.parametrize("seqlen_context", [1000, 1024, 2000, 2048, 4000, 4096],
                         ids=lambda s: f"ctx{s}")
def test_context_attention(dim, head_kv, beam_width, seqlen_context):
    batch = 4
    head_q = 16
    _test_single(batch, beam_width, seqlen_context, head_q, head_kv, dim)


# ---------------------------------------------------------------------------
# Quick smoke test (python tests/test_context.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Kernel 1: Context Attention tests (via interface)")
    print("=" * 70)

    configs = [
        # (batch, beam_width, seqlen_context, head_q, head_kv, dim)
        (4, 128,  1024, 16, 16, 128),
        (4, 256,  2048, 16, 16,  64),
        (4, 512,  4096, 16, 16, 128),
        (4, 128,  1000, 16, 4, 128),
        (4, 1024, 2000, 16, 4,  64),
        (4, 256,  4000, 16, 4, 128),
        (4, 512,  1024, 16, 1, 128),
        (4, 1024, 4096, 16, 1,  64),
    ]
    for batch, bw, ctx, hq, hkv, d in configs:
        _test_single(batch, bw, ctx, hq, hkv, d)

    print("=" * 70)
    print("All quick tests passed.")
