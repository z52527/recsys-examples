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

"""Kernel 2 (Beam Sparse Attention): correctness tests via interface.

Validates that _beam_sparse_attention (sparse decode with topK gather)
produces results matching beam_sparse_attention_ref.

Usage:
    pytest tests/test_beam.py -v                     # full suite (384 cases)
    pytest tests/test_beam.py -v -k "d128"           # d=128 only
    pytest tests/test_beam.py -v -k "bw512"          # beam_width=512 only
    pytest tests/test_beam.py -v -k "dn1-"           # decode_nums=1 only
    python tests/test_beam.py                         # quick smoke test
"""

import math

import pytest
import torch

from tests.reference import beam_sparse_attention_ref, generate_test_data
from interface import _beam_sparse_attention


# ---------------------------------------------------------------------------
# Reference (native dtype baseline for FA-style tolerance)
# ---------------------------------------------------------------------------

def _sparse_attention_ref_native(q, k_beam, v_beam, topk_indices, decode_nums,
                                 softmax_scale):
    """Native-dtype baseline: same algorithm as ref but softmax in native dtype."""
    batch, seqlen_q, beam_width, head_q, dim = q.shape
    head_kv = k_beam.shape[2]
    ngroups = head_q // head_kv

    k_f = k_beam.float()
    v_f = v_beam.float()
    if ngroups > 1:
        k_f = k_f.repeat_interleave(ngroups, dim=2)
        v_f = v_f.repeat_interleave(ngroups, dim=2)

    idx = topk_indices[:, :, :, :decode_nums, :]
    idx = idx.permute(0, 1, 4, 2, 3).contiguous()

    b_idx = torch.arange(batch, device=q.device)[:, None, None, None, None]
    h_idx = torch.arange(head_q, device=q.device)[None, None, None, :, None]

    k_gathered = k_f[b_idx, idx, h_idx]
    v_gathered = v_f[b_idx, idx, h_idx]

    scores = torch.einsum('bqwhd,bqwhnd->bqwhn',
                          q.float() * softmax_scale, k_gathered)
    attn = torch.softmax(scores, dim=-1).to(q.dtype).float()
    out = torch.einsum('bqwhn,bqwhnd->bqwhd', attn, v_gathered)

    return out


# ---------------------------------------------------------------------------
# Single test case
# ---------------------------------------------------------------------------

def _test_single(batch, beam_width, head_q, head_kv, dim, decode_nums,
                 dtype=torch.bfloat16):
    device = "cuda"
    softmax_scale = 1.0 / math.sqrt(dim)
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    q, _, _, k_beam, v_beam, topk_indices, dn = generate_test_data(
        batch=batch, seqlen_q=1, beam_width=beam_width, head_q=head_q,
        head_kv=head_kv, dim=dim, seqlen_context=0, decode_nums=decode_nums,
        max_decode_nums=decode_nums, dtype=dtype, device=device,
    )

    # fp32 reference (golden)
    out_ref, lse_ref = beam_sparse_attention_ref(
        q, k_beam, v_beam, topk_indices, dn, softmax_scale,
    )

    # Native dtype baseline (precision baseline for tolerance)
    out_pt = _sparse_attention_ref_native(
        q, k_beam, v_beam, topk_indices, dn, softmax_scale,
    )

    # Kernel under test: through interface
    B, _, W, Hq, D = q.shape
    out = torch.empty(B, W, Hq, D, device='cuda', dtype=torch.float32)
    lse = torch.empty(B, Hq, W, device='cuda', dtype=torch.float32)
    _beam_sparse_attention(q, k_beam, v_beam, topk_indices, dn, softmax_scale,
                           out=out, lse=lse)
    out = out.unsqueeze(1)  # → [B, 1, W, Hq, D]
    lse = lse.transpose(-1, -2).unsqueeze(1)  # → [B, 1, W, Hq]

    # FA-style output tolerance: kernel_diff <= rtol * pt_diff + fwd_atol
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 3  # FMA accumulation order differences
    kernel_diff = (out.float() - out_ref).abs().max().item()
    pt_diff = (out_pt.float() - out_ref).abs().max().item()
    fwd_tol = rtol * pt_diff + fwd_atol

    # LSE correctness
    finite_mask = lse.isfinite() & lse_ref.isfinite()
    lse_diff = (
        (lse - lse_ref).abs()[finite_mask].max().item()
        if finite_mask.any() else 0.0
    )
    lse_tol = 1e-3

    gqa_ratio = head_q // head_kv
    passed = (kernel_diff <= fwd_tol) and (lse_diff <= lse_tol)
    tag = "PASS" if passed else "FAIL"
    print(
        f"  {tag} bs={batch} bw={beam_width} gqa={gqa_ratio} d={dim} "
        f"dn={decode_nums}: "
        f"kernel={kernel_diff:.6f} pt={pt_diff:.6f} tol={fwd_tol:.6f} "
        f"lse_diff={lse_diff:.6f}"
    )
    assert passed, (
        f"kernel_diff={kernel_diff:.6f} > fwd_tol={fwd_tol:.6f}, "
        f"lse_diff={lse_diff:.6f} > lse_tol={lse_tol}"
    )


# ---------------------------------------------------------------------------
# Pytest parametrized tests (384 cases: 2 dim × 3 hkv × 4 bw × 16 dn)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dim", [64, 128], ids=lambda d: f"d{d}")
@pytest.mark.parametrize("head_kv", [1, 4, 16], ids=lambda h: f"hkv{h}")
@pytest.mark.parametrize("beam_width", [128, 256, 512, 1024], ids=lambda w: f"bw{w}")
@pytest.mark.parametrize("decode_nums", list(range(1, 17)), ids=lambda n: f"dn{n}")
def test_sparse_decode(dim, head_kv, beam_width, decode_nums):
    batch = 4
    head_q = 16
    _test_single(batch, beam_width, head_q, head_kv, dim, decode_nums)


# ---------------------------------------------------------------------------
# Quick smoke test (python tests/test_beam.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Kernel 2: Beam Sparse Attention tests (via interface)")
    print("=" * 70)

    configs = [
        # (batch, beam_width, head_q, head_kv, dim, decode_nums)
        (4, 128,  16, 16, 128, 1),
        (4, 256,  16, 16,  64, 8),
        (4, 512,  16, 16, 128, 16),
        (4, 128,  16, 4, 128, 4),
        (4, 1024, 16, 4,  64, 16),
        (4, 256,  16, 1, 128, 1),
        (4, 512,  16, 1,  64, 8),
        (4, 1024, 16, 1, 128, 16),
    ]
    for batch, bw, hq, hkv, d, dn in configs:
        _test_single(batch, bw, hq, hkv, d, dn)

    print("=" * 70)
    print("All quick tests passed.")
