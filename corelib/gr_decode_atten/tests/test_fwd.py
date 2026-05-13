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

"""End-to-end beam attention correctness test.

Tests both 3-kernel pipeline and fused kernel (with split-KV) against
golden reference, all through the interface API.

Usage:
    python tests/test_fwd.py                          # quick test
    pytest tests/test_fwd.py -v -k "3kernel"          # 3-kernel only
    pytest tests/test_fwd.py -v -k "fused"            # fused only
    pytest tests/test_fwd.py -v                        # all (672 cases)
"""


import pytest
import torch
from interface import beam_decode_attn
from tests.reference import beam_attention_ref, generate_test_data

# ============== Correctness helper ==============


def _test_single(
    batch,
    seqlen_q,
    beam_width,
    head_q,
    head_kv,
    dim,
    seqlen_context,
    decode_nums,
    dtype=torch.bfloat16,
    backend="dsl",
):
    """Run a single correctness test through the interface.

    Compares beam_decode_attn (interface API) against beam_attention_ref
    (single-pass fp32 golden truth).
    """
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.empty_cache()

    max_decode_nums = max(decode_nums + 16, decode_nums)
    data = generate_test_data(
        batch=batch,
        seqlen_q=seqlen_q,
        beam_width=beam_width,
        head_q=head_q,
        head_kv=head_kv,
        dim=dim,
        seqlen_context=seqlen_context,
        decode_nums=decode_nums,
        max_decode_nums=max_decode_nums,
        dtype=dtype,
        device=device,
    )
    q, k_ctx, v_ctx, k_beam, v_beam, topk_idx, dn = data

    # Ground truth: fp32 single-pass reference
    out_ref, lse_ref = beam_attention_ref(
        q,
        k_ctx,
        v_ctx,
        k_beam,
        v_beam,
        topk_idx,
        dn,
    )
    # bf16 precision baseline (same algorithm, native dtype)
    out_pt, _ = beam_attention_ref(
        q,
        k_ctx,
        v_ctx,
        k_beam,
        v_beam,
        topk_idx,
        dn,
        upcast=False,
    )

    # Kernel under test
    out, lse = beam_decode_attn(
        q,
        k_ctx,
        v_ctx,
        k_beam,
        v_beam,
        topk_idx,
        dn,
        return_lse=True,
        backend=backend,
    )

    # FA-style output tolerance: kernel_diff <= rtol * pt_diff + fwd_atol
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2
    kernel_diff = (out.float() - out_ref).abs().max().item()
    pt_diff = (out_pt.float() - out_ref).abs().max().item()
    fwd_tol = rtol * pt_diff + fwd_atol

    # LSE correctness (fp32 vs fp32, combine step adds ~1e-4 rounding)
    finite_mask = lse.isfinite() & lse_ref.isfinite()
    lse_diff = (
        (lse - lse_ref).abs()[finite_mask].max().item() if finite_mask.any() else 0.0
    )
    lse_tol = 1e-3

    passed = (out.dtype == dtype) and (kernel_diff <= fwd_tol) and (lse_diff <= lse_tol)

    mha_str = "mha" if head_q == head_kv else f"gqa({head_q}/{head_kv})"
    tag = "PASS" if passed else "FAIL"
    print(
        f"  {tag} [{backend}] bs={batch} sq={seqlen_q} bw={beam_width} {mha_str} d={dim} "
        f"ctx={seqlen_context} dn={decode_nums}: "
        f"kernel={kernel_diff:.6f} pt={pt_diff:.6f} tol={fwd_tol:.6f} "
        f"lse_diff={lse_diff:.6f}"
    )
    assert passed, (
        f"kernel_diff={kernel_diff} > fwd_tol={fwd_tol}, "
        f"lse_diff={lse_diff} > lse_tol={lse_tol}"
    )


# ============== 3-kernel Pytest (288 cases) ==============


@pytest.mark.parametrize("dim", [64, 128], ids=lambda d: f"d{d}")
@pytest.mark.parametrize("head_kv", [1, 4, 16], ids=lambda h: f"hkv{h}")
@pytest.mark.parametrize("beam_width", [128, 256, 512, 1024], ids=lambda w: f"bw{w}")
@pytest.mark.parametrize("decode_nums", [1, 4, 8, 16], ids=lambda n: f"dn{n}")
@pytest.mark.parametrize("seqlen_context", [256, 1024, 2048], ids=lambda s: f"ctx{s}")
def test_beam_attention_3kernel(dim, head_kv, beam_width, decode_nums, seqlen_context):
    total_kv = seqlen_context + decode_nums * beam_width
    batch = 1 if total_kv * head_kv * dim * beam_width > 256 * 1024 * 1024 else 4
    head_q = 16
    _test_single(
        batch,
        1,
        beam_width,
        head_q,
        head_kv,
        dim,
        seqlen_context,
        decode_nums,
        backend="3kernel",
    )


# ============== Fused Pytest (384 cases, covers split-KV) ==============


@pytest.mark.parametrize("dim", [64, 128], ids=lambda d: f"d{d}")
@pytest.mark.parametrize("head_kv", [1, 4, 16], ids=lambda h: f"hkv{h}")
@pytest.mark.parametrize("beam_width", [128, 256, 512, 1024], ids=lambda w: f"bw{w}")
@pytest.mark.parametrize("decode_nums", [1, 4, 8, 16], ids=lambda n: f"dn{n}")
@pytest.mark.parametrize(
    "seqlen_context", [256, 1024, 2048, 4096], ids=lambda s: f"ctx{s}"
)
def test_beam_attention_fused(dim, head_kv, beam_width, decode_nums, seqlen_context):
    total_kv = seqlen_context + decode_nums * beam_width
    batch = 1 if total_kv * head_kv * dim * beam_width > 256 * 1024 * 1024 else 4
    head_q = 16
    _test_single(
        batch,
        1,
        beam_width,
        head_q,
        head_kv,
        dim,
        seqlen_context,
        decode_nums,
        backend="dsl",
    )


# ============== Quick test (python tests/test_fwd.py) ==============


def run_quick_tests():
    print("Quick correctness tests")
    print("=" * 70)
    configs = [
        # (batch, seqlen_q, beam_width, head_q, head_kv, dim, seqlen_ctx, decode_nums, backend)
        # 3-kernel: basic coverage
        (4, 1, 128, 16, 16, 128, 256, 1, "3kernel"),
        (4, 1, 256, 16, 16, 128, 256, 8, "3kernel"),
        (4, 1, 512, 16, 16, 64, 256, 16, "3kernel"),
        (4, 1, 128, 16, 4, 128, 256, 4, "3kernel"),
        (4, 1, 1024, 16, 4, 64, 256, 16, "3kernel"),
        # Fused: no split (small ctx)
        (4, 1, 128, 16, 16, 128, 256, 1, "dsl"),
        (4, 1, 256, 16, 4, 128, 256, 8, "dsl"),
        # Fused: split-KV (large ctx)
        (4, 1, 128, 16, 16, 128, 1024, 3, "dsl"),
        (4, 1, 128, 16, 4, 128, 2048, 3, "dsl"),
        (4, 1, 128, 16, 1, 128, 2048, 3, "dsl"),
        (4, 1, 256, 16, 4, 128, 4096, 8, "dsl"),
        (4, 1, 128, 16, 4, 64, 1024, 1, "dsl"),
        # Edge: decode_nums=0
        (4, 1, 128, 16, 4, 128, 256, 0, "3kernel"),
        (4, 1, 128, 16, 4, 128, 1024, 0, "dsl"),
    ]
    for batch, sq, bw, hq, hk, d, ctx, dn, backend in configs:
        _test_single(batch, sq, bw, hq, hk, d, ctx, dn, backend=backend)
    print("=" * 70)
    print("All quick tests passed.")


# ============== Main ==============

if __name__ == "__main__":
    run_quick_tests()
