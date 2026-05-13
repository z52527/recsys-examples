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

"""
PyTorch reference implementations for beam search decode attention.

Tensor shapes:
    Q:            [batch, seqlen_q, beam_width, head_q, dim]
    K/V context:  [batch, seqlen_context, head_kv, dim]
    K/V beam:     [batch, decode_nums * beam_width, head_kv, dim]   (decode-step-major)
    topK indices: [batch, seqlen_q, head_q, max_decode_nums, beam_width]
                  Absolute indices into dim-1 of K/V beam.

Three kernels:
    1. beam_context_attention_ref  — dense attention Q vs shared context KV
    2. beam_sparse_attention_ref   — sparse attention via topK gather from beam KV
    3. beam_attention_combine_ref  — log-sum-exp merge of two partial results
"""

import math
import torch
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------

def generate_test_data(
    batch, seqlen_q, beam_width, head_q, head_kv, dim,
    seqlen_context, decode_nums, max_decode_nums=None,
    dtype=torch.bfloat16, device="cuda",
):
    """Generate random tensors for beam attention testing.

    Args:
        batch: Batch size.
        seqlen_q: Query sequence length (1 for decode).
        beam_width: Number of parallel beams in beam search.
        head_q: Number of query heads.
        head_kv: Number of KV heads (head_q must be divisible by head_kv).
        dim: Head dimension.
        seqlen_context: Context KV sequence length.
        decode_nums: Number of decoded steps; beam KV has decode_nums * beam_width entries.
        max_decode_nums: Allocated capacity for topk_indices dim-3 (default: decode_nums + 16).
        dtype: Data type for Q/K/V tensors.
        device: CUDA device.

    Returns:
        q:            [batch, seqlen_q, beam_width, head_q, dim]
        k_context:    [batch, seqlen_context, head_kv, dim]
        v_context:    [batch, seqlen_context, head_kv, dim]
        k_beam:       [batch, decode_nums * beam_width, head_kv, dim]
        v_beam:       [batch, decode_nums * beam_width, head_kv, dim]
        topk_indices: [batch, seqlen_q, head_q, max_decode_nums, beam_width]
        decode_nums:  int
    """
    if max_decode_nums is None:
        max_decode_nums = decode_nums + 16

    q = torch.randn(batch, seqlen_q, beam_width, head_q, dim,
                     dtype=dtype, device=device)
    k_context = torch.randn(batch, seqlen_context, head_kv, dim,
                            dtype=dtype, device=device)
    v_context = torch.randn(batch, seqlen_context, head_kv, dim,
                            dtype=dtype, device=device)

    seqlen_beam = decode_nums * beam_width
    k_beam = torch.randn(batch, seqlen_beam, head_kv, dim,
                         dtype=dtype, device=device)
    v_beam = torch.randn(batch, seqlen_beam, head_kv, dim,
                         dtype=dtype, device=device)

    # topk_indices: same GQA group shares identical indices (beam search invariant).
    qhead_per_kv = head_q // head_kv
    topk_kv = torch.randint(
        0, max(seqlen_beam, 1),
        (batch, seqlen_q, head_kv, max_decode_nums, beam_width),
        dtype=torch.int32, device=device,
    )
    topk_indices = topk_kv if qhead_per_kv == 1 else topk_kv.repeat_interleave(qhead_per_kv, dim=2)

    return q, k_context, v_context, k_beam, v_beam, topk_indices, decode_nums


# ---------------------------------------------------------------------------
# Kernel 1: Context Attention (dense, tensor-core friendly)
# ---------------------------------------------------------------------------

def beam_context_attention_ref(
    q: torch.Tensor,
    k_context: torch.Tensor,
    v_context: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full dense attention between Q and context KV (shared across beams).

    Args:
        q:         [batch, seqlen_q, beam_width, head_q, dim]
        k_context: [batch, seqlen_context, head_kv, dim]
        v_context: [batch, seqlen_context, head_kv, dim]

    Returns:
        out: [batch, seqlen_q, beam_width, head_q, dim]  fp32
        lse: [batch, seqlen_q, beam_width, head_q]        fp32
    """
    batch, seqlen_q, beam_width, head_q, dim = q.shape
    seqlen_context = k_context.shape[1]
    head_kv = k_context.shape[2]
    ngroups = head_q // head_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    if seqlen_context == 0:
        out = torch.zeros(batch, seqlen_q, beam_width, head_q, dim,
                          device=q.device, dtype=torch.float32)
        lse = torch.full((batch, seqlen_q, beam_width, head_q), float('-inf'),
                         device=q.device, dtype=torch.float32)
        return out, lse

    q_f = q.float()
    k_f = k_context.float()
    v_f = v_context.float()

    if ngroups > 1:
        k_f = k_f.repeat_interleave(ngroups, dim=2)
        v_f = v_f.repeat_interleave(ngroups, dim=2)

    # scores: [batch, seqlen_q, beam_width, head_q, seqlen_context]
    scores = torch.einsum('bqwhd,bshd->bqwhs', q_f * softmax_scale, k_f)

    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('bqwhs,bshd->bqwhd', attn, v_f)

    return out, lse


# ---------------------------------------------------------------------------
# Kernel 2: Sparse Beam Attention (CUDA-core, gather via topK)
# ---------------------------------------------------------------------------

def beam_sparse_attention_ref(
    q: torch.Tensor,
    k_beam: torch.Tensor,
    v_beam: torch.Tensor,
    topk_indices: torch.Tensor,
    decode_nums: int,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention: gather KV by topK indices, then compute attention.

    Args:
        q:            [batch, seqlen_q, beam_width, head_q, dim]
        k_beam:       [batch, decode_nums * beam_width, head_kv, dim]
        v_beam:       [batch, decode_nums * beam_width, head_kv, dim]
        topk_indices: [batch, seqlen_q, head_q, max_decode_nums, beam_width]
        decode_nums:  number of valid entries along dim-3 of topk_indices

    Returns:
        out: [batch, seqlen_q, beam_width, head_q, dim]  fp32
        lse: [batch, seqlen_q, beam_width, head_q]        fp32
    """
    batch, seqlen_q, beam_width, head_q, dim = q.shape
    head_kv = k_beam.shape[2]
    ngroups = head_q // head_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    if decode_nums == 0:
        out = torch.zeros(batch, seqlen_q, beam_width, head_q, dim,
                          device=q.device, dtype=torch.float32)
        lse = torch.full((batch, seqlen_q, beam_width, head_q), float('-inf'),
                         device=q.device, dtype=torch.float32)
        return out, lse

    q_f = q.float()
    k_f = k_beam.float()
    v_f = v_beam.float()

    if ngroups > 1:
        k_f = k_f.repeat_interleave(ngroups, dim=2)
        v_f = v_f.repeat_interleave(ngroups, dim=2)

    # [batch, seqlen_q, head_q, decode_nums, beam_width]
    idx = topk_indices[:, :, :, :decode_nums, :]
    # -> [batch, seqlen_q, beam_width, head_q, decode_nums]
    idx = idx.permute(0, 1, 4, 2, 3).contiguous()

    # Advanced indexing to gather selected KV
    # k_f: [batch, seqlen_kv, head_q, dim]
    b_idx = torch.arange(batch, device=q.device)[:, None, None, None, None]
    h_idx = torch.arange(head_q, device=q.device)[None, None, None, :, None]

    k_gathered = k_f[b_idx, idx, h_idx]   # [B, Sq, W, Hq, Dn, D]
    v_gathered = v_f[b_idx, idx, h_idx]

    scores = torch.einsum('bqwhd,bqwhnd->bqwhn', q_f * softmax_scale, k_gathered)

    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('bqwhn,bqwhnd->bqwhd', attn, v_gathered)

    return out, lse


# ---------------------------------------------------------------------------
# Kernel 3: Combine (log-sum-exp merge)
# ---------------------------------------------------------------------------

def beam_attention_combine_ref(
    out1: torch.Tensor,
    lse1: torch.Tensor,
    out2: torch.Tensor,
    lse2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine two partial attention results via numerically-stable log-sum-exp.

    Math:
        lse  = log(exp(lse1) + exp(lse2))
        out  = exp(lse1 - lse) * out1 + exp(lse2 - lse) * out2

    Args:
        out1, out2: [batch, seqlen_q, beam_width, head_q, dim]  fp32
        lse1, lse2: [batch, seqlen_q, beam_width, head_q]       fp32

    Returns:
        out: [batch, seqlen_q, beam_width, head_q, dim]  fp32
        lse: [batch, seqlen_q, beam_width, head_q]        fp32
    """
    m = torch.maximum(lse1, lse2)
    safe_m = torch.where(m.isinf() & (m < 0), torch.zeros_like(m), m)

    exp1 = torch.exp(lse1 - safe_m)
    exp2 = torch.exp(lse2 - safe_m)
    s = exp1 + exp2

    lse = safe_m + torch.log(s)
    lse = torch.where(s == 0, torch.full_like(lse, float('-inf')), lse)

    inv_s = torch.where(s == 0, torch.zeros_like(s), 1.0 / s)
    scale1 = (exp1 * inv_s).unsqueeze(-1)
    scale2 = (exp2 * inv_s).unsqueeze(-1)

    out = scale1 * out1 + scale2 * out2
    return out, lse


# ---------------------------------------------------------------------------
# Golden reference: single-pass attention (for correctness validation)
# ---------------------------------------------------------------------------

def beam_attention_ref(
    q: torch.Tensor,
    k_context: torch.Tensor,
    v_context: torch.Tensor,
    k_beam: torch.Tensor,
    v_beam: torch.Tensor,
    topk_indices: torch.Tensor,
    decode_nums: int,
    softmax_scale: Optional[float] = None,
    upcast: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Concatenate context KV + gathered beam KV, run single-pass attention.

    Mathematically equivalent to the 3-kernel pipeline.  Used solely to
    validate that the decomposition is correct.

    Args:
        upcast: If True (default), compute in fp32 for maximum precision.
            If False, keep the original dtype (e.g. bf16) to establish
            the numerical error baseline of low-precision arithmetic.
    """
    batch, seqlen_q, beam_width, head_q, dim = q.shape
    seqlen_context = k_context.shape[1]
    head_kv = k_context.shape[2]
    ngroups = head_q // head_kv

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(dim)

    if upcast:
        q_f = q.float()
        k_ctx_f = k_context.float()
        v_ctx_f = v_context.float()
        k_beam_f = k_beam.float()
        v_beam_f = v_beam.float()
    else:
        q_f = q
        k_ctx_f = k_context
        v_ctx_f = v_context
        k_beam_f = k_beam
        v_beam_f = v_beam

    if ngroups > 1:
        k_ctx_f = k_ctx_f.repeat_interleave(ngroups, dim=2)
        v_ctx_f = v_ctx_f.repeat_interleave(ngroups, dim=2)
        k_beam_f = k_beam_f.repeat_interleave(ngroups, dim=2)
        v_beam_f = v_beam_f.repeat_interleave(ngroups, dim=2)

    # Context KV -> [batch, 1, 1, head_q, seqlen_context, dim]
    k_ctx_exp = k_ctx_f.permute(0, 2, 1, 3).unsqueeze(1).unsqueeze(2)
    k_ctx_exp = k_ctx_exp.expand(batch, seqlen_q, beam_width, head_q,
                                 seqlen_context, dim)
    v_ctx_exp = v_ctx_f.permute(0, 2, 1, 3).unsqueeze(1).unsqueeze(2)
    v_ctx_exp = v_ctx_exp.expand(batch, seqlen_q, beam_width, head_q,
                                 seqlen_context, dim)

    if decode_nums > 0:
        idx = topk_indices[:, :, :, :decode_nums, :]
        idx = idx.permute(0, 1, 4, 2, 3).contiguous()

        b_idx = torch.arange(batch, device=q.device)[:, None, None, None, None]
        h_idx = torch.arange(head_q, device=q.device)[None, None, None, :, None]

        k_beam_g = k_beam_f[b_idx, idx, h_idx]
        v_beam_g = v_beam_f[b_idx, idx, h_idx]

        k_all = torch.cat([k_ctx_exp, k_beam_g], dim=4)
        v_all = torch.cat([v_ctx_exp, v_beam_g], dim=4)
    else:
        k_all = k_ctx_exp
        v_all = v_ctx_exp

    if k_all.shape[4] == 0:
        out = torch.zeros_like(q_f)
        lse = torch.full((batch, seqlen_q, beam_width, head_q), float('-inf'),
                         device=q.device, dtype=torch.float32)
        return out, lse

    scores = torch.einsum('bqwhd,bqwhsd->bqwhs', q_f * softmax_scale, k_all)
    lse = torch.logsumexp(scores, dim=-1)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('bqwhs,bqwhsd->bqwhd', attn, v_all)

    return out, lse
