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
Beam Search Decode Attention — BeamDecodeAttn

3-kernel pipeline:
  1. Context Attention:  Q × Context KV  (tensor core, SM90)
  2. Sparse Attention:   Q × Beam KV[topK]  (CUDA core)
  3. Combine:            log-sum-exp merge

External tensor shapes:
    Q:            [batch, seqlen_q, beam_width, head_q, dim]   bf16/fp16
    K/V context:  [batch, seqlen_context, head_kv, dim]        bf16/fp16
    K/V beam:     [batch, decode_nums * beam_width, head_kv, dim]  bf16/fp16
    topK indices: [batch, seqlen_q, head_q, max_decode_nums, beam_width]  int32
    Output:       [batch, seqlen_q, beam_width, head_q, dim]   same dtype as Q
    LSE:          [batch, seqlen_q, beam_width, head_q]        fp32
"""

import math
import torch
from typing import Optional, Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute

from src.common.kernel_config import KernelConfig
from src.common.cute_dsl_utils import to_cute_tensor
from src.sm80.flash_fwd import FlashAttentionForwardSm80
from src.sm90.flash_fwd import FlashAttentionForwardSm90
from src.sm100.flash_fwd import FlashAttentionForwardSm100
from src.sm120.flash_fwd import FlashAttentionForwardSm120
from src.decode.flash_fwd import FlashAttentionForwardDecode


TORCH_TO_CUTLASS = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

# Module-level compile cache (shared across all calls)
_compile_cache: dict = {}


# ---------------------------------------------------------------------------
# num_splits heuristic (ported from FA Hopper C++ heuristics.h)
# ---------------------------------------------------------------------------

def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, num_m_blocks,
                         size_one_kv_head, is_causal_or_local=False, max_splits=128):
    """Determine optimal number of KV splits for SM occupancy.

    Ported from flash-attention/hopper/heuristics.h.
    Finds the smallest num_splits achieving ≥85% of best SM efficiency.

    Args:
        total_mblocks: batch * num_kv_heads * ceil(seqlen_q / tile_m)
        num_SMs: number of streaming multiprocessors
        num_n_blocks: ceil(seqlen_k / tile_n)
        num_m_blocks: ceil(seqlen_q / tile_m) per batch-head
        size_one_kv_head: seqlen_k * dim * sizeof(dtype) * 2 (K+V)
        is_causal_or_local: causal or local attention
        max_splits: hard cap on splits (default 128)
    """
    # If SMs are ≥80% utilized, don't split (unless KV exceeds L2 cache)
    if total_mblocks >= 0.8 * num_SMs:
        size_l2 = 50 * 1024 * 1024  # 50MB assumed L2 size
        if (size_one_kv_head > size_l2 and num_m_blocks >= num_SMs * 2
                and not is_causal_or_local):
            return min((size_one_kv_head + size_l2 - 1) // size_l2, max_splits)
        return 1
    # Too few KV blocks to benefit from splitting
    if num_n_blocks <= 4:
        return 1
    # Find smallest num_splits with ≥85% of best efficiency
    max_splits = min(max_splits, num_SMs, num_n_blocks)
    max_eff = 0.0
    effs = []
    for ns in range(1, max_splits + 1):
        n_waves = total_mblocks * ns / num_SMs
        eff = n_waves / math.ceil(n_waves)
        max_eff = max(max_eff, eff)
        effs.append(eff)
    for ns in range(1, max_splits + 1):
        if effs[ns - 1] >= 0.85 * max_eff:
            return ns
    return 1



def _validate_inputs(q, k_context, v_context, k_beam, v_beam, topk_indices, decode_nums,
                     jagged_k_context=False):
    """Validate shapes, dtypes, and device placement.

    When jagged_k_context=True, k_context/v_context are 3-D
    (total_k, Hkv, D) and the per-batch length is encoded in
    cu_seqlens_k (validated separately by the caller).
    """
    assert q.dim() == 5, f"q must be 5-D [B, Sq, W, Hq, D], got {q.dim()}-D"
    batch, seqlen_q, beam_width, head_q, dim = q.shape

    if jagged_k_context:
        assert k_context.dim() == 3, (
            f"k_context must be 3-D [total_k, Hkv, D] in jagged mode, got "
            f"{k_context.dim()}-D"
        )
        assert v_context.shape == k_context.shape
        head_kv = k_context.shape[1]
        assert k_context.shape[2] == dim
    else:
        assert k_context.dim() == 4, f"k_context must be 4-D [B, Sk, Hkv, D], got {k_context.dim()}-D"
        assert v_context.shape == k_context.shape
        assert k_context.shape[0] == batch
        head_kv = k_context.shape[2]
        assert k_context.shape[3] == dim

    seqlen_beam = decode_nums * beam_width
    assert k_beam.shape == (batch, seqlen_beam, head_kv, dim), (
        f"k_beam shape: expected {(batch, seqlen_beam, head_kv, dim)}, got {tuple(k_beam.shape)}"
    )
    assert v_beam.shape == k_beam.shape

    assert topk_indices.dim() == 5
    assert topk_indices.shape[:3] == (batch, seqlen_q, head_q)
    assert topk_indices.shape[4] == beam_width
    assert decode_nums <= topk_indices.shape[3]

    assert head_q % head_kv == 0
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.dtype == k_context.dtype == v_context.dtype == k_beam.dtype == v_beam.dtype
    assert topk_indices.dtype in (torch.int32, torch.int64)
    assert q.is_cuda
    assert q.stride(-1) == 1 and k_context.stride(-1) == 1 and k_beam.stride(-1) == 1


def _num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits=128):
    """FA split-KV heuristic: target ~1 wave of blocks per SM."""
    if num_n_blocks <= 4:
        return 1
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


# ---------------------------------------------------------------------------
# Kernel 1: Context Attention
# ---------------------------------------------------------------------------

def _get_compute_capability():
    return torch.cuda.get_device_capability()[0]


_KERNEL_CLS_MAP = {
    8: FlashAttentionForwardSm80,
    9: FlashAttentionForwardSm90,
    10: FlashAttentionForwardSm100,
    11: FlashAttentionForwardSm100,   # SM100 supports SM10.x ~ SM11.x
    12: FlashAttentionForwardSm120,
}


def _context_attention(q, k_context, v_context, softmax_scale, out, lse,
                       num_splits=1, seqused_k=None, cu_seqlens_k=None):
    """K1: Q × Context KV.

    When num_splits=1: out is (B, W, Hq, D) fp32, lse is (B, Hq, W) fp32.
    When num_splits>1: out is (ns, B, W, Hq, D) fp32, lse is (ns, B, Hq, W) fp32.

    K context shape:
        - Dense  (cu_seqlens_k=None): k_context is (B, Sk, Hkv, D)
        - Jagged (cu_seqlens_k set):  k_context is (total_k, Hkv, D), and
          cu_seqlens_k is a [B+1] int32 tensor with cu_seqlens_k[0]=0 and
          cu_seqlens_k[B]=total_k.

    Args:
        seqused_k: Optional [B] int32 tensor giving the per-sample valid
            length of k_context. Positions >= seqused_k[b] are masked out
            of the K-side attention. Only valid in dense mode.
        cu_seqlens_k: Optional [B+1] int32 tensor of jagged offsets. When
            provided, k_context/v_context are read in jagged 3-D layout
            and no padding compute is performed.
    """
    B = q.shape[0]
    W = q.shape[2]
    Hq, D = q.shape[3], q.shape[4]

    has_cu_seqlens_k = cu_seqlens_k is not None
    if has_cu_seqlens_k:
        # Jagged mode: k_context is (total_k, Hkv, D)
        Hkv = k_context.shape[1]
        empty = k_context.shape[0] == 0
    else:
        Sk = k_context.shape[1]
        Hkv = k_context.shape[2]
        empty = Sk == 0

    if empty:
        out.fill_(0)
        lse.fill_(float('-inf'))
        return

    cc = _get_compute_capability()
    kernel_cls = _KERNEL_CLS_MAP.get(cc)
    assert kernel_cls is not None, f"Unsupported compute capability: {cc}"

    # [B, 1, W, Hq, D] → [B, W, Hq, D] (zero-copy view, Sq=1)
    q_flat = q.reshape(B, W, Hq, D)

    cutlass_dtype = TORCH_TO_CUTLASS[q.dtype]
    qhead_per_kvhead = Hq // Hkv
    tile_m = 128 if D <= 128 else 64
    tile_n = 128 if D <= 128 else 64

    has_seqused_k = seqused_k is not None
    seqused_k_arg = to_cute_tensor(seqused_k) if has_seqused_k else None
    cu_seqlens_k_arg = to_cute_tensor(cu_seqlens_k) if has_cu_seqlens_k else None

    # cute.compile produces different specialized kernels depending on
    # whether the optional inputs are None vs real tensors (the FA kernel
    # uses const_expr branches and switches its expected mK layout from
    # 4-D dense to 3-D jagged). Include both presence flags in the cache
    # key so the specializations don't alias.
    key = ("k1", cc, D, qhead_per_kvhead, cutlass_dtype,
           num_splits > 1, has_seqused_k, has_cu_seqlens_k)
    if key not in _compile_cache:
        config = KernelConfig(
            head_dim=D, qhead_per_kvhead=qhead_per_kvhead,
            pack_gqa=False, tile_m=tile_m, tile_n=tile_n,
        )
        kernel = kernel_cls(config, dtype=cutlass_dtype, is_split_kv=True)
        _compile_cache[key] = cute.compile(
            kernel,
            to_cute_tensor(q_flat), to_cute_tensor(k_context),
            to_cute_tensor(v_context), to_cute_tensor(out),
            to_cute_tensor(lse, assumed_align=4),
            softmax_scale,
            None, cu_seqlens_k_arg, None, seqused_k_arg, None,  # cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, page_table
            None, None, None, None, 0, 0,  # beam params (mQ_beam, mK_beam, mV_beam, topk, bw, dn)
            num_splits,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _compile_cache[key](
        q_flat.detach(), k_context.detach(), v_context.detach(),
        out.detach(), lse,
        softmax_scale,
        None,
        cu_seqlens_k.detach() if has_cu_seqlens_k else None,
        None,
        seqused_k.detach() if has_seqused_k else None,
        None,
        None, None, None, None, 0, 0,
        num_splits,
    )


# ---------------------------------------------------------------------------
# Kernel 2: Beam Sparse Attention
# ---------------------------------------------------------------------------

def _beam_sparse_attention(q, k_beam, v_beam, topk_indices, decode_nums,
                           softmax_scale, out, lse):
    """K2: Q × Beam KV[topK]. Writes fp32 out (B, W, Hq, D) and lse (B, Hq, W)."""
    B = q.shape[0]
    W = q.shape[2]
    Hq, D = q.shape[3], q.shape[4]
    Hkv = k_beam.shape[2]
    qhead_per_kv = Hq // Hkv

    if decode_nums == 0:
        out.fill_(0)
        lse.fill_(float('-inf'))
        return

    # Flatten Q: [B, 1, W, Hq, D] → [B*W, 1, Hq, D] (zero-copy view)
    q_flat = q.reshape(B * W, 1, Hq, D)
    # K2 output: fp32, viewed as [B*W, 1, Hq, D] into pre-allocated out (B, W, Hq, D)
    out_flat = out.reshape(B * W, 1, Hq, D)

    # topk_indices: [B, 1, Hq, max_dn, W] → per kv_head: [B, Hkv, dn, W]
    # For MHA (qpk=1), ::1 is no-op; for GQA, selects one head per group.
    # With int32 input and Sq=1, reshape is zero-copy when qpk=1.
    topk_kv = topk_indices[:, 0, ::qhead_per_kv, :decode_nums, :]
    cutlass_dtype = TORCH_TO_CUTLASS[q.dtype]
    vec_size = max(128 // cutlass_dtype.width, D // 32)
    bdx = D // vec_size
    bdy = qhead_per_kv
    num_threads = bdx * bdy

    key = ("k2", D, qhead_per_kv, cutlass_dtype)
    if key not in _compile_cache:
        config = KernelConfig(
            head_dim=D, qhead_per_kvhead=qhead_per_kv,
            pack_gqa=False, tile_m=1, tile_n=decode_nums,
        )
        kernel = FlashAttentionForwardDecode(
            config, dtype=cutlass_dtype, num_threads=num_threads, is_sparse=True,
        )
        _compile_cache[key] = cute.compile(
            kernel,
            to_cute_tensor(q_flat), to_cute_tensor(k_beam), to_cute_tensor(v_beam),
            to_cute_tensor(out_flat), to_cute_tensor(lse, assumed_align=4),
            softmax_scale,
            to_cute_tensor(topk_kv), W, decode_nums,
            None, None, None, None, None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _compile_cache[key](
        q_flat.detach(), k_beam.detach(), v_beam.detach(),
        out_flat.detach(), lse,
        softmax_scale,
        topk_kv, W, decode_nums,
        None, None, None, None, None,
    )


# ---------------------------------------------------------------------------
# Kernel 3: Combine
# ---------------------------------------------------------------------------

def _combine(o_partial, lse_partial, o_out, lse_out):
    """K3: Log-sum-exp merge. Reads pre-allocated partials, writes output."""
    D = o_partial.shape[-1]
    num_splits = o_partial.shape[0]

    # Programmatic Dependent Launch (PDL): SM90/SM100 support, SM80/SM120 do not
    cc = _get_compute_capability()
    use_pdl = cc in (9, 10, 11)

    # Match FA's parameter selection for combine kernel
    # (flash_fwd_combine_launch_template.h:58-77)
    k_block_size = 64 if D <= 64 else 128
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(max(num_splits, 2))), 4)
    if tile_m == 8:
        log_max_splits = max(log_max_splits, 5)

    key = ("k3", D, k_block_size, tile_m, log_max_splits, use_pdl)
    if key not in _compile_cache:
        from quack.compile_utils import make_fake_tensor
        from src.flash_fwd_combine import FlashAttentionForwardCombine

        kernel = FlashAttentionForwardCombine(
            dtype=cutlass.BFloat16,
            dtype_partial=cutlass.Float32,
            head_dim=D,
            tile_m=tile_m,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
            use_pdl=use_pdl,
        )
        num_splits, batch, sq, nheads = (
            cute.sym_int64() for _ in range(4)
        )
        div = 128 // cutlass.Float32.width
        mO_partial = make_fake_tensor(
            cutlass.Float32, (num_splits, batch, sq, nheads, D), divisibility=div,
        )
        mLSE_partial = make_fake_tensor(
            cutlass.Float32, (num_splits, batch, sq, nheads), divisibility=1, leading_dim=2,
        )
        mO = make_fake_tensor(cutlass.BFloat16, (batch, sq, nheads, D), divisibility=div)
        mLSE = make_fake_tensor(
            cutlass.Float32, (batch, sq, nheads), divisibility=1, leading_dim=1,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        _compile_cache[key] = cute.compile(
            kernel,
            mO_partial, mLSE_partial, mO, mLSE,
            None, None, None, None, None,
            stream,
            options="--enable-tvm-ffi",
        )

    _compile_cache[key](
        o_partial, lse_partial, o_out, lse_out,
        None, None, None, None, None,
    )


# ---------------------------------------------------------------------------
# Fused context + beam kernel
# ---------------------------------------------------------------------------

def _fused_context_beam(q, k_context, v_context, k_beam, v_beam, topk_indices,
                        decode_nums, softmax_scale, num_splits, is_split_kv, out, lse):
    """Run fused context + beam kernel (auto SM80/SM90 dispatch).

    Args:
        out: (B, W, Hq, D) bf16 when is_split_kv=False
             (num_splits, B, W, Hq, D) fp32 when is_split_kv=True
        lse: (B, Hq, W) fp32 when is_split_kv=False
             (num_splits, B, Hq, W) fp32 when is_split_kv=True
    """
    B = q.shape[0]
    W = q.shape[2]
    Hq, D = q.shape[3], q.shape[4]
    Hkv = k_context.shape[2]
    qhead_per_kv = Hq // Hkv

    q_flat = q.reshape(B, W, Hq, D)
    topk_kv = topk_indices[:, 0, ::qhead_per_kv, :decode_nums, :]

    cc = _get_compute_capability()
    cutlass_dtype = TORCH_TO_CUTLASS[q.dtype]
    tile_m = 128 if D <= 128 else 64
    tile_n = 128 if D <= 128 else 64

    key = ("fused", cc, D, qhead_per_kv, cutlass_dtype, is_split_kv)
    if key not in _compile_cache:
        config = KernelConfig(
            head_dim=D, qhead_per_kvhead=qhead_per_kv,
            pack_gqa=False, tile_m=tile_m, tile_n=tile_n,
        )
        kernel_cls = _KERNEL_CLS_MAP.get(cc)
        assert kernel_cls is not None, f"Unsupported compute capability: {cc}"
        if cc == 9:
            kernel = kernel_cls(config, dtype=cutlass_dtype,
                                is_split_kv=is_split_kv, has_beam_sparse=True)
        elif cc in (10, 11):
            kernel = kernel_cls(config, dtype=cutlass_dtype,
                                is_split_kv=is_split_kv, has_beam_sparse=True)
        else:
            kernel = kernel_cls(config, dtype=cutlass_dtype,
                                num_stages=1, num_threads=256,
                                is_split_kv=is_split_kv, has_beam_sparse=True)
        _compile_cache[key] = cute.compile(
            kernel,
            to_cute_tensor(q_flat), to_cute_tensor(k_context),
            to_cute_tensor(v_context), to_cute_tensor(out),
            to_cute_tensor(lse, assumed_align=4),
            softmax_scale,
            None, None, None, None, None,  # cu_seqlens, seqused, page_table
            to_cute_tensor(q_flat),        # mQ_beam
            to_cute_tensor(k_beam), to_cute_tensor(v_beam),
            to_cute_tensor(topk_kv), W, decode_nums,
            num_splits,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )

    _compile_cache[key](
        q_flat.detach(), k_context.detach(), v_context.detach(),
        out.detach(), lse,
        softmax_scale,
        None, None, None, None, None,
        q_flat.detach(), k_beam.detach(), v_beam.detach(), topk_kv, W, decode_nums,
        num_splits,
    )


# ---------------------------------------------------------------------------
# BeamDecodeAttn (torch.autograd.Function)
# ---------------------------------------------------------------------------

class BeamDecodeAttn(torch.autograd.Function):
    """Beam search decode attention with dynamic num_splits.

    Uses fused context+beam kernel (default) or 3-kernel pipeline (fallback).
    num_splits determined by Hopper heuristic for SM occupancy.

    Backward: not supported (forward-only).
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k_context: torch.Tensor,
        v_context: torch.Tensor,
        k_beam: torch.Tensor,
        v_beam: torch.Tensor,
        topk_indices: torch.Tensor,
        decode_nums: int,
        softmax_scale: float,
        backend: str,
        seqused_k: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ):
        B, Sq, W, Hq, D = q.shape
        assert Sq == 1, "Decode mode: seqlen_q must be 1"
        if cu_seqlens_k is not None:
            # Jagged k_context: [total_k, Hkv, D]
            Hkv = k_context.shape[1]
            Sk = -1  # not used in this branch; per-sample length comes from cu_seqlens_k
        else:
            Hkv = k_context.shape[2]
            Sk = k_context.shape[1]

        cc = _get_compute_capability()
        # SM100: use 3-kernel with split-KV on K1 (fused beam slower on SM100)
        # SM8x/SM90/SM120: use fused (default "dsl" path below)
        if backend == "3kernel" or cc in (10, 11):
            # Compute num_splits for K1 (context attention)
            tile_m = 128 if D <= 128 else 64
            tile_n = 128 if D <= 128 else 64
            num_m_blocks = math.ceil(W / tile_m)
            if cu_seqlens_k is not None:
                # Jagged mode: split-KV's partial-output geometry assumes a
                # uniform Sk per sample (it slices [:ns] of the [Sk] dim).
                # For jagged input this isn't well-defined, so force ns=1.
                ns = 1
            else:
                num_n_blocks = math.ceil(Sk / tile_n) if Sk > 0 else 0
                total_mblocks = B * Hq * num_m_blocks  # Hq not Hkv: grid dispatches per Q head
                size_one_kv_head = Sk * D * 2 * 2
                num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count
                ns = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks,
                                          num_m_blocks, size_one_kv_head)
                if ns > 1 and num_n_blocks > 0:
                    ns = min(ns, num_n_blocks)
                # FIXME: split-KV (num_splits>1) + seqused_k currently hangs
                # the K1 launch on SM90 (verified on H100 PCIe). Workaround:
                # force ns=1 when seqused_k is set. See feat(K1): support
                # seqused_k commit message for the fuller writeup.
                if seqused_k is not None and ns > 1:
                    ns = 1

            total_splits = ns + 1  # ns for K1 splits, 1 for K2
            o_partial = torch.empty(total_splits, B, W, Hq, D, device=q.device, dtype=torch.float32)
            lse_partial_raw = torch.empty(total_splits, B, Hq, W, device=q.device, dtype=torch.float32)
            o_out = torch.empty(B, W, Hq, D, device=q.device, dtype=torch.bfloat16)
            lse_out = torch.empty(B, Hq, W, device=q.device, dtype=torch.float32).transpose(-1, -2)

            # K1: context attention with split-KV
            if ns > 1:
                _context_attention(q, k_context, v_context, softmax_scale,
                                   out=o_partial[:ns], lse=lse_partial_raw[:ns],
                                   num_splits=ns, seqused_k=seqused_k,
                                   cu_seqlens_k=cu_seqlens_k)
            else:
                _context_attention(q, k_context, v_context, softmax_scale,
                                   out=o_partial[0], lse=lse_partial_raw[0],
                                   seqused_k=seqused_k,
                                   cu_seqlens_k=cu_seqlens_k)
            # K2: beam sparse attention
            _beam_sparse_attention(q, k_beam, v_beam, topk_indices, decode_nums,
                                   softmax_scale,
                                   out=o_partial[ns], lse=lse_partial_raw[ns])
            # K3: combine all partials
            lse_partial = lse_partial_raw.transpose(-1, -2)
            _combine(o_partial, lse_partial, o_out, lse_out)
            return o_out.unsqueeze(1), lse_out.unsqueeze(1)

        # Fused path: dynamic num_splits via Hopper heuristic
        tile_m = 128 if D <= 128 else 64
        tile_n = 128 if D <= 128 else 64
        num_m_blocks = math.ceil(W / tile_m)
        num_n_blocks = math.ceil(Sk / tile_n) if Sk > 0 else 0
        total_mblocks = B * Hq * num_m_blocks  # Hq not Hkv: grid dispatches per Q head
        size_one_kv_head = Sk * D * 2 * 2  # K+V bf16 bytes
        num_SMs = torch.cuda.get_device_properties(q.device).multi_processor_count
        ns = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks,
                                  num_m_blocks, size_one_kv_head)
        # Floor-based splitting: cap num_splits to num_n_blocks so every
        # split gets at least floor(n/ns) >= 1 block.
        if ns > 1 and num_n_blocks > 0:
            ns = min(ns, num_n_blocks)
        is_split_kv = ns > 1

        if is_split_kv:
            o_partial = torch.empty(ns, B, W, Hq, D, device=q.device, dtype=torch.float32)
            lse_partial_raw = torch.empty(ns, B, Hq, W, device=q.device, dtype=torch.float32)
            o_out = torch.empty(B, W, Hq, D, device=q.device, dtype=torch.bfloat16)
            lse_out = torch.empty(B, Hq, W, device=q.device, dtype=torch.float32).transpose(-1, -2)

            _fused_context_beam(q, k_context, v_context, k_beam, v_beam, topk_indices,
                                decode_nums, softmax_scale, ns, True,
                                out=o_partial, lse=lse_partial_raw)

            lse_partial = lse_partial_raw.transpose(-1, -2)
            _combine(o_partial, lse_partial, o_out, lse_out)
            return o_out.unsqueeze(1), lse_out.unsqueeze(1)
        else:
            # Fused, no split → bf16 direct
            o_out = torch.empty(B, W, Hq, D, device=q.device, dtype=torch.bfloat16)
            lse_out = torch.empty(B, Hq, W, device=q.device, dtype=torch.float32)

            _fused_context_beam(q, k_context, v_context, k_beam, v_beam, topk_indices,
                                decode_nums, softmax_scale, 1, False,
                                out=o_out, lse=lse_out)
            return o_out.unsqueeze(1), lse_out.transpose(-1, -2).unsqueeze(1)

    @staticmethod
    def backward(ctx, *args):
        raise NotImplementedError("Beam decode attention is forward-only")


# ---------------------------------------------------------------------------
# Functional interface
# ---------------------------------------------------------------------------

def beam_decode_attn(
    q: torch.Tensor,
    k_context: torch.Tensor,
    v_context: torch.Tensor,
    k_beam: torch.Tensor,
    v_beam: torch.Tensor,
    topk_indices: torch.Tensor,
    decode_nums: int,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    backend: str = "dsl",
    seqused_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Functional interface for beam decode attention.

    Args:
        q:            [batch, seqlen_q, beam_width, head_q, dim]
        k_context:    Dense  (default):       [batch, seqlen_context, head_kv, dim]
                      Jagged (cu_seqlens_k):  [total_k, head_kv, dim]
        v_context:    same shape as k_context
        k_beam:       [batch, decode_nums * beam_width, head_kv, dim]
        v_beam:       same shape as k_beam
        topk_indices: [batch, seqlen_q, head_q, max_decode_nums, beam_width]
        decode_nums:  number of valid decode steps
        softmax_scale: default 1/sqrt(dim)
        return_lse:   return combined LSE
        out:          optional pre-allocated output
        backend:      "dsl" (fused kernel, default), "3kernel" (K1+K2+K3 pipeline)
        seqused_k:    Optional [batch] int32 tensor giving the per-sample
            valid length of k_context (dense mode). Positions
            >= seqused_k[b] are masked out of the K1 launch. Only
            supported with backend="3kernel".
        cu_seqlens_k: Optional [batch+1] int32 tensor of jagged offsets
            for k_context / v_context, with cu_seqlens_k[0] = 0 and
            cu_seqlens_k[batch] = total_k. When provided, k_context and
            v_context must be 3-D [total_k, head_kv, dim]; no padding
            compute is performed in the K1 launch. Only supported with
            backend="3kernel" and cannot be combined with seqused_k.

    Returns:
        out: [batch, seqlen_q, beam_width, head_q, dim]  same dtype as q
        lse: [batch, seqlen_q, beam_width, head_q]  fp32, or None
    """
    _validate_inputs(q, k_context, v_context, k_beam, v_beam, topk_indices, decode_nums,
                     jagged_k_context=cu_seqlens_k is not None)

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if seqused_k is not None:
        assert seqused_k.dim() == 1 and seqused_k.shape[0] == q.shape[0], (
            f"seqused_k must be [B={q.shape[0]}], got {tuple(seqused_k.shape)}"
        )
        assert seqused_k.dtype == torch.int32, "seqused_k must be int32"
        assert backend == "3kernel", (
            "seqused_k is currently only supported with backend='3kernel' "
            "(the fused/dsl path does not thread seqused_k through to the "
            "kernel and would silently produce wrong results on padded "
            "batches)."
        )

    if cu_seqlens_k is not None:
        assert cu_seqlens_k.dim() == 1 and cu_seqlens_k.shape[0] == q.shape[0] + 1, (
            f"cu_seqlens_k must be [B+1={q.shape[0] + 1}], got {tuple(cu_seqlens_k.shape)}"
        )
        assert cu_seqlens_k.dtype == torch.int32, "cu_seqlens_k must be int32"
        assert backend == "3kernel", (
            "cu_seqlens_k is currently only supported with backend='3kernel' "
            "(the fused/dsl path does not thread it through and would "
            "silently produce wrong results)."
        )
        assert seqused_k is None, (
            "cu_seqlens_k and seqused_k are mutually exclusive — jagged "
            "mode encodes per-sample length via offsets, dense+seqused_k "
            "encodes it via valid-length tensor."
        )

    out_raw, lse = BeamDecodeAttn.apply(
        q, k_context, v_context, k_beam, v_beam,
        topk_indices, decode_nums, softmax_scale, backend, seqused_k, cu_seqlens_k,
    )

    if out is not None:
        assert out.shape == q.shape and out.dtype == q.dtype
        out.copy_(out_raw)
    else:
        out = out_raw

    return out, lse if return_lse else None
