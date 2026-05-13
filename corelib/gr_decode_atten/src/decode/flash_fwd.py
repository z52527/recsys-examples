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

# Copyright (c) 2025, cjerry.
# Decode attention forward kernel using CUDA core scalar FMA (MmaUniversalOp).
# Supports tile_m=1 (single query token decode), compatible with SM80/SM90/SM100.
# Reference: FlashInfer decode attention + CUTLASS sgemm.py MmaUniversalOp pattern.
#
# Modes:
#   Dense:  sequential K/V scan (standard decode attention)
#   Sparse: topK gather from beam KV cache (beam search decode)

import operator
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync

from ..common import utils
from ..common.kernel_config import KernelConfig


class FlashAttentionForwardDecode:
    """tile_m=1 decode attention using scalar FMA (CUDA core).

    Uses MmaUniversalOp (1x1x1 scalar FMA atom) via CuTe DSL.
    Compatible with SM80/SM90/SM100 — no tensor core instructions.

    Thread block layout (following FlashInfer pattern):
        tx (bdx) = head_dim / vec_size   — one warp covers full head_dim
        ty (bdy) = qhead_per_kvhead      — GQA group heads
        tz (bdz) = num_threads / (bdx * bdy) — K/V tile parallelism

    Sparse mode (beam search):
        Grid: (batch * beam_width, kv_heads, 1)
        K/V loaded via topK gather instead of sequential scan.
        Same GQA group shares topk_indices → KV loaded once per group.
    """

    def __init__(
        self,
        config: KernelConfig,
        dtype,
        num_threads: int = 128,
        is_split_kv: bool = False,
        is_sparse: bool = False,
    ):
        self.dtype = dtype
        self.head_dim = config.head_dim_padded
        self.head_dim_v = config.head_dim_v_padded
        self.qhead_per_kvhead = config.qhead_per_kvhead
        self.tile_n = config.tile_n
        self.num_threads = num_threads
        self.is_split_kv = is_split_kv
        self.is_sparse = is_sparse

        # Thread block dimensions (FlashInfer pattern)
        self.vec_size = max(128 // dtype.width, self.head_dim // 32)
        self.bdx = self.head_dim // self.vec_size  # <= 32
        self.bdy = self.qhead_per_kvhead
        self.bdz = num_threads // (self.bdx * self.bdy)
        assert self.bdx <= 32, f"bdx={self.bdx} must be <= 32 (one warp)"
        assert self.bdx * self.bdy * self.bdz == num_threads

        # K/V positions processed per iteration per warp-z
        self.tile_size_per_bdx = 8 if self.bdy == 1 else 1
        self.tile_size = self.bdy * self.tile_size_per_bdx  # per tz
        # Total K/V positions per iteration (all tz warps)
        self.kv_per_iter = self.tile_size * self.bdz
        self.num_stages = 1 if is_sparse else 2  # sparse: single stage (no pipeline)

        # Thread cooperation for gather loads (used in sparse mode)
        async_copy_elems = 128 // dtype.width
        self.copy_threads_per_row = self.head_dim // async_copy_elems
        self.copy_rows_per_iter = num_threads // self.copy_threads_per_row

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # dense: (b, 1, h, d); sparse: (b*W, 1, h, d)
        mK: cute.Tensor,  # (b, s_k, h_k, d)
        mV: cute.Tensor,  # (b, s_k, h_k, d_v)
        mO: cute.Tensor,  # dense: (b, 1, h, d_v); sparse: (b*W, 1, h, d_v)
        mLSE: Optional[cute.Tensor],  # dense: (b, h); sparse: (b*W, h)
        softmax_scale: Float32,
        mTopkIdxs: Optional[cute.Tensor] = None,  # sparse: (b, h_k, dn, W)
        beam_width: Int32 = Int32(0),
        decode_nums: Int32 = Int32(0),
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        KernelConfig.check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    mSeqUsedQ,
                    mSeqUsedK,
                )
            )
        )
        self.o_dtype = mO.element_type
        softmax_scale_log2 = softmax_scale * Float32(1.44269504088896340736)

        # Smem layouts for K/V: [kv_tile_rows, head_dim, num_stages] row-major
        kv_tile_rows = self.tile_size * self.bdz
        sK_layout = cute.make_layout(
            (kv_tile_rows, self.head_dim, self.num_stages),
            stride=(self.head_dim, 1, kv_tile_rows * self.head_dim),
        )
        sV_layout = cute.make_layout(
            (kv_tile_rows, self.head_dim_v, self.num_stages),
            stride=(self.head_dim_v, 1, kv_tile_rows * self.head_dim_v),
        )

        # Tiled copy for K/V: cp.async 128-bit gmem → smem (SM80 pattern)
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        copy_threads_per_row = self.head_dim // async_copy_elems
        copy_rows_per_iter = self.num_threads // copy_threads_per_row
        tKV_layout = cute.make_ordered_layout(
            (copy_rows_per_iter, copy_threads_per_row), order=(1, 0)
        )
        vKV_layout = cute.make_layout((1, async_copy_elems))
        tiled_copy_KV = cute.make_tiled_copy_tv(atom_async_copy, tKV_layout, vKV_layout)

        # Assume aligned for 128-bit cp.async
        from ..common.cute_dsl_utils import assume_tensor_aligned

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]

        # Layout transposes: (b, s, h, d) → (s, d, h, b)
        QO_transpose = [1, 3, 2, 0]
        KV_transpose = [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=QO_transpose))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=QO_transpose))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=KV_transpose))
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=KV_transpose))

        # Grid batch dimension: Q's batch (may include beam_width for sparse)
        grid_batch = mQ.shape[3]
        num_kv_heads = mK.shape[2]
        # seqlen_k: dense uses K's seqlen, sparse uses decode_nums
        seqlen_k = decode_nums if const_expr(self.is_sparse) else mK.shape[0]

        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale_log2,
            sK_layout,
            sV_layout,
            tiled_copy_KV,
            seqlen_k,
            mTopkIdxs,
            beam_width,
        ).launch(
            grid=(grid_batch, num_kv_heads, 1),
            block=(self.bdx, self.bdy, self.bdz),
            stream=stream,
        )

    @cute.jit
    def _gather_load_tile(
        self,
        gKV: cute.Tensor,  # (s_k, d) global KV for this (batch, kv_head)
        sKV: cute.Tensor,  # (kv_tile_rows, d, stages) shared memory
        gTopk: cute.Tensor,  # (decode_nums,) topk indices for this (batch, kv_head, beam)
        kv_base: Int32,  # starting row within decode_nums
        decode_nums: Int32,
        tidx: Int32,
        stage_idx: Int32,
    ):
        """Gather load one tile of KV rows from scattered positions into SMEM.

        Thread cooperation: copy_threads_per_row threads load one row in
        parallel, copy_rows_per_iter rows are loaded simultaneously.
        """
        thread_row = tidx // self.copy_threads_per_row
        thread_col = tidx % self.copy_threads_per_row
        async_copy_elems = 128 // self.dtype.width
        col_offset = thread_col * async_copy_elems

        num_load_batches = cute.ceil_div(
            self.tile_size * self.bdz, self.copy_rows_per_iter
        )
        for load_batch in cutlass.range(num_load_batches):
            row = load_batch * self.copy_rows_per_iter + thread_row
            global_row = kv_base + row
            if global_row < decode_nums:
                kv_idx = gTopk[global_row]
                gKV_row = gKV[kv_idx, None]  # (d,)
                for e in cutlass.range_constexpr(async_copy_elems):
                    sKV[row, col_offset + e, stage_idx] = gKV_row[col_offset + e]
            else:
                for e in cutlass.range_constexpr(async_copy_elems):
                    sKV[row, col_offset + e, stage_idx] = self.dtype(0)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b) after transpose
        mK: cute.Tensor,  # (s_k, d, h_k, b) after transpose
        mV: cute.Tensor,  # (s_k, d_v, h_k, b) after transpose
        mO: cute.Tensor,  # (s_q, d_v, h, b) after transpose
        mLSE: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        sK_layout: cute.Layout,
        sV_layout: cute.Layout,
        tiled_copy_KV: cute.TiledCopy,
        seqlen_k: Int32,
        mTopkIdxs: Optional[cute.Tensor],  # sparse: (b, h_k, dn, W)
        beam_width: Int32,
    ):
        tx, ty, tz = cute.arch.thread_idx()
        flat_batch_idx = cute.arch.block_idx()[0]
        kv_head_idx = cute.arch.block_idx()[1]
        qo_head_idx = kv_head_idx * self.bdy + ty
        tidx = (tz * self.bdy + ty) * self.bdx + tx

        # Sparse: split flat batch into original batch + beam index
        if const_expr(self.is_sparse):
            batch_idx = flat_batch_idx // beam_width
            beam_idx = flat_batch_idx % beam_width
        else:
            batch_idx = flat_batch_idx

        # =====================================================================
        # Shared memory allocation
        # =====================================================================
        smem = cutlass.utils.SmemAllocator()
        sK = smem.allocate_tensor(self.dtype, sK_layout, 128)
        sV = smem.allocate_tensor(self.dtype, sV_layout, 128)
        smem_o = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.bdz * self.bdy, self.head_dim_v)),
            4,
        )
        smem_md = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.bdz * self.bdy, 2)),
            4,
        )

        # =====================================================================
        # Step 1: Load Q into registers (vectorized gmem → register)
        # Q: 1 row × head_dim. Each tx loads vec_size consecutive elements.
        # FlashInfer pattern: Q stays in registers, no smem.
        # For sparse: mQ batch dim is (b*W), use flat_batch_idx.
        # =====================================================================
        q_reg = cute.make_rmem_tensor((self.vec_size,), Float32)
        gQ = mQ[0, None, qo_head_idx, flat_batch_idx]  # (d,)
        for i in cutlass.range_constexpr(self.vec_size):
            q_reg[i] = Float32(gQ[tx * self.vec_size + i])

        # =====================================================================
        # Step 2: Setup K/V access
        # Dense: sequential tiled copy with cp.async pipeline
        # Sparse: gather load via topk_indices
        # =====================================================================
        gK = mK[None, None, kv_head_idx, batch_idx]  # (s_k, d)
        gV = mV[None, None, kv_head_idx, batch_idx]  # (s_k, d_v)

        # Sparse: pre-slice topk indices for this (batch, kv_head, beam)
        if const_expr(self.is_sparse):
            # mTopkIdxs: (b, h_k, dn, W) → slice to (dn,)
            gTopk = mTopkIdxs[batch_idx, kv_head_idx, None, beam_idx]  # (decode_nums,)

        kv_tile_rows = self.tile_size * self.bdz
        num_iters = cute.ceil_div(seqlen_k, self.kv_per_iter)

        if const_expr(not self.is_sparse):
            # --- Dense path: sequential tiled copy setup ---
            gK_tiled = cute.local_tile(gK, (kv_tile_rows, self.head_dim), (None, 0))
            gV_tiled = cute.local_tile(gV, (kv_tile_rows, self.head_dim_v), (None, 0))

            thr_copy = tiled_copy_KV.get_slice(tidx)
            tKgK = thr_copy.partition_S(gK_tiled)
            tKsK = thr_copy.partition_D(sK)
            tVgV = thr_copy.partition_S(gV_tiled)
            tVsV = thr_copy.partition_D(sV)

            cKV = cute.make_identity_tensor((kv_tile_rows, self.head_dim))
            tKcK = thr_copy.partition_S(cKV)
            t0KcK = tiled_copy_KV.get_slice(0).partition_S(cKV)

        # =====================================================================
        # Step 3: Online softmax state
        # =====================================================================
        # Use finite sentinel instead of -inf to avoid NaN from -inf - (-inf)
        # in exp2() when all scores in a tz group are -inf (sparse padding).
        # FlashInfer uses -5e4; any value << min real score works.
        st_m = Float32(-5e4)
        st_d = Float32(0.0)
        o_reg = cute.make_rmem_tensor((self.vec_size,), Float32)
        o_reg.fill(Float32(0.0))
        s_reg = cute.make_rmem_tensor((self.tile_size,), Float32)

        # =====================================================================
        # Step 4: Prologue — prefetch first num_stages tiles of K and V
        # Dense: cp.async prefetch num_stages tiles
        # Sparse: no prologue (single stage, load in mainloop)
        # =====================================================================
        if const_expr(not self.is_sparse):
            for stage in cutlass.range_constexpr(self.num_stages):
                seqlen_limit = seqlen_k - stage * self.kv_per_iter - tKcK[0][0]
                for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                    if t0KcK[0, n, 0][0] < seqlen_limit:
                        cute.copy(
                            tiled_copy_KV,
                            tKgK[None, n, None, stage],
                            tKsK[None, n, None, stage],
                        )
                cute.arch.cp_async_commit_group()
                for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                    if t0KcK[0, n, 0][0] < seqlen_limit:
                        cute.copy(
                            tiled_copy_KV,
                            tVgV[None, n, None, stage],
                            tVsV[None, n, None, stage],
                        )
                cute.arch.cp_async_commit_group()

        # =====================================================================
        # Step 5: Mainloop
        # Dense: 2-stage pipeline (wait → compute → prefetch next)
        # Sparse: single-stage (gather load → wait → compute)
        # =====================================================================
        stage_idx = Int32(0)

        for iter_idx in cutlass.range(num_iters):
            kv_base = iter_idx * self.kv_per_iter
            tz_offset = tz * self.tile_size

            # === Sparse: gather load K tile into sK (synchronous) ===
            if const_expr(self.is_sparse):
                self._gather_load_tile(
                    gK,
                    sK,
                    gTopk,
                    kv_base,
                    seqlen_k,
                    tidx,
                    stage_idx,
                )

            # === PHASE 1: wait K → compute QK ===
            if const_expr(not self.is_sparse):
                cute.arch.cp_async_wait_group(2 * self.num_stages - 1)
            cute.arch.sync_threads()

            m_prev = st_m
            for j in cutlass.range_constexpr(self.tile_size):
                row_idx = tz_offset + j
                score = Float32(0.0)
                for i in cutlass.range_constexpr(self.vec_size):
                    k_val = Float32(sK[row_idx, tx * self.vec_size + i, stage_idx])
                    score = score + q_reg[i] * k_val
                score = utils.warp_reduce(score, operator.add, width=self.bdx)
                score = score * softmax_scale_log2
                kv_pos = kv_base + tz_offset + j
                if kv_pos >= seqlen_k:
                    score = -Float32.inf
                if score > st_m:
                    st_m = score
                s_reg[j] = score

            # Online softmax update
            o_scale = cute.math.exp2(m_prev - st_m, fastmath=True)
            st_d = st_d * o_scale
            for j in cutlass.range_constexpr(self.tile_size):
                s_reg[j] = cute.math.exp2(s_reg[j] - st_m, fastmath=True)
                st_d = st_d + s_reg[j]
            for i in cutlass.range_constexpr(self.vec_size):
                o_reg[i] = o_reg[i] * o_scale
            cute.arch.sync_threads()

            # === Dense: prefetch next K ===
            if const_expr(not self.is_sparse):
                next_block = iter_idx + self.num_stages
                if next_block < num_iters:
                    seqlen_limit = seqlen_k - next_block * self.kv_per_iter - tKcK[0][0]
                    for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                        if t0KcK[0, n, 0][0] < seqlen_limit:
                            cute.copy(
                                tiled_copy_KV,
                                tKgK[None, n, None, next_block],
                                tKsK[None, n, None, stage_idx],
                            )
                cute.arch.cp_async_commit_group()

            # === Sparse: gather load V tile into sV (synchronous) ===
            if const_expr(self.is_sparse):
                self._gather_load_tile(
                    gV,
                    sV,
                    gTopk,
                    kv_base,
                    seqlen_k,
                    tidx,
                    stage_idx,
                )

            # === PHASE 3: wait V → compute PV ===
            if const_expr(not self.is_sparse):
                cute.arch.cp_async_wait_group(2 * self.num_stages - 1)
            cute.arch.sync_threads()

            for j in cutlass.range_constexpr(self.tile_size):
                row_idx = tz_offset + j
                for i in cutlass.range_constexpr(self.vec_size):
                    v_val = Float32(sV[row_idx, tx * self.vec_size + i, stage_idx])
                    o_reg[i] = o_reg[i] + s_reg[j] * v_val
            cute.arch.sync_threads()

            # === Dense: prefetch next V ===
            if const_expr(not self.is_sparse):
                if next_block < num_iters:
                    seqlen_limit_v = (
                        seqlen_k - next_block * self.kv_per_iter - tKcK[0][0]
                    )
                    for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                        if t0KcK[0, n, 0][0] < seqlen_limit_v:
                            cute.copy(
                                tiled_copy_KV,
                                tVgV[None, n, None, next_block],
                                tVsV[None, n, None, stage_idx],
                            )
                cute.arch.cp_async_commit_group()

            # === ADVANCE stage (dense only, sparse always uses stage 0) ===
            if const_expr(not self.is_sparse):
                stage_idx = stage_idx + 1
                if stage_idx == self.num_stages:
                    stage_idx = Int32(0)

        if const_expr(not self.is_sparse):
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

        # =====================================================================
        # Step 6: Multi-warp state merge (if bdz > 1)
        # =====================================================================
        if const_expr(self.bdz > 1):
            warp_idx = tz * self.bdy + ty
            for i in cutlass.range_constexpr(self.vec_size):
                smem_o[warp_idx, tx * self.vec_size + i] = o_reg[i]
            smem_md[warp_idx, 0] = st_m
            smem_md[warp_idx, 1] = st_d
            cute.arch.sync_threads()

            if tz == 0:
                st_m = Float32(-5e4)
                st_d = Float32(0.0)
                o_reg.fill(Float32(0.0))
                for z in cutlass.range_constexpr(self.bdz):
                    other_idx = z * self.bdy + ty
                    other_m = smem_md[other_idx, 0]
                    other_d = smem_md[other_idx, 1]
                    m_prev = st_m
                    if other_m > m_prev:
                        st_m = other_m
                    else:
                        st_m = m_prev
                    scale_prev = cute.math.exp2(m_prev - st_m, fastmath=True)
                    scale_other = cute.math.exp2(other_m - st_m, fastmath=True)
                    st_d = st_d * scale_prev + other_d * scale_other
                    for i in cutlass.range_constexpr(self.vec_size):
                        other_o = smem_o[other_idx, tx * self.vec_size + i]
                        o_reg[i] = o_reg[i] * scale_prev + other_o * scale_other

        # =====================================================================
        # Step 7: Epilogue — normalize and store O, LSE
        # =====================================================================
        is_writer = const_expr(self.bdz == 1) or tz == 0

        if is_writer:
            # Guard zero/NaN row_sum (same as FA softmax.finalize)
            st_d_is_zero_or_nan = (st_d == Float32(0.0)) or (st_d != st_d)
            inv_d = cute.arch.rcp_approx(
                st_d if not st_d_is_zero_or_nan else Float32(1.0)
            )
            for i in cutlass.range_constexpr(self.vec_size):
                o_reg[i] = o_reg[i] * inv_d

            gO = mO[0, None, qo_head_idx, flat_batch_idx]
            for i in cutlass.range_constexpr(self.vec_size):
                gO[tx * self.vec_size + i] = self.o_dtype(o_reg[i])

            if const_expr(mLSE is not None):
                if tx == 0:
                    # LSE: log2 → natural log, same as FA softmax.finalize():
                    #   (row_max * scale_log2 + log2(row_sum)) * LN2
                    # Here st_m already contains row_max * scale_log2.
                    import math as _math

                    LN2 = _math.log(2.0)
                    lse_val = (
                        (st_m + cute.math.log2(st_d, fastmath=True)) * LN2
                        if not st_d_is_zero_or_nan
                        else -Float32.inf
                    )
                    # Sparse: mLSE is (B, Hq, W), index as [batch, head, beam]
                    # Dense:  mLSE is (B*W, Hq), index as [flat_batch, head]
                    if const_expr(self.is_sparse):
                        mLSE[batch_idx, qo_head_idx, beam_idx] = lse_val
                    else:
                        mLSE[flat_batch_idx, qo_head_idx] = lse_val
