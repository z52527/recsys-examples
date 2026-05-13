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

# Copyright (c) 2025, Tri Dao.

from typing import Optional, Callable, TypeAlias
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr

from quack import layout_utils
from . import utils as utils
from .seqlen_info import SeqlenInfoQK

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions < limit (exclusive upper bound).

    Positions 0..limit-1 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    """32-bit R2P bitmask keeping positions >= limit (inclusive lower bound).

    Positions limit..31 in chunk `s` get bit=1 (keep), the rest bit=0 (mask).
    Uses inline PTX to avoid shift-by-type-width UB.
    """
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    """Apply R2P masking with a custom bitmask generator.

    mask_gen_fn(chunk_idx: constexpr int) -> Uint32:
        Returns a 32-bit bitmask for the chunk. Bit i set means column
        chunk_idx * chunk_size + i is KEPT; bit i clear means masked to -inf.
    """
    ncol = const_expr(cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape))
    # 32-column chunks. The mask_gen_fn returns a Uint32 bitmask (1=keep).
    CHUNK_SIZE = MASK_R2P_CHUNK_SIZE
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        # This needs to be range_constexpr, o/w the compiler can't generate the R2P instruction
        for i in cutlass.range_constexpr(min(CHUNK_SIZE, ncol - s * CHUNK_SIZE)):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def sm90_col_to_r2p_idx(col_limit: Int32) -> Int32:
    """Transform SM90 MMA column coordinate to R2P element index.

    SM90 MMA accumulator column indices are non-contiguous: 0, 1, 8, 9, 16, 17, ...
    Element indices are contiguous: 0, 1, 2, 3, 4, 5, ...
    This converts a column-space threshold to element-space for r2p_bitmask_below/above.
    """
    return col_limit // 8 * 2 + min(col_limit % 8, 2)


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
    """Convert a row coordinate to an R2P element index in the warp-group interleaved layout.

    In the SM100 backward pass, 2 warp groups share TMEM. The TMEM load atom
    distributes rows in an interleaved pattern: elements 0..num_rep-1 map to
    rows 0..num_rep-1 (warp group 0), elements num_rep..2*num_rep-1 map to
    rows num_rep*num_wg..num_rep*num_wg+num_rep-1 (warp group 1), and so on.
    Row-coordinate thresholds (causal limits, window bounds, uih_len) must be
    converted to element indices before use with r2p_bitmask_above/below.

    Rows not owned by this thread (in the gap between warp groups) are clamped
    to the boundary element index, which is safe because R2P thresholds are
    monotonic.

    Example with num_rep=16, num_wg=2:
        row  0 -> elem  0,  row 15 -> elem 15,
        row 16 -> elem 16 (clamped), row 31 -> elem 16 (clamped),
        row 32 -> elem 16, row 33 -> elem 17, row 47 -> elem 31.
    """
    return x // (num_rep * num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1  # only pass in if we're doing PackGQA
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask(
        self,
        acc_S: cute.Tensor,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        thr_mma: cute.TiledMma,
        mask_seqlen: cutlass.Constexpr[bool],
    ) -> None:
        """Apply seqlen-only masking (no causal, no local, no mask_mod)."""
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
        t0ScS_mn = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB
        )
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_mn[0][COL]
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(mask_seqlen):
            r2p = const_expr(not self.swap_AB)
            if const_expr(not r2p):
                for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
                    oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
                    for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                        acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
            else:
                seqlenk_col_limit_r2p = sm90_col_to_r2p_idx(seqlenk_col_limit)
                mask_r2p_lambda(acc_S_mn, lambda s: r2p_bitmask_below(seqlenk_col_limit_r2p, s))

    @cute.jit
    def apply_mask_sm100(
        self,
        acc_S: cute.Tensor,
        m_block: Int32,
        n_block: Int32,
        thr_mma: cute.TiledMma,
        thr_tmem_load: cute.TiledCopy,
        mask_seqlen: cutlass.Constexpr[bool],
        batch_idx: Int32 = None,
        head_idx: Int32 = None,
        head_divmod=None,
        check_q_boundary: bool = False,
    ) -> None:
        """Apply seqlen-only masking for SM100."""
        acc_shape = (self.tile_m, self.tile_n)
        cS = cute.make_identity_tensor(acc_shape if not self.swap_AB else acc_shape[::-1])
        tScS = thr_mma.partition_C(cS)
        tScS = tScS[(None, None), 0, 0]
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        if n_block < 0:
            n_block = 0
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n
        if const_expr(mask_seqlen):
            mask_r2p_lambda(
                acc_S,
                lambda s: r2p_bitmask_below(seqlenk_col_limit, s),
                rank1=True,
            )

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        check_m_boundary: bool = True,
    ) -> None:
        """Apply seqlen-only masking for SM100 transposed layout."""
        COL = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_t2r[0][COL]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
        if const_expr(mask_seqlen):
            if seqlenk_col_limit <= 0:
                for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                    acc_S[i] = -cutlass.Float32.inf
