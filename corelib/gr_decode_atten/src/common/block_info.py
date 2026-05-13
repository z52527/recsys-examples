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

# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
from typing import Tuple, Optional
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr

from .seqlen_info import SeqlenInfoQK, SeqlenInfoQKNewK


@dataclass(frozen=True)
class BlockInfo:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    is_causal: cutlass.Constexpr[bool] = False
    is_local: cutlass.Constexpr[bool] = False
    is_split_kv: cutlass.Constexpr[bool] = False
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_info: SeqlenInfoQK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        n_block_max = cute.ceil_div(seqlen_info.seqlen_k, self.tile_n)
        n_block_min = 0
        if cutlass.const_expr(self.is_split_kv):
            # Floor-based split: first `rem` splits get (base+1) blocks,
            # remaining splits get `base` blocks. Beam phase runs on the
            # last split which has the least context work.
            # Guarantees no empty splits when num_splits <= total (interface).
            total = n_block_max - n_block_min
            base = total // num_splits
            rem = total - base * num_splits
            orig_min = n_block_min
            n_block_min = orig_min + split_idx * base + cutlass.min(split_idx, rem)
            n_block_max = orig_min + (split_idx + 1) * base + cutlass.min(split_idx + 1, rem)
        return n_block_min, n_block_max

    @cute.jit
    def get_m_block_min_max(self, seqlen_info: SeqlenInfoQK, n_block: Int32) -> Tuple[Int32, Int32]:
        m_block_max = cute.ceil_div(seqlen_info.seqlen_q, self.tile_m)
        m_block_min = 0
        return m_block_min, m_block_max

    @cute.jit
    def get_n_block_k_new_min_max(
        self,
        seqlen_info: SeqlenInfoQKNewK,
        m_block: Int32,
        split_idx: Int32 = 0,
        num_splits: Int32 = 1,
    ) -> Tuple[Int32, Int32]:
        """Get the block range for new K tokens (append KV)."""
        n_block_min, n_block_max = self.get_n_block_min_max(
            seqlen_info,
            m_block,
            split_idx,
            num_splits,
        )
        idx_k_new_min = cutlass.max(n_block_min * self.tile_n - seqlen_info.seqlen_k_og, 0)
        idx_k_new_max = cutlass.min(
            n_block_max * self.tile_n - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new
        )
        n_block_new_min = idx_k_new_min // self.tile_n
        n_block_new_max = (
            cute.ceil_div(idx_k_new_max, self.tile_n)
            if idx_k_new_max > idx_k_new_min
            else n_block_new_min
        )
        return n_block_new_min, n_block_new_max
