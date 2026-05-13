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

"""Unified kernel configuration shared by all architectures.

Centralizes head_dim padding, dtype validation, and common parameters
that SM80/SM90/SM100 all need. Each architecture kernel uses this via
composition (not inheritance).
"""

import math
from typing import Type, Optional

import cutlass
from cutlass import Float32, Int32, const_expr


def _ceil_to_multiple(x: int, multiple: int) -> int:
    return int(math.ceil(x / multiple) * multiple)


class KernelConfig:
    """Common kernel configuration for all architectures.

    Usage in each smXX kernel:
        config = KernelConfig(head_dim=128, head_dim_v=128, ...)
        # Access: config.head_dim_padded, config.check_hdim_oob, etc.
    """

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
    ):
        hdim_multiple_of = 16
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim

        self.head_dim_padded = _ceil_to_multiple(head_dim, hdim_multiple_of)
        self.head_dim_v_padded = _ceil_to_multiple(head_dim_v, hdim_multiple_of)
        self.check_hdim_oob = head_dim != self.head_dim_padded
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.same_hdim_kv = head_dim == head_dim_v

        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m
        self.tile_n = tile_n

    @staticmethod
    def check_type(
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Optional[Type[cutlass.Numeric]],
        mCuSeqlensQ_type: Optional[Type[cutlass.Numeric]] = None,
        mCuSeqlensK_type: Optional[Type[cutlass.Numeric]] = None,
        mSeqUsedQ_type: Optional[Type[cutlass.Numeric]] = None,
        mSeqUsedK_type: Optional[Type[cutlass.Numeric]] = None,
    ):
        """Unified dtype validation for all architectures."""
        if const_expr(not (mQ_type == mK_type == mV_type)):
            raise TypeError("Q/K/V tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported for Q/K/V")
        if const_expr(mO_type not in [cutlass.Float16, cutlass.BFloat16, Float32]):
            raise TypeError("O tensor must be Float16, BFloat16, or Float32")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")
