/******************************************************************************
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
******************************************************************************/
/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "utils.h"

namespace flash {

using namespace cute;

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask(Tensor<Engine, Layout>& tensor,
                                           const int max_seqlen_k,
                                           const int col_idx_offset_ = 0) {
  // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
  static_assert(Layout::rank == 2, "Only support 2D Tensor");
  const int lane_id = threadIdx.x % 32;
  const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
  #pragma unroll
  for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
    const int col_idx_base = col_idx_offset + nj * 8;
    #pragma unroll
    for (int j = 0; j < size<1, 0>(tensor); ++j) {
      const int col_idx = col_idx_base + j;
      if (col_idx >= max_seqlen_k) {
        // Without the "make_coord" we get wrong results
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); ++mi) {
          tensor(mi, make_coord(j, nj)) = 0.f;
        }
      }
    }
  }
}

template <bool HasWSLeft = true, typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_local(Tensor<Engine, Layout>& tensor,
                                                 const int col_idx_offset_,
                                                 const int max_seqlen_k,
                                                 const int row_idx_offset,
                                                 const int max_seqlen_q,
                                                 const int warp_row_stride,
                                                 const int window_size_left,
                                                 const int window_size_right) {
  // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
  static_assert(Layout::rank == 2, "Only support 2D Tensor");
  const int lane_id = threadIdx.x % 32;
  const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
  #pragma unroll
  for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
    #pragma unroll
    for (int i = 0; i < size<0, 0>(tensor); ++i) {
      const int row_idx = row_idx_base + i * 8;
      const int col_idx_limit_left =
          std::max(0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
      const int col_idx_limit_right =
          std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q +
                                     window_size_right);
      #pragma unroll
      for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j) {
          const int col_idx = col_idx_base + j;
          if (col_idx >= col_idx_limit_right ||
              (HasWSLeft && col_idx < col_idx_limit_left)) {
            tensor(make_coord(i, mi), make_coord(j, nj)) = 0.f;
          }
        }
      }
      // if (cute::thread0()) {
      //     printf("mi = %d, i = %d, row_idx = %d, max_seqlen_k = %d\n", mi, i,
      //     row_idx, max_seqlen_k); print(tensor(make_coord(i, mi), _));
      //     // print(tensor(_, j + nj * size<1, 0>(tensor)));
      // }
    }
  }
}

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_mask_causal(
    Tensor<Engine, Layout>& tensor,
    const int col_idx_offset_,
    const int max_seqlen_k,
    const int row_idx_offset,
    const int max_seqlen_q,
    const int warp_row_stride) {
  // Causal masking is equivalent to local masking with window_size_left =
  // infinity and window_size_right = 0
  apply_mask_local</*HasWSLeft=*/false>(tensor, col_idx_offset_, max_seqlen_k,
                                        row_idx_offset, max_seqlen_q,
                                        warp_row_stride, -1, 0);
}

template <typename Engine0,
          typename Layout0,
          typename Engine1,
          typename Layout1>
__forceinline__ __device__ void apply_mask_causal_w_idx(
    Tensor<Engine0, Layout0>& tensor,
    Tensor<Engine1, Layout1> const& idx_rowcol,
    const int col_idx_offset_,
    const int max_seqlen_k,
    const int row_idx_offset) {
  // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
  static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  static_assert(Layout1::rank == 2, "Only support 2D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(tensor) == size<0>(idx_rowcol));
  CUTE_STATIC_ASSERT_V(size<1>(tensor) == size<1>(idx_rowcol));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor); ++mi) {
    const int col_idx_limit =
        std::min(max_seqlen_k, 1 + row_idx_offset + get<0>(idx_rowcol(mi, 0)));
    #pragma unroll
    for (int ni = 0; ni < size<1, 1>(tensor); ++ni) {
      if (col_idx_offset_ + get<1>(idx_rowcol(0, ni)) >= col_idx_limit) {
        tensor(mi, ni) = -INFINITY;
      }
    }
    // if (cute::thread0()) {
    //     printf("ni = %d, j = %d, col_idx = %d, max_seqlen_k = %d\n", ni, j,
    //     col_idx, max_seqlen_k); print(tensor(_, make_coord(j, ni)));
    //     // print(tensor(_, j + ni * size<1, 0>(tensor)));
    // }
  }
}

template <bool Is_causal, bool Is_local>
struct Mask {
  const int max_seqlen_k, max_seqlen_q;
  const int window_size_left, window_size_right;
  const bool is_in_target;

  __forceinline__ __device__ Mask(const int max_seqlen_k,
                                  const int max_seqlen_q,
                                  const int window_size_left,
                                  const int window_size_right)
      : max_seqlen_k(max_seqlen_k),
        max_seqlen_q(max_seqlen_q),
        window_size_left(window_size_left),
        window_size_right(window_size_right){};
  
  __forceinline__ __device__ Mask(const int max_seqlen_k,
                                  const int max_seqlen_q,
                                  const int window_size_left,
                                  const int window_size_right,
                                  const bool is_in_target)
      : max_seqlen_k(max_seqlen_k),
        max_seqlen_q(max_seqlen_q),
        window_size_left(window_size_left),
        window_size_right(window_size_right),
        is_in_target(is_in_target){};

  // Causal_mask: whether this particular iteration needs causal masking
  template <bool Causal_mask = false,
            bool Is_even_MN = true,
            typename Engine,
            typename Layout>
  __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout>& tensor_,
                                             const int col_idx_offset_,
                                             const int row_idx_offset,
                                             const int warp_row_stride) {
    static_assert(!(Causal_mask && Is_local),
                  "Cannot be both causal and local");
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4,
                  "First dimension must be 4");
    static constexpr bool Need_masking =
        Causal_mask || Is_local || !Is_even_MN;
    if constexpr (Need_masking) {
      // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M),
      // ncol=(2, MMA_N))
      Tensor tensor = make_tensor(
          tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
      // Do we need both row and column indices, or just column incides?
      static constexpr bool Col_idx_only = !Is_local && !Causal_mask;
      const int lane_id = threadIdx.x % 32;
      const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
      if constexpr (Col_idx_only) {
        #pragma unroll
        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
          const int col_idx_base = col_idx_offset + nj * 8;
          #pragma unroll
          for (int j = 0; j < size<1, 0>(tensor); ++j) {
            const int col_idx = col_idx_base + j;
            #pragma unroll
            for (int mi = 0; mi < size<0>(tensor); ++mi) {
              // No causal, no local
              if constexpr (!Is_even_MN) {
                if (col_idx >= max_seqlen_k) {
                  tensor(mi, make_coord(j, nj)) = 0.f;
                }
              }
            }
          }
        }
      } else {
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
          const int row_idx_base = row_idx_offset + mi * warp_row_stride;
          #pragma unroll
          for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit_left = std::max(
                0, row_idx + max_seqlen_k - max_seqlen_q - window_size_left);
            const int col_idx_limit_right =
                std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k -
                                           max_seqlen_q + window_size_right);
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
              const int col_idx_base = col_idx_offset + nj * 8;
              #pragma unroll
              for (int j = 0; j < size<1, 0>(tensor); ++j) {
                const int col_idx = col_idx_base + j;
                if constexpr (Causal_mask) {
                  if (is_in_target) {
                    if (col_idx != col_idx_limit_right - 1) {
                      tensor(make_coord(i, mi), make_coord(j, nj)) = 0.f;
                    }
                  } else {
                    if (col_idx >= col_idx_limit_right) {
                      tensor(make_coord(i, mi), make_coord(j, nj)) = 0.f;
                    }
                  }
                }
                if constexpr (Is_local) {
                  if (col_idx >= col_idx_limit_right ||
                      col_idx < col_idx_limit_left) {
                    tensor(make_coord(i, mi), make_coord(j, nj)) = 0.f;
                  }
                }
                if constexpr (!Causal_mask && !Is_local && !Is_even_MN) {
                  // Causal and Local already handles MN masking
                  if (col_idx >= max_seqlen_k) {
                    tensor(make_coord(i, mi), make_coord(j, nj)) = 0.f;
                  }
                }
              }
            }
          }
        }
      }
    }
  };
};

}  // namespace flash
