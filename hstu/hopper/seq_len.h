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
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cute/layout.hpp>

namespace flash {

static constexpr int kMaxTileSize = 128;

template <bool UseVarSeqLen> class SeqLenTraits {
public:
  // Total number of queries / keys. Unpadded.
  int sum_s = 0;
  // seq len offsets.
  int *cu_seq_len = nullptr;
  // targets nums
  int *num_targets = nullptr;
  // context nums
  int *num_contexts = nullptr;
  // seq len of the current batch.
  int max_seq_len = -1;
  int actual_seq_len = -1;
  int actual_seq_len_h = -1;
  int actual_seq_len_c = 0; // will be modified to -1

  static constexpr bool kUseVarSeqLen = UseVarSeqLen;

  using ShapeT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t, int32_t>, 
      cute::Shape<int32_t, int32_t, int32_t, int32_t>
  >;
  using StrideT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1, int64_t>, 
      cute::Shape<int64_t, _1, int64_t, int64_t>
  >;
  using LayoutT = cute::Layout<ShapeT, StrideT>;

  using ShapeRabT = cute::Shape<int32_t, int32_t, int32_t, int32_t>;
  using StrideRabT = cute::Shape<int64_t, _1, int64_t, int64_t>;
  using LayoutRabT = cute::Layout<ShapeRabT, StrideRabT>;

  using ShapeLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int32_t, int32_t>, 
      cute::Shape<int32_t, int32_t, int32_t>
  >;
  using StrideLseT = std::conditional_t<
      UseVarSeqLen, 
      cute::Shape<int64_t, _1>, 
      cute::Shape<int64_t, int64_t, _1>
  >;
  using LayoutLseT = cute::Layout<ShapeLseT, StrideLseT>;

  CUTLASS_HOST SeqLenTraits() {}

  CUTLASS_HOST SeqLenTraits(
      int sum_s, int max_seq_len, int *cu_seq_len = nullptr, int *num_targets = nullptr, int *num_contexts = nullptr):
      sum_s(sum_s), max_seq_len(max_seq_len), cu_seq_len(cu_seq_len), num_targets(num_targets), num_contexts(num_contexts) {}
  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum.
  CUTLASS_HOST_DEVICE auto get_gmem_layout(
      int m, int k, int h, int b, 
      int64_t m_stride, int64_t h_stride, int64_t b_stride,
      bool padded = false) const {
    static_assert(!UseVarSeqLen, "Default implementation is for FixedSeqLen.");
    return make_layout(make_shape(m, k, h, b),
                       make_stride(m_stride, cute::_1{}, h_stride, b_stride));
  }

  // Returns the layout of a tensor in MKHB format in global memory.
  // padded: only useful for var-seq-len for dq_accum.
  CUTLASS_HOST_DEVICE auto get_lse_gmem_layout(
      int m, int h, int b, bool padded = false) const {
    static_assert(!UseVarSeqLen, "Default implementation is for FixedSeqLen.");
    return make_layout(make_shape(b, h, m),
                       make_stride(int64_t(h * m), int64_t(m), cute::_1()));
  }

  CUTLASS_DEVICE void init(int bidb) {}
  CUTLASS_DEVICE void init_h(int bidb) {}
  CUTLASS_DEVICE void init_c(int bidb) {}

  template <typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, bool padded = false) const {
    auto g_tensor = local_tile(
      m_tensor(_, _, bidh, bidb), tile_shape, make_coord(_, _0{}));
    return g_tensor;
  }

  template <typename MTensor, typename Shape>
  CUTLASS_DEVICE auto get_lse_local_tile_tensor(
      const MTensor &m_tensor, const Shape &tile_shape, 
      int bidh, int bidb, bool padded = false) const {
    auto g_tensor = local_tile(m_tensor(bidb, bidh, _), tile_shape, make_coord(_));
    return g_tensor;
  }
};

using FixedSeqLenTraits = SeqLenTraits<false>;

using VarSeqLenTraits = SeqLenTraits<true>;

// Returns the static layout of a var-seq-len tensor in global memory based on
// max_seq_len and max_batch_size.
// padded: only useful for var-seq-len for dq_accum.
// When padded is True, use B_M + kMaxTileSize * B as the total B_M.
template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_gmem_layout(
    int m, int k, int h, int b, 
    int64_t m_stride, int64_t h_stride, int64_t b_stride,
    bool padded) const {
  return make_layout(
    make_shape(sum_s + (padded ? kMaxTileSize * b : 0), k, h), 
    make_stride(m_stride, cute::_1{}, h_stride));
}

// padded: only useful for var-seq-len for dq_accum.
// When padded is True, use B_M + kMaxTileSize * B as the total B_M.
template <>
CUTLASS_HOST_DEVICE auto VarSeqLenTraits::get_lse_gmem_layout(
    int m, int h, int b, bool padded) const {
  return make_layout(
    make_shape(h, sum_s + (padded ? kMaxTileSize * b : 0)), 
    make_stride(int64_t(sum_s + (padded ? kMaxTileSize * b : 0)), cute::_1()));
}

template <>
CUTLASS_DEVICE void VarSeqLenTraits::init(int bidb) {
  actual_seq_len = cu_seq_len[bidb + 1] - cu_seq_len[bidb];
}

template <>
CUTLASS_DEVICE void VarSeqLenTraits::init_h(int bidb) {
  int actual_seq_len_t = 0;
  if (num_targets != nullptr) { // could remove this check in the future
    actual_seq_len_t = num_targets[bidb];
  } 
  actual_seq_len_h = actual_seq_len - actual_seq_len_t;
}

template <>
CUTLASS_DEVICE void VarSeqLenTraits::init_c(int bidb) {
  if (num_contexts != nullptr) { // could remove this check in the future
    actual_seq_len_c = num_contexts[bidb];
  } 
}

template <>
template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, bool padded) const {
  auto g_offset = local_tile(
      m_tensor(_, _, bidh), 
      cute::make_shape(1, get<1>(tile_shape)), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0), _0{}));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(
        cute::make_shape(actual_seq_len, get<1>(tile_shape)), 
        g_offset.stride()
      ));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_, _0{}));
  return g_tensor;
}

template <>
template <typename MTensor, typename Shape>
CUTLASS_DEVICE auto VarSeqLenTraits::get_lse_local_tile_tensor(
    const MTensor &m_tensor, const Shape &tile_shape,
    int bidh, int bidb, bool padded) const {
  auto g_offset = local_tile(
      m_tensor(bidh, _), cute::make_shape(_1{}), 
      make_coord(cu_seq_len[bidb] + (padded ? kMaxTileSize * bidb : 0)));
  auto g_sequence = make_tensor(
      g_offset.data(), 
      make_layout(cute::make_shape(actual_seq_len), cute::make_shape(_1{})));
  auto g_tensor = local_tile(g_sequence, tile_shape, make_coord(_));
  return g_tensor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
