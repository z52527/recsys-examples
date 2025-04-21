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
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.
 ******************************************************************************/
#pragma once
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "hstu.h"

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include <cute/tensor.hpp>

#include "block_info.h"
#include "kernel_traits.h"
#include "mask.h"
#include "static_switch.h"
#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, typename Params>
inline __device__ void hstu_compute_attn_1rowblock(const Params& params,
                                                   const int bidb,
                                                   const int bidh,
                                                   int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using index_t = typename Kernel_traits::index_t;

  // Shared memory.
  extern __shared__ char smem_[];

  // The thread index.
  const int tidx = threadIdx.x;
  constexpr bool Is_delta_q = Kernel_traits::Is_delta_q;
  constexpr bool Is_causal = Kernel_traits::Is_causal;
  constexpr bool Is_target = Kernel_traits::Is_target;
  constexpr bool Is_context = Kernel_traits::Is_context;

  constexpr bool Is_even_MN = Kernel_traits::Is_even_MN;
  constexpr bool Is_even_K = Kernel_traits::Is_even_K;
  constexpr bool Is_even_Rab = Kernel_traits::Is_even_Rab;
  constexpr bool Is_local = Kernel_traits::Is_local;
  constexpr bool Has_rab = Kernel_traits::Has_rab;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const HstuBlockInfo binfo(params, bidb);
  if (m_block * kBlockM >= binfo.actual_seqlen_q) {
    return;
  }

  const int actual_seq_q = binfo.actual_seqlen_q;
  const int actual_seq_k = binfo.actual_seqlen_k;
  const int actual_seq_t = Is_target ? binfo.actual_seqlen_t : 0;
  const int actual_seq_c = Is_context ? binfo.actual_seqlen_c : 0;
  const int actual_seq_h = Is_target ? actual_seq_k - actual_seq_t : actual_seq_k; // note: use k for delta_q
  const bool is_jump = Is_target && m_block * kBlockM > actual_seq_h;
  const bool is_context = Is_context && (m_block + 1) * kBlockM <= actual_seq_c;
  const bool is_mixed_context = Is_context && (m_block + 1) * kBlockM > actual_seq_c && m_block * kBlockM < actual_seq_c;

  const int n_block_history = cute::ceil_div(actual_seq_h, kBlockN);
  const int actual_seq_offset = Is_delta_q ? actual_seq_k - actual_seq_q : 0;  
  const int target_index = (m_block * kBlockM - actual_seq_h) / params.target_group_size;

  // calculate n_block_min and n_block_max
  int n_block_min =
    (!Is_local)
        ? 0
        : std::max(0, (m_block * kBlockM + actual_seq_offset - params.window_size_left) /
                          kBlockN);
  if constexpr (Is_context) {
    n_block_min = (is_context || is_mixed_context) ? 0 : n_block_min;
  }
  int n_block_max = cute::ceil_div(actual_seq_k, kBlockN);
  if constexpr (Is_causal || Is_local) {
    n_block_max = std::min(
        n_block_max,
        cute::ceil_div((m_block + 1) * kBlockM + actual_seq_offset + params.window_size_right,
                      kBlockN));
  }
  if constexpr (Is_context) {
    n_block_max = (is_context || is_mixed_context) ? std::max(cute::ceil_div(actual_seq_h, kBlockN), n_block_max) : n_block_max;
  }

  // calculate n_masking_block_max and n_masking_block_min
  int n_masking_block_max = cute::ceil_div(std::min(actual_seq_k, (m_block + 1) * kBlockM + actual_seq_offset), kBlockN); // up
  int n_masking_block_min = (m_block * kBlockM + actual_seq_offset) / kBlockN;
  if constexpr (Is_target) {
    n_masking_block_min = is_jump ? (actual_seq_h + actual_seq_offset + target_index * params.target_group_size) / kBlockN : n_masking_block_min;
  }
  if constexpr (Is_context) {
    n_masking_block_min = is_mixed_context ? n_block_min : n_masking_block_min;
    n_masking_block_max = is_mixed_context ? n_block_max : n_masking_block_max;
  }
  const int n_masking_steps = (!Is_causal || is_context) ? 0 : n_masking_block_max - n_masking_block_min;

  // We exit early and write 0 to gO and gLSE. This also covers the case where
  // actual_seqlen_k == 0. Otherwise we might read OOB elements from gK and gV.
  if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) +
                      binfo.q_offset(params.o_row_stride)),
        make_shape(actual_seq_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                          make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    clear(tOrO);
    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(
        size<0>(gO), size<1>(gO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if constexpr (!Is_even_K) {
      #pragma unroll
      for (int k = 0; k < size(tOpO); ++k) {
        tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
      }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false,
                /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
        actual_seq_q - m_block * kBlockM);
    return;
  }

  // We iterate over the blocks in reverse order. This is because the last block
  // is the only one that needs masking when we read K and V from global memory.
  // Moreover, iterating in reverse might save us 1 register (we just need
  // n_block instead of both n_block and n_block_max).

  Tensor mQ =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) +
                                binfo.q_offset(params.q_row_stride)),
                  make_shape(actual_seq_q, params.h, params.d),
                  make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
  Tensor mK =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) +
                                binfo.k_offset(params.k_row_stride)),
                  make_shape(actual_seq_k, params.h_k, params.d),
                  make_stride(params.k_row_stride, params.k_head_stride, _1{}));
  Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
  Tensor mV =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) +
                                binfo.k_offset(params.v_row_stride)),
                  make_shape(actual_seq_k, params.h_k, params.d),
                  make_stride(params.v_row_stride, params.v_head_stride, _1{}));
  Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)

  const int bidh_rab = (params.h_rab > 1) ? bidh : 0;
  size_t rab_qkv_not_equal_offset = bidb * params.rab_seqlen_qk_stride + bidh_rab * params.rab_seqlen_q_stride + params.seqlen_k_rounded * actual_seq_offset;
  auto mRab =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.rab_ptr) + rab_qkv_not_equal_offset),
                  make_shape(actual_seq_q, params.seqlen_k_rounded),
                  make_stride(params.rab_seqlen_k_stride, _1{}));
  auto gRab = local_tile(mRab(_, _),
                        make_shape(Int<kBlockM>{}, Int<kBlockN>{}),
                        make_coord(m_block, _));

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
  Tensor sK =
      make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                  typename Kernel_traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
  Tensor sVt =
      make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(
      sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});
  Tensor sRab = make_tensor(sV.data() + size(sV),
                            typename Kernel_traits::SmemLayoutRab{});

  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  typename Kernel_traits::GmemTiledCopyRab gmem_tiled_copy_Rab;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  auto gmem_thr_copy_Rab = gmem_tiled_copy_Rab.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK =
      gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV =
      gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
  auto tQgRab = gmem_thr_copy_Rab.partition_S(gRab);
  auto tQsRab = gmem_thr_copy_Rab.partition_D(sRab);

  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);  // (MMA,MMA_M,MMA_K)
  Tensor tSrK = thr_mma.partition_fragment_B(sK);  // (MMA,MMA_N,MMA_K)
  Tensor tOrVt =
      thr_mma.partition_fragment_B(sVtNoSwizzle);  // (MMA, MMA_K,MMA_N)
  Tensor acc_o = partition_fragment_C(
      tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

  //
  // Copy Atom retiling
  //

  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(
      typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  auto smem_tiled_copy_rab =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(tidx);
  auto tSsRab = smem_thr_copy_rab.partition_S(sRab);
  //
  // PREDICATES
  //
  // Construct identity layout for sQ and sK
  // c = coord
  Tensor cQ = make_identity_tensor(
      make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  Tensor cKV = make_identity_tensor(
      make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)

  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(
      cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(
      cKV);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

  // Set predicates for k bounds
  if constexpr (!Is_even_K) {
    #pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
    #pragma unroll
    for (int k = 0; k < size(tKVpKV); ++k) {
      tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
    }
  }

  auto cRab = make_identity_tensor(make_shape(size<0>(sRab), size<1>(sRab)));
  auto tQcRab = gmem_thr_copy_Rab.partition_S(cRab);
  
  // Prologue
  int n_block = n_block_max - 1;
  
  auto copy_if_g2s_rab = [&](int n_block_id) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
    #pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
      if (get<0>(tQcRab(0, m, 0)) < (actual_seq_q - m_block * kBlockM)) {
        #pragma unroll
        for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
          if (Is_even_Rab || get<1>(tQcRab(0, m, k)) < (actual_seq_k - n_block_id * kBlockN)) {
            cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k),
                      tQsRab(_, m, k));
          } 
        }
      }
    }
  };
  auto copy_g2s_rab = [&](int n_block_id) {
    auto ctQgRab_view = tQgRab(_, _, _, n_block_id);
    #pragma unroll
    for (int m = 0; m < size<1>(ctQgRab_view); ++m) {
      #pragma unroll
      for (int k = 0; k < size<2>(ctQgRab_view); ++k) {
        if (Is_even_Rab || get<0>(tQcRab(0, m, k)) < (actual_seq_q - m_block * kBlockM)) {
          cute::copy(gmem_tiled_copy_Rab, ctQgRab_view(_, m, k),
                    tQsRab(_, m, k));
        }
      }
    }
  };

  if constexpr (Has_rab) {
    copy_if_g2s_rab(n_block);
  }

  // We don't need to clear the sQ smem tiles since we'll only write out the
  // valid outputs
  // prefill q
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                     actual_seq_q - m_block * kBlockM);
  if (Kernel_traits::Is_Q_in_regs) {
    cute::cp_async_fence();
  }

  if (Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<0>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    __syncthreads();
  }

  // prefill k
  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false>(
    gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    actual_seq_k - n_block * kBlockN);
  cute::cp_async_fence();

  if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    flash::cp_async_wait<1>();
    __syncthreads();
    Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));  // M
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
  }

  clear(acc_o);

  auto col_limit_right = [&](int row, int n_block) {
    return std::min(
        actual_seq_k,
        row + 1 + actual_seq_offset + params.window_size_right
    );
  };
  auto col_limit_left = [&](int row, int n_block) {
    return std::max(
        0,
        row + actual_seq_offset - params.window_size_left
    );
  };

  auto apply_mask = [&](auto& tensor, int n_block) {
    static constexpr int Row = 0, Col = 1;
    Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thr_mma.partition_C(cS);
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
      int row = int(get<Row>(tScS(i))) + m_block * kBlockM;
      int col = int(get<Col>(tScS(i))) + n_block * kBlockN;
      if constexpr (!Is_causal && !Is_local) { 
        if (col >= int(actual_seq_k)) { 
          tensor(i) = -INFINITY; 
        }
      } else {  
        if constexpr (Is_context && !Is_delta_q) { // because the definition is unclear
          if (row < actual_seq_c && col < actual_seq_h) { // The scores located upper on actual_seq_c do not require masking.
            continue;
          }
        }
        if (col >= col_limit_right(row, n_block)) { // causal mask
          tensor(i) = -INFINITY;
        }
        if constexpr (Is_local) {
          if (col < col_limit_left(row, n_block)) {
            tensor(i) = -INFINITY;
          }
        }
        if constexpr (Is_target && !Is_delta_q) { // because the definition is unclear
          if (row >= actual_seq_h) {
            const int target_index = (row - actual_seq_h) / params.target_group_size; 
            const int target_col_limit_left = actual_seq_h + target_index * params.target_group_size;
            if (col < target_col_limit_left && col >= actual_seq_h) {
              tensor(i) = -INFINITY;
            }
          }
        }
      }
    }
  };

  // For performance reason, we separate out two kinds of iterations:
  // those that need masking on S, and those that don't.
  // We need masking on S for the very last block when K and V has length not
  // multiple of kBlockN. We also need masking on S if it's causal, for the last
  // ceil_div(kBlockM, kBlockN) blocks. We will have at least 1 "masking"
  // iteration.
  
  for (int masking_step = 0; n_block >= n_block_min; ++masking_step, --n_block) {
    // When jumps occur, it is necessary to apply a mask for the mixed situation
    const bool is_masking = masking_step < n_masking_steps || (n_block + 1) * kBlockN > actual_seq_h;
    Tensor acc_s = partition_fragment_C(
    tiled_mma,
    Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
    flash::cp_async_wait<0>();
    __syncthreads();

    // async load(next(v))
    // Advance gV
    if (masking_step > 0) {
      flash::copy</*Is_even_MN=*/true, Is_even_K>(
          gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
    } else {
      flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, actual_seq_k - n_block * kBlockN);
    }
    cute::cp_async_fence();
    
    // compute q @ k
    if constexpr (Has_rab) {
      Tensor rRab = make_tensor<Element>(
        partition_shape_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}));
      auto tSrRab_view = smem_thr_copy_rab.retile_D(rRab);
      cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _),
                          tSrRab_view(_, _, _));
      flash::convert_type_safe(rRab, acc_s);
    } else {
      clear(acc_s);
    }
    flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
    acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
    smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

    if (Is_local || is_masking) {
      apply_mask(acc_s, n_block);
    }

    flash::cp_async_wait<0>();
    __syncthreads();
    if (n_block > n_block_min) {
      if (is_jump && masking_step == n_masking_steps - 1) {
        n_block = std::min(n_block, n_block_history); // need reconsider
      }
      // async load(next(k))
      flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV,
                                                  tKgK(_, _, _, n_block - 1),
                                                  tKsK, tKVcKV, tKVpKV);
      // This cp_async_fence needs to be in the if block, otherwise the
      // synchronization isn't right and we get race conditions.
      if constexpr (Has_rab) {
        copy_g2s_rab(n_block - 1);
      }
      cute::cp_async_fence();
    }
    for (int i = 0; i < size(acc_s); ++i) {
      acc_s(i) *= params.alpha;
    }
    silu(acc_s);

    // Convert acc_s from fp32 to fp16/bf16
    Tensor rP = make_tensor_like<Element>(acc_s);
    flash::convert_type_safe(acc_s, rP);

    // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
    // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
    Tensor tOrP = make_tensor(
        rP.data(),
        flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
    // compute qk @ v
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V,
                    smem_thr_copy_V);
  }

  // scale acc_o
  for (int i = 0; i < size(acc_o); ++i) {
    acc_o(i) /= params.seqlen_q;
  }

  // Epilogue
  // Convert acc_o from fp32 to fp16/bf16
  Tensor rO = make_tensor_like<Element>(acc_o);
  flash::convert_type_safe(acc_o, rO);
  Tensor sO = make_tensor(
      sQ.data(), typename Kernel_traits::SmemLayoutO{});  // (SMEM_M,SMEM_N)
  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO =
      smem_thr_copy_O.retile_S(rO);  // ((Atom,AtomNum), MMA_M, MMA_N)
  Tensor taccOsO =
      smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  // sO has the same size as sQ, so we don't need to sync here.
  if (Kernel_traits::Share_Q_K_smem) {
    __syncthreads();
  }

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor mO =
      make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) +
                                binfo.q_offset(params.o_row_stride)),
                  make_shape(actual_seq_q, params.h, params.d),
                  make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO =
      gmem_thr_copy_O.partition_S(sO);  // ((Atom,AtomNum),ATOM_M,ATOM_N)
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  __syncthreads();

  Tensor tOrO = make_tensor<Element>(shape(tOgO));
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

  // Construct identity layout for sO
  Tensor cO = make_identity_tensor(
      make_shape(size<0>(sO), size<1>(sO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tOcO =
      gmem_thr_copy_O.partition_D(cO);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
  if (!Is_even_K) {
    #pragma unroll
    for (int k = 0; k < size(tOpO); ++k) {
      tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
    }
  }
  // Clear_OOB_K must be false since we don't want to write zeros to gmem
  flash::copy<Is_even_MN, Is_even_K,
              /*Clear_OOB_MN=*/false,
              /*Clear_OOB_K=*/false>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO,
                                    actual_seq_q - m_block * kBlockM);
}

template <typename Kernel_traits, typename Params>
__global__ void hstu_fwd_kernel(Params params) {
  int m_block = gridDim.x - blockIdx.x - 1;
  // The block index for the batch.
  const int bidb = blockIdx.y;
  // The block index for the head.
  const int bidh = blockIdx.z;

  // We want the fwd and bwd to generate the same dropout pattern (RNG), without
  // restricting them to have the same number of threads or have to traverse the
  // attention matrix in the same order. In the Philox RNG, we use the offset to
  // store the batch, head, and the lane id (within a warp). We use the
  // subsequence to store the location of the 16 x 32 blocks within the
  // attention matrix. This way, as long as we have the batch, head, and the
  // location of the 16 x 32 block within the attention matrix, we can generate
  // the exact same dropout pattern.

  hstu_compute_attn_1rowblock<Kernel_traits>(params, bidb, bidh, m_block);
}

}  // namespace flash

template <typename elem_type,
          int kHeadDim,
          int kBlockM,
          int kBlockN,
          int kNWarps,
          bool Is_delta_q,
          bool Is_causal,
          bool Is_target,
          bool Is_context,
          bool Is_local,
          bool Has_rab,
          bool Is_even_K = true,
          bool Is_even_Rab = true,
          bool Is_even_MN = false,
          bool Is_Q_in_regs = false,
          bool Share_Q_K_smem = false>
void run_hstu_fwd_impl(Hstu_fwd_params& params, cudaStream_t stream) {
  using Kernel_traits = Hstu_fwd_kernel_traits<
      kHeadDim, kBlockM, kBlockN, kNWarps, Is_delta_q, Is_causal, Is_target, Is_context, Is_local && !Is_causal,
      Has_rab, Is_even_K, Is_even_Rab, Is_even_MN, Is_Q_in_regs, Share_Q_K_smem, elem_type>;
  
  constexpr size_t smem_size = Kernel_traits::kSmemSize;
  // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
  // https://github.com/kokkos/kokkos-kernels/issues/349
  // https://github.com/HazyResearch/flash-attention/issues/21
  TORCH_CHECK(Kernel_traits::kHeadDim == params.d);
  const int num_m_block =
      (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.b, params.h);
  auto kernel = &flash::hstu_fwd_kernel<Kernel_traits, Hstu_fwd_params>;

  if (smem_size >= 48 * 1024) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }

  kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename elem_type, int kHeadDim, bool Has_rab, bool Is_local,
          bool Is_causal, bool Is_context, bool Is_target, bool Is_delta_q>
void run_hstu_fwd_(Hstu_fwd_params& params, cudaStream_t stream) {
  const bool even_Rab = params.seqlen_q % 128 == 0 && params.seqlen_k % 64 == 0; // BLOCK_M = 128, BLOCK_N = 64
  assert((params.d == kHeadDim) && "error: params.d should equal kHeadDim!");
  BOOL_SWITCH(even_Rab, Is_even_Rab, [&] {
    if constexpr (kHeadDim <= 128) {
      run_hstu_fwd_impl<elem_type, kHeadDim, 128, 64, 8, Is_delta_q, Is_causal, Is_target, Is_context, Is_local,
                      Has_rab, /*Is_even_K=*/true, /*Is_even_Rab=*/Is_even_Rab, /*Is_even_MN=*/false, /*Is_Q_in_regs=*/true, /*Share_Q_K_smem=*/true>(params, stream);
      return;
    }
    run_hstu_fwd_impl<elem_type, kHeadDim, 128, 64, 8, Is_delta_q, Is_causal, Is_target, Is_context, Is_local,
                      Has_rab, /*Is_even_K=*/true, /*Is_even_Rab=*/Is_even_Rab>(params, stream);
  });
}