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

#include "cute/tensor.hpp"

#include "cutlass/cluster_launch.hpp"
#include "cutlass/device_kernel.h"  // For device_kernel

#include "static_switch.h"
#include "flash.h"
#include "flash_bwd_postprocess_kernel.h"
#include "tile_scheduler_bwd.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_bwd_sm90_tma.hpp"
#include "flash_bwd_kernel.h"

using namespace cute;

template <int Arch, int kHeadDim, int kBlockM, int kBlockN, typename Element,
          bool Has_Rab, bool Has_dRab, bool Is_causal, bool Is_target, bool Is_context, bool Is_delta_q, bool Is_local, bool Deterministic,
          int Stages_dO=2, int Stages_dS=2,
          bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using ElementAccum = float;
    int const total_q_padded_rounded = cute::round_up(params.total_q + params.b * kBlockM, kBlockM);
    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;
    static constexpr int Stages = Stages_dS;
    using CollectiveMainloop = flash::CollectiveMainloopBwd<Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm90,
            Has_Rab, Has_dRab, Is_causal, Is_target, /*Is_context=*/false, Is_delta_q, Is_local, Deterministic, // TO DEBUG: remove Is_context
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ>;
    using CollectiveEpilogue = flash::CollectiveEpilogueBwd<TileShape_MNK, Element, CollectiveMainloop::NumMmaThreads, dKV_swapAB, NumMmaWarpGroups / AtomLayoutNdKV>;
    using Scheduler = flash::SingleTileSchedulerBwd;
    using AttnKernel = flash::FlashAttnBwd<CollectiveMainloop, CollectiveEpilogue, Scheduler>;
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {params.total_q, params.d, params.h, 1},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, 0},  // stride_Q
        static_cast<Element const*>(params.rab_ptr),
        {Is_delta_q ? params.seqlen_k : params.seqlen_q, params.seqlen_k, params.h_rab, params.b},  // shape_Rab
        {params.rab_row_stride, _1{}, params.rab_head_stride, params.rab_batch_stride},  // stride_Rab
        static_cast<Element const*>(params.k_ptr),
        {params.total_k, params.d, params.h_k, 1},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, 0},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {params.v_row_stride, _1{}, params.v_head_stride, 0},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, 0},  // stride_dO
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {total_q_padded_rounded, params.d_rounded, params.h, 1},  // shape_dQaccum
        {params.d_rounded, _1{}, params.d_rounded * total_q_padded_rounded, 0}, // stride_dQaccum
        static_cast<Element*>(params.drab_ptr),
        {Is_delta_q ? params.seqlen_k : params.seqlen_q, params.seqlen_k, params.h_rab, params.b},  // shape_dRab
        {params.drab_row_stride, _1{}, params.drab_head_stride, params.drab_batch_stride},  // stride_dRab
        params.seqlen_q, params.seqlen_k, params.seqlen_i,
        params.window_size_left, params.window_size_right,
        params.b, params.alpha,
        params.dq_semaphore,
        params.cu_seqlens_q, params.cu_seqlens_k, params.num_targets
    };

    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<Element*>(params.dk_ptr),
        {params.total_k, params.d, params.h, 1},  // shape_dK
        {params.dk_row_stride, _1{}, params.dk_head_stride, 0},  // stride_dK
        static_cast<Element*>(params.dv_ptr),
        {params.dv_row_stride, _1{}, params.dv_head_stride, 0},
        params.cu_seqlens_k
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename Scheduler::Arguments scheduler_args {
        num_blocks_n, params.h, params.b, params.tile_count_semaphore, params.cu_seqlens_k
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_q));
    // int smem_size_rab = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_rab));
    // int smem_size_do = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_do));
    // int smem_size_ds = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_ds));
    // int smem_size_dqacc = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_dqacc));
    // int smem_size_k = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_k));
    // int smem_size_v = sizeof(decltype((typename AttnKernel::SharedStorage{}).mainloop.smem_v));
    // printf("smem_size = %d, q = %d, rab = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d\n", smem_size, smem_size_q, smem_size_rab, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc);
    // Get the ptr to kernel function.

    void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    CHECK_CUDA_KERNEL_LAUNCH();

    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_MK, Element, ElementAccum, cutlass::arch::Sm90,
        AttnKernel::CollectiveMainloop::kNThreadsdQ,
        typename AttnKernel::CollectiveMainloop::SmemLayoutdQaccumTMA,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB
        >;
    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        {total_q_padded_rounded, params.d_rounded, params.h, 1},  // shape_dQaccum
        {params.d_rounded, _1{}, params.d_rounded * total_q_padded_rounded, 0}, // stride_dQaccum
        static_cast<Element*>(params.dq_ptr),
        {params.total_q, params.d, params.h, 1},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
        params.cu_seqlens_q
    };
    typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
    int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
    dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
    // Get the ptr to kernel function.
    auto postprocess_kernel = cutlass::device_kernel<PostprocessKernel>;
    int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
    if (smem_size_postprocess >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(postprocess_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    postprocess_kernel<<<grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream>>>(postprocess_params);
    CHECK_CUDA_KERNEL_LAUNCH();

}

template<int Arch, typename T>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            DRAB_SWITCH(params.has_drab, Has_dRab, [&] {
                                // BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                                    run_flash_bwd<Arch, Headdim, 64, 128, T, Has_Rab, Has_dRab, Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, false, 1, 2, true, false, false, 2, 1, 2, 1>(params, stream);
                                // });
                            });
                        });
                    });
                });
            });
        });
    });
}

template<int Arch, typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            DRAB_SWITCH(params.has_drab, Has_dRab, [&] {
                                // BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                                    run_flash_bwd<Arch, Headdim, 64, 128, T, Has_Rab, Has_dRab, Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, false, 1, 2, true, false, true, 2, 1, 2, 2>(params, stream);
                                // });
                            });
                        });
                    });
                });
            });
        });
    });
}

template<int Arch, typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            DRAB_SWITCH(params.has_drab, Has_dRab, [&] {
                                // BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                                    run_flash_bwd<Arch, Headdim, 64, 64, T, Has_Rab, Has_dRab, Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, false, 1, 2, false, true, true, 2, 1, 1, 1>(params, stream);
                                // });
                            });
                        });
                    });
                });
            });
        });
    });
}

template<int Arch, typename T>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            DRAB_SWITCH(params.has_drab, Has_dRab, [&] {
                                // BOOL_SWITCH(params.deterministic, Deterministic, [&] {
                                    run_flash_bwd<Arch, Headdim, 64, 64, T, Has_Rab, Has_dRab, Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, false, 1, 2, false, true, true, 2, 1, 1, 1>(params, stream);
                                // });
                            });
                        });
                    });
                });
            });
        });
    });
}
