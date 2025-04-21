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

#include "cutlass/cutlass.h"
#include "cutlass/cluster_launch.hpp"

#include "static_switch.h"
#include "flash.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"
#include "seq_len.h"
#include "utils.h"


template<int Arch, typename Kernel_traits, bool Is_causal, bool Is_target, bool Is_context, bool Is_delta_q, bool Is_local, bool Has_Rab, typename Seqlen_traits>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using Element = typename Kernel_traits::Element;
    using OutputType = typename Kernel_traits::OutputType;
    using TileShape_MNK = typename Kernel_traits::TileShape_MNK;
    using ClusterShape = typename Kernel_traits::ClusterShape_MNK;

    using CollectiveMainloop = flash::CollectiveMainloopFwd<Kernel_traits, Is_causal, Is_target, Is_context, Is_delta_q, Is_local, Has_Rab, Seqlen_traits>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<Kernel_traits, Seqlen_traits>;
    using Scheduler = std::conditional_t<
        Seqlen_traits::kUseVarSeqLen || Is_local, 
        flash::SingleTileScheduler,
        std::conditional_t<!Is_causal,
            flash::StaticPersistentTileScheduler,
            flash::DynamicPersistentTileScheduler<Kernel_traits::kNThreads - cutlass::NumThreadsPerWarpGroup, Kernel_traits::NumProducerThreads>
    >>;
    // using Scheduler = flash::SingleTileScheduler;
    Seqlen_traits seqlen_traits_q(
        params.total_q, params.seqlen_q, params.cu_seqlens_q, params.num_targets, params.num_contexts);
    Seqlen_traits seqlen_traits_k(
        params.total_k, params.seqlen_k, params.cu_seqlens_k, params.num_targets, params.num_contexts);
    typename CollectiveMainloop::Params mainloop_params =
        CollectiveMainloop::to_underlying_arguments({
            static_cast<Element const*>(params.q_ptr),
            seqlen_traits_q.get_gmem_layout(
                params.seqlen_q, params.d, params.h, params.b, 
                params.q_row_stride, params.q_head_stride, params.q_batch_stride
            ),  // layout_Q
            static_cast<OutputType const*>(params.rab_ptr),
            make_layout(make_shape(Is_delta_q ? params.seqlen_k : params.seqlen_q, params.seqlen_k, params.h_rab, params.b),
                        make_stride(params.rab_row_stride, _1{},
                                    params.rab_head_stride,
                                    params.rab_batch_stride)), // layout_Rab
            static_cast<Element const*>(params.k_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b, 
                params.k_row_stride, params.k_head_stride, params.k_batch_stride
            ),  // layout_K
            static_cast<Element const*>(params.v_ptr),
            seqlen_traits_k.get_gmem_layout(
                params.seqlen_k, params.d, params.h_k, params.b, 
                params.v_row_stride, params.v_head_stride, params.v_batch_stride
            ),  // layout_V
            params.descale_q_ptr,
            params.descale_k_ptr,
            params.descale_v_ptr,
            params.window_size_left,
            params.window_size_right,
            params.target_group_size,
            params.alpha
        });
    typename CollectiveEpilogue::Params epilogue_params =
        CollectiveEpilogue::to_underlying_arguments({
            static_cast<OutputType*>(params.o_ptr),
            seqlen_traits_q.get_gmem_layout(
                params.seqlen_q, params.d, params.h, params.b,
                params.o_row_stride, params.o_head_stride, params.o_batch_stride
            )  // layout_O
        });

    int num_blocks_m = cutlass::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    num_blocks_m = cutlass::ceil_div(num_blocks_m, size<0>(ClusterShape{})) * size<0>(ClusterShape{});
    typename Scheduler::Arguments scheduler_args = {num_blocks_m, params.h, params.b, params.tile_count_semaphore};
    typename Scheduler::Params scheduler_params = Scheduler::to_underlying_arguments(scheduler_args);

    // Get the ptr to kernel function.
    void *kernel;
    if constexpr(cutlass::sizeof_bits_v<Element> == 8)
        kernel = (void *)flash::compute_attn_ws_fp8<Kernel_traits, Is_causal, Is_local, Has_Rab, Scheduler, Seqlen_traits>;
    else
        kernel = (void *)flash::compute_attn_ws<Kernel_traits, Is_causal, Is_target, Is_context, Is_local, Has_Rab, Is_delta_q, Scheduler, Seqlen_traits>;
    int smem_size = sizeof(typename Kernel_traits::SharedStorage);
    // int smem_size_q = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_q));
    // int smem_size_k = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_k));
    // int smem_size_rab = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_rab));
    // int smem_size_v = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_v));
    // int smem_size_o = sizeof(decltype((typename Kernel_traits::SharedStorage{}).smem_o));
    // printf("smem_size = %d, q = %d, k = %d, rab = %d, v = %d, o = %d.\n", smem_size, smem_size_q, smem_size_k, smem_size_rab, smem_size_v, smem_size_o);
    if (smem_size >= 48 * 1024) {
       CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }

    int device;
    cudaGetDevice(&device);
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    dim3 grid_dims = Scheduler::get_grid_dim(scheduler_args, multiprocessor_count);
    static constexpr int ctaSize = Kernel_traits::kNWarps * 32;
    dim3 block_dims(ctaSize);
    dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
    cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
    cutlass::launch_kernel_on_cluster(
        launch_params, kernel, mainloop_params, epilogue_params, 
        scheduler_params, seqlen_traits_q, seqlen_traits_k);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<int Arch, typename T>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
                                run_flash_fwd<Arch,
                                    Flash_fwd_kernel_traits<Headdim, 192, 128, 16, 2, Has_Rab, false, 1, T>,
                                    Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, Has_Rab, Seqlen_traits
                                >(params, stream);
                            });
                        });
                    });
                });
            });
        });
    });
}

template<int Arch, typename T>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
                                run_flash_fwd<Arch,
                                    Flash_fwd_kernel_traits<Headdim, 128, 128, 12, 2, Has_Rab, false, 1, T>,
                                    Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, Has_Rab, Seqlen_traits
                                >(params, stream);
                            });
                        });
                    });
                });
            });
        });
    });
}

template<int Arch, typename T>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
                                run_flash_fwd<Arch,
                                    Flash_fwd_kernel_traits<Headdim, 128, 64, 12, 2, Has_Rab, false, 1, T>,
                                    Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, Has_Rab, Seqlen_traits
                                >(params, stream);
                            });
                        });
                    });
                });
            });
        });
    });
}

template<int Arch, typename T>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    DELTA_Q_SWITCH(params.is_delta_q, Is_delta_q, [&] {
        CONTEXT_SWITCH(params.is_context, Is_context, [&] {
            TARGET_SWITCH(params.is_target, Is_target, [&] {
                CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
                    LOCAL_SWITCH(params.is_local, Is_local, [&] {
                        RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                            SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
                                run_flash_fwd<Arch,
                                    Flash_fwd_kernel_traits<Headdim, 128, 64, 12, 2, Has_Rab, false, 1, T>,
                                    Is_causal, Is_target, Is_context, Is_delta_q, Is_local && !Is_causal, Has_Rab, Seqlen_traits
                                >(params, stream);
                            });
                        });
                    });
                });
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim64_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    constexpr static int kBlockM = 192;
    constexpr static int kBlockN = 128;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 2;
    CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
        LOCAL_SWITCH(params.is_local, Is_local, [&] {
            RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
                    run_flash_fwd<90,
                        Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages, Has_Rab, false, 1>,
                        Is_causal, false, false, false, Is_local && !Is_causal, Has_Rab, Seqlen_traits
                    >(params, stream);
                });
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim128_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 2;
    CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
        LOCAL_SWITCH(params.is_local, Is_local, [&] {
            RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
                    run_flash_fwd<90,
                        Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages, Has_Rab, false, 1>,
                        Is_causal, false, false, false, Is_local && !Is_causal, Has_Rab, Seqlen_traits
                    >(params, stream);
                });
            });
        });
    });
}

template<typename T>
void run_mha_fwd_hdim256_fp8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256; 
    constexpr static int kBlockM = 128;
    constexpr static int kBlockN = 64;
    constexpr static int kNWarps = 4 + kBlockM/16;
    constexpr static int kStages = 2;
    CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
        LOCAL_SWITCH(params.is_local, Is_local, [&] {
            RAB_SWITCH(params.has_rab, Has_Rab, [&] {
                SEQLEN_SWITCH(params.cu_seqlens_q, Seqlen_traits, [&] {
                    run_flash_fwd<90,
                        Flash_fwd_kernel_traits_fp8<Headdim, kBlockM, kBlockN, kNWarps, kStages, Has_Rab, false, 1>,
                        Is_causal, false, false, false, Is_local && !Is_causal, Has_Rab, Seqlen_traits
                    >(params, stream);
                });
            });
        });
    });
}
