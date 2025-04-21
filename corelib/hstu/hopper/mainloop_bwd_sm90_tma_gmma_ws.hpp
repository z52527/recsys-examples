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
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/barrier.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

template <bool A, class Mma, class Tensor0>
CUTLASS_DEVICE
auto mma_partition_fragment_AB(Mma const& mma, Tensor0 const& tensor0) {
    if constexpr (A) {
        return mma.partition_fragment_A(tensor0);
    } else {
        return mma.partition_fragment_B(tensor0);
    }
}

using namespace cute;

template <int Stages, int Stages_dO, int Stages_dS, class ClusterShape_, class TileShape_MNK_, class Element_, class ElementAccum_, class ArchTag_,
        bool Has_Rab_, bool Has_dRab_, bool Is_causal_, bool Is_target_, bool Is_context_, bool Is_delta_q_, bool Is_local_, bool Deterministic,
        bool SdP_swapAB_, bool dKV_swapAB_, bool dQ_swapAB_,
        int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1>
struct CollectiveMainloopBwd {

    static constexpr int kStages = Stages;
    static constexpr int kStages_dO = Stages_dO;
    static constexpr int kStages_dS = Stages_dS;
    static_assert(kStages >= kStages_dO);
    static_assert(Stages_dS == kStages);
    using ClusterShape = ClusterShape_;
    using TileShape_MNK = TileShape_MNK_;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Has_Rab = Has_Rab_;
    static constexpr bool Has_dRab = Has_dRab_;
    static constexpr bool Is_causal = Is_causal_;
    static constexpr bool Is_delta_q = Is_delta_q_;
    static constexpr bool Is_target = Is_target_;
    static constexpr bool Is_context = Is_context_;
    static constexpr bool Is_local = Is_local_;
    static constexpr bool SdP_swapAB = SdP_swapAB_;
    static constexpr bool dKV_swapAB = dKV_swapAB_;
    static constexpr bool dQ_swapAB = dQ_swapAB_;

    static constexpr bool Q_dO_same_stages = kStages == kStages_dO;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    static constexpr int NumdQWarpGroups = NumMmaWarpGroups;
    static constexpr int kNThreadsdQ = NumdQWarpGroups * cutlass::NumThreadsPerWarpGroup;

    static_assert(ArchTag::kMinComputeCapability >= 90);
    static_assert(get<0>(ClusterShape{}) == 1 && get<2>(ClusterShape{}) == 1);

    static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;

    static_assert(NumMmaWarpGroups % AtomLayoutMSdP == 0);
    static_assert(NumMmaWarpGroups % AtomLayoutNdKV == 0);
    static_assert(NumdQWarpGroups % AtomLayoutMdQ == 0);
    static constexpr bool Mma_dKV_is_RS = AtomLayoutMSdP == 1 && AtomLayoutNdKV == NumMmaWarpGroups && SdP_swapAB && !dKV_swapAB;
    static constexpr bool Mma_dQ_is_RS = AtomLayoutMSdP == NumMmaWarpGroups && AtomLayoutMdQ == NumMmaWarpGroups && !SdP_swapAB && !dQ_swapAB;  // If dQ_swapAB we can't use RS
    
    static constexpr GMMA::Major PdS_Major = GMMA::Major::K;
    // static constexpr GMMA::Major PdS_Major = GMMA::Major::MN;
    static constexpr GMMA::Major PdSt_Major = PdS_Major == GMMA::Major::K ? GMMA::Major::MN : GMMA::Major::K;

    using TileShapeAtomSdP = std::conditional_t<
        !SdP_swapAB,
        Shape<Int<kBlockM>, Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>, Int<kHeadDim>>,
        Shape<Int<kBlockN>, Int<kBlockM / AtomLayoutMSdP>, Int<kHeadDim>>
    >;
    using AtomLayoutSdP = std::conditional_t<
        !SdP_swapAB,
        Layout<Shape<Int<AtomLayoutMSdP>, Int<NumMmaWarpGroups / AtomLayoutMSdP>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutMSdP>, Int<AtomLayoutMSdP>, _1>>
    >;
    using TiledMmaSdP = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomSdP>(),
        AtomLayoutSdP{}));

    using TileShapeAtomdKV = std::conditional_t<
        !dKV_swapAB,
        Shape<Int<kBlockN>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>, Int<kBlockM>>,
        Shape<Int<kHeadDim>, Int<kBlockN / AtomLayoutNdKV>, Int<kBlockM>>
    >;
    using AtomLayoutdKV = std::conditional_t<
        !dKV_swapAB,
        Layout<Shape<Int<AtomLayoutNdKV>, Int<NumMmaWarpGroups / AtomLayoutNdKV>, _1>>,
        Layout<Shape<Int<NumMmaWarpGroups / AtomLayoutNdKV>, Int<AtomLayoutNdKV>, _1>>
    >;
    using TiledMmadKV = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Mma_dKV_is_RS,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, GMMA::Major::K, GMMA::Major::MN>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdKV, !dKV_swapAB ? PdSt_Major : GMMA::Major::MN, !dKV_swapAB ? GMMA::Major::MN : PdSt_Major>())
        >{},
        AtomLayoutdKV{}));

    using TileShapeAtomdQ = std::conditional_t<
        !dQ_swapAB,
        Shape<Int<kBlockM>, Int<kHeadDim / (NumdQWarpGroups / AtomLayoutMdQ)>, Int<kBlockN>>,
        Shape<Int<kHeadDim>, Int<kBlockM / AtomLayoutMdQ>, Int<kBlockN>>
    >;
    using AtomLayoutdQ = std::conditional_t<
        !dQ_swapAB,
        Layout<Shape<Int<AtomLayoutMdQ>, Int<NumdQWarpGroups / AtomLayoutMdQ>, _1>>,
        Layout<Shape<Int<NumdQWarpGroups / AtomLayoutMdQ>, Int<AtomLayoutMdQ>, _1>>
    >;
    using TiledMmadQ = decltype(cute::make_tiled_mma(
        std::conditional_t<
            Mma_dQ_is_RS,
            decltype(cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, GMMA::Major::K, GMMA::Major::MN>()),
            decltype(cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShapeAtomdQ, !dQ_swapAB ? PdS_Major : GMMA::Major::MN, !dQ_swapAB ? GMMA::Major::MN : PdS_Major>())
        >{},
        AtomLayoutdQ{}));

    using SmemLayoutAtomQdO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockM>, Int<kHeadDim / (NumMmaWarpGroups / AtomLayoutNdKV)>>());
    using SmemLayoutQ =
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutdO = 
        decltype(tile_to_shape(SmemLayoutAtomQdO{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages_dO>{})));

    using SmemLayoutAtomRab = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                    decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<1>(TileShape_MNK{}))>());
    using SmemLayoutRab =
        decltype(tile_to_shape(SmemLayoutAtomRab{},
                 make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutRabt =
        decltype(cute::composition(SmemLayoutRab{},
                                   make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
                                     Int<kBlockN>, Int<kHeadDim / (NumdQWarpGroups / AtomLayoutMdQ)>>());
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtomK{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element,
        decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtomV{}, select<1, 2>(TileShape_MNK{})));

    using SmemLayoutAtomPdS = decltype(cutlass::gemm::collective::detail::ss_smem_selector<PdS_Major, Element,
        Int<kBlockM / AtomLayoutMSdP>,
        Int<kBlockN / (NumMmaWarpGroups / AtomLayoutMSdP)>>());
    using SmemLayoutPdS = decltype(tile_to_shape(
        SmemLayoutAtomPdS{},
        make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kStages_dS>{}),
        std::conditional_t<PdS_Major == GMMA::Major::K, cute::Step<_1, _2, _3>, cute::Step<_2, _1, _3>>{}));

    // Note this is the transpose in terms of the view, not in terms of memory.
    using SmemLayoutQt =
        decltype(cute::composition(SmemLayoutQ{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutdOt =
        decltype(cute::composition(SmemLayoutdO{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<0>(TileShape_MNK{}), Int<kStages_dO>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kHeadDim>{}))));
    using SmemLayoutKt =
        decltype(cute::composition(SmemLayoutK{},
                                   make_layout(make_shape(get<2>(TileShape_MNK{}), get<1>(TileShape_MNK{})),
                                               make_stride(Int<kBlockN>{}, _1{}))));
    using SmemLayoutPdSt =
        decltype(cute::composition(SmemLayoutPdS{},
                                   make_layout(make_shape(Int<kBlockN>{}, Int<kBlockM>{}, Int<kStages_dS>{}),
                                               make_stride(Int<kBlockM>{}, _1{}, Int<kBlockM * kBlockN>{}))));

    // Thread layout, 256 or 384 threads per row
    using R2SLayoutAtomdQaccum = Layout<Shape<Int<kNThreadsdQ>>, Stride<_1>>;
    using R2STiledCopydQaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{}, R2SLayoutAtomdQaccum{},
                                                         Layout<Shape < _4>>{}));  // Val layout, 4 vals per store
    using SmemLayoutdQaccum = Layout<Shape<Int<kBlockM * kHeadDim>>, Stride<_1>>;
    using SmemLayoutAtomdQaccumTMA =
        decltype(composition(Swizzle<0, 4, 3>{},  // We don't want any swizzle
                             Layout<Shape<Int<8>, Int<kHeadDim>>,
                             Stride<Int<kHeadDim>, _1>>{}));
    using SmemLayoutdQaccumTMA =
        decltype(tile_to_shape(SmemLayoutAtomdQaccumTMA{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutdQaccumTMANoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutdQaccumTMA{}));

    static constexpr int kNumPdSStore = kBlockM * kBlockN / NumMmaThreads;
    // If !SdP_swapAB, the accum registers hold P / dS, otherwise they hold Pt / dSt.
    // If PdS_major is MN, then we need to "transpose" the write.
    using SmemCopyAtomPdS = Copy_Atom<
        std::conditional_t<(!SdP_swapAB) ^ (PdS_Major == GMMA::Major::MN),
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U32x4_STSM_N, cute::SM90_U32x2_STSM_N>,
            std::conditional_t<kNumPdSStore % 8 == 0, cute::SM90_U16x8_STSM_T, cute::SM90_U16x4_STSM_T>
        >,
            Element
    >;
    using SmemCopyAtomRab = Copy_Atom<std::conditional_t<!SdP_swapAB, cute::SM75_U32x4_LDSM_N, cute::SM75_U16x8_LDSM_T>, Element>;

    static constexpr bool dQacc_use_TMA = kHeadDim < 256;
    // These are for the case where we don't use TMA to do atomicAdd on dQaccum, but just use direct atomicAdd.
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = cutlass::gcd(kHeadDim / kGmemElemsPerLoad, int(kNThreadsdQ));
    using GmemLayoutAtomdQaccum = Layout<Shape <Int<kNThreadsdQ / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                         Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopydQaccumAtomic = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomdQaccum{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 4 vals per store

    using GmemTiledCopyQdO = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(shape<1>(ClusterShape{})));
    using GmemTiledCopyRab = cute::SM90_TMA_LOAD;
    using GmemTiledCopyKV = cute::SM90_TMA_LOAD;
    using GmemTiledCopydQaccum = cute::SM90_TMA_REDUCE_ADD;
    using GmemTiledCopydRab = cute::SM90_TMA_REDUCE_ADD;

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQKV = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using ShapeRab = ShapeQKV;
    using StrideRab = StrideQKV;

    using TMA_QdO = decltype(make_tma_copy(
        GmemTiledCopyQdO{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        take<0, 2>(SmemLayoutQ{}),
        select<0, 2>(TileShape_MNK{}),
        size<1>(ClusterShape{}))); // mcast along N mode for this M load, if any
    
    using TMA_Rab = decltype(make_tma_copy(
        GmemTiledCopyRab{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeRab{}, StrideRab{}),
        take<0, 2>(SmemLayoutRab{}),
        select<0, 1>(TileShape_MNK{}),
         _1{})); // no mcast for Rab

    using TMA_K = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutK{},
        select<1, 2>(TileShape_MNK{}),
        _1{})); // no mcast for KV

    using TMA_V = decltype(make_tma_copy(
        GmemTiledCopyKV{},
        make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutV{},
        select<1, 2>(TileShape_MNK{}),
        _1{})); // no mcast for KV

    using TMA_add_dQ = decltype(make_tma_copy(
        GmemTiledCopydQaccum{},
        make_tensor(make_gmem_ptr(static_cast<ElementAccum*>(nullptr)), ShapeQKV{}, StrideQKV{}),
        SmemLayoutdQaccumTMA{},
        select<0, 2>(TileShape_MNK{}),
        _1{})); // no mcast for dQ

    using TMA_store_dRab = decltype(make_tma_copy(
        GmemTiledCopydRab{},
        make_tensor(make_gmem_ptr(static_cast<Element*>(nullptr)), ShapeRab{}, StrideRab{}),
        take<0, 2>(SmemLayoutPdS{}),
        select<0, 1>(TileShape_MNK{}),
        _1{})); // no mcast for dRab


    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using MainloopPipeline_dO = typename cutlass::PipelineTmaAsync<kStages_dO>;
    using PipelineState_dO = typename MainloopPipeline_dO::PipelineState;
    using MainloopPipeline_dRab = cutlass::PipelineAsync<kStages_dS>;
    using PipelineState_dRab = typename MainloopPipeline_dRab::PipelineState;

    // Set the bytes transferred in this TMA transaction (may involve multiple issues)
    static constexpr uint32_t TmaTransactionBytesQ = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutQ{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesRab = static_cast<uint32_t>(size(take<0, 2>(SmemLayoutRab{})) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesK = static_cast<uint32_t>(size(SmemLayoutK{}) * cutlass::sizeof_bits_v<Element> / 8);
    static constexpr uint32_t TmaTransactionBytesV = static_cast<uint32_t>(size(SmemLayoutV{}) * cutlass::sizeof_bits_v<Element> / 8);

    static constexpr size_t SmemAlignmentP = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    static constexpr size_t SmemAlignmentdS = cutlass::detail::alignment_for_swizzle(SmemLayoutPdS{});
    // Without this SmemAlignment, with hdim 256 we get "misaligned address" error in TMA
    static constexpr size_t SmemAlignmentQKVdO = kHeadDim % 256 == 0 ? 256 : 128;
    using SmemRabStorage = cute::conditional_t<
        Has_Rab,
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutRab>, SmemAlignmentQKVdO>,
        cute::array_aligned<Element, 128/sizeof(Element)>>;
    static_assert(SmemAlignmentP >= 128 && SmemAlignmentdS >= 128, "Require at least 128B alignment");
    struct TensorStorage : cute::aligned_struct<cute::max(SmemAlignmentP, SmemAlignmentdS, SmemAlignmentQKVdO)> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>, SmemAlignmentQKVdO> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>, SmemAlignmentQKVdO> smem_v;
        cute::array_aligned<ElementAccum, dQacc_use_TMA ? cute::cosize_v<SmemLayoutdQaccum> : 0> smem_dqacc;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>, SmemAlignmentQKVdO> smem_q;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>, SmemAlignmentQKVdO> smem_do;
        cute::array_aligned<Element, Mma_dKV_is_RS ? 0 : cute::cosize_v<SmemLayoutPdS>, SmemAlignmentP> smem_p;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutPdS>, SmemAlignmentdS> smem_ds;
        SmemRabStorage smem_rab;
    };

    // These are tuned for speed. They don't affect correctness.
    // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
    // this helps quite a bit to not have to do causal masking for most of the iterations.
    // For hdim 192, separating masking iterations results in register spills.
    static constexpr bool SeparateMaskingIterations = kHeadDim <= 64;
    // For hdim256, we want to slice the dQ MMA (64 x 256 on 2 WGs) into two (64 x 128 on 2 WGs) so that we can
    // do atomic add on one half before doing the other half of the MMA, to reduce register pressure.
    static constexpr bool Slice_dQKV_Mma = kHeadDim == 256 && dQ_swapAB && AtomLayoutMdQ == 1 && NumMmaWarpGroups == 2;
    static_assert(!(Deterministic && Slice_dQKV_Mma), "Deterministic mode not supported with Slice_dQKV_Mma");

    // Host side kernel arguments
    struct Arguments {
        Element const* ptr_Q;
        ShapeQKV const shape_Q;
        StrideQKV const stride_Q;
        Element const* ptr_Rab;
        ShapeQKV const shape_Rab;
        StrideQKV const stride_Rab;
        Element const* ptr_K;
        ShapeQKV const shape_K;
        StrideQKV const stride_K;
        Element const* ptr_V;
        StrideQKV const stride_V;
        Element const* ptr_dO;
        StrideQKV const stride_dO;
        ElementAccum* ptr_dQaccum;
        ShapeQKV const shape_dQaccum;
        StrideQKV const stride_dQaccum;
        Element * ptr_dRab;
        ShapeQKV const shape_dRab;
        StrideQKV const stride_dRab;
        int const seqlen_q;
        int const seqlen_k;
        int const seqlen_i;
        int const window_size_left, window_size_right;
        int const num_batch;
        float const alpha;
        int* dq_semaphore;
        int const* cu_seqlens_q = nullptr;
        int const* cu_seqlens_k = nullptr;
        int const* num_targets = nullptr;
    };

    // Device side kernel params
    struct Params {
        ShapeQKV const shape_Q;
        ShapeQKV const shape_K;
        ShapeQKV const shape_Rab;
        ShapeQKV const shape_dQaccum;
        ShapeQKV const shape_dRab;
        ElementAccum* ptr_dQaccum;
        StrideQKV stride_dQaccum;
        Element * ptr_dRab;
        StrideQKV const stride_dRab;
        cutlass::FastDivmod qhead_per_khead_divmod;
        cutlass::FastDivmod qhead_per_rabhead_divmod;
        TMA_QdO tma_load_Q, tma_load_dO;
        TMA_Rab tma_load_Rab;        
        TMA_K tma_load_K;
        TMA_V tma_load_V;
        TMA_add_dQ tma_add_dQ;
        TMA_store_dRab tma_store_dRab;
        int const seqlen_q;
        int const seqlen_k;
        int const seqlen_i;
        int const window_size_left, window_size_right;
        int const num_batch;
        float const alpha;
        int* dq_semaphore;
        int const* cu_seqlens_q = nullptr;
        int const* cu_seqlens_k = nullptr;
        int const* num_targets = nullptr;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
        TMA_QdO tma_load_Q = make_tma_copy(
            GmemTiledCopyQdO{},
            mQ,
            SmemLayoutQ{}(_, _, _0{}),
            select<0, 2>(TileShape_MNK{}),
            size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
        Tensor mRab = make_tensor(make_gmem_ptr(args.ptr_Rab), args.shape_Rab, args.stride_Rab);
        TMA_Rab tma_load_Rab = make_tma_copy(
            GmemTiledCopyRab{},
            mRab,
            SmemLayoutRab{}(_, _, _0{}),
            select<0, 1>(TileShape_MNK{}),
            _1{}); // no mcast for Rab
        Tensor mdO = make_tensor(make_gmem_ptr(args.ptr_dO), args.shape_Q, args.stride_dO);
        TMA_QdO tma_load_dO = make_tma_copy(
            GmemTiledCopyQdO{},
            mdO,
            SmemLayoutdO{}(_, _, _0{}),
            select<0, 2>(TileShape_MNK{}),
            size<1>(ClusterShape{})); // mcast along N mode for this M load, if any
        Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
        TMA_K tma_load_K = make_tma_copy(
            GmemTiledCopyKV{},
            mK,
            SmemLayoutK{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for KV
        Tensor mV = make_tensor(make_gmem_ptr(args.ptr_V), args.shape_K, args.stride_V);
        TMA_V tma_load_V = make_tma_copy(
            GmemTiledCopyKV{},
            mV,
            SmemLayoutV{},
            select<1, 2>(TileShape_MNK{}),
            _1{}); // no mcast for KV
        Tensor mdQaccum = make_tensor(make_gmem_ptr(args.ptr_dQaccum), args.shape_dQaccum, args.stride_dQaccum);
        TMA_add_dQ tma_add_dQ = make_tma_copy(
            GmemTiledCopydQaccum{},
            mdQaccum,
            SmemLayoutdQaccumTMA{},
            select<0, 2>(TileShape_MNK{}),
            _1{}); // no mcast for dQaccum
        Tensor mdRab = make_tensor(make_gmem_ptr(args.ptr_dRab), args.shape_dRab, args.stride_dRab);
        TMA_store_dRab tma_store_dRab = make_tma_copy(
            GmemTiledCopydRab{},
            mdRab,
            SmemLayoutPdS{}(_, _, _0{}),
            select<0, 1>(TileShape_MNK{}),
            _1{}); // no mcast for dRab
        if constexpr (Deterministic) { assert(args.dq_semaphore != nullptr); }
        assert(args.cu_seqlens_q != nullptr);
        assert(args.cu_seqlens_k != nullptr);
        return {args.shape_Q, args.shape_K, args.shape_Rab, args.shape_dQaccum, args.shape_dRab,
                args.ptr_dQaccum, args.stride_dQaccum, args.ptr_dRab, args.stride_dRab,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_Rab))),
                tma_load_Q, tma_load_dO, tma_load_Rab, tma_load_K, tma_load_V, tma_add_dQ, tma_store_dRab,
                args.seqlen_q, args.seqlen_k, args.seqlen_i,
                args.window_size_left, args.window_size_right,
                args.num_batch, args.alpha, args.dq_semaphore,
                args.cu_seqlens_q, args.cu_seqlens_k, args.num_targets};
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& params) {
        cute::prefetch_tma_descriptor(params.tma_load_Q.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_dO.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_K.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_load_V.get_tma_descriptor());
        cute::prefetch_tma_descriptor(params.tma_add_dQ.get_tma_descriptor());
        if (Has_Rab){
            cute::prefetch_tma_descriptor(params.tma_load_Rab.get_tma_descriptor());
        }
    }

    CUTLASS_DEVICE
    int get_seqlen_q(Params const& params, int bidb) {
        return params.cu_seqlens_q == nullptr
            ? get<0>(params.shape_Q)
            : params.cu_seqlens_q[bidb + 1] - params.cu_seqlens_q[bidb];
    }

    CUTLASS_DEVICE
    int get_seqlen_k(Params const& params, int bidb) {
        return params.cu_seqlens_k == nullptr
            ? get<0>(params.shape_K)
            : params.cu_seqlens_k[bidb + 1] - params.cu_seqlens_k[bidb];
    }

    CUTLASS_DEVICE
    int get_seqlen_t(Params const& params, int bidb) {
        if constexpr (!Is_target) {
            return 0;
        } else {
            return params.num_targets[bidb];
        }
    }

    CUTLASS_DEVICE
    int get_target_mask(Params const& params, int n_block, int bidb) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        static constexpr int Mix = 1;
        static constexpr int Target = 2;
        // We found that this severely affects performance, but we cannot identify the cause
        // if constexpr (!Is_target) {
        //     return 0;
        // } 
        // int seqlen_i = get_seqlen_q(params, bidb) - get_seqlen_t(params, bidb);
        int seqlen_i = get_seqlen_k(params, bidb) - get_seqlen_t(params, bidb); // for delta_q
        if ((n_block + 1) * kBlockN <= seqlen_i) {
            return 0;
        } else if (n_block * kBlockN < seqlen_i) {
            return Mix; 
        } else {
            return Target; 
        }
    }

    CUTLASS_DEVICE
    int get_m_block_min(Params const& params, int n_block, int bidb) {  
        if constexpr (Is_causal || Is_local) {
            static constexpr int kBlockM = get<0>(TileShape_MNK{});
            static constexpr int kBlockN = get<1>(TileShape_MNK{});
            int const seqlen_q = get_seqlen_q(params, bidb);
            int const seqlen_k = get_seqlen_k(params, bidb);
            return std::max(0, (n_block * kBlockN + seqlen_q - seqlen_k - params.window_size_right) / kBlockM);
        } else {
            return 0;
        }
    }

    CUTLASS_DEVICE
    int get_m_block_max(Params const& params, int n_block, int bidb) {
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});
        int const seqlen_q = get_seqlen_q(params, bidb);
        int const seqlen_k = get_seqlen_k(params, bidb);
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        if constexpr (Is_local) {
            int const seqlen_k = get_seqlen_k(params, bidb);
            m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + params.window_size_left, kBlockM));
        }
        if constexpr (Is_target) {
            static constexpr int Target = 2;
            int target_mask = get_target_mask(params, n_block, bidb); 
            if (target_mask == Target) {
                static constexpr int n_masking_steps = cute::ceil_div(kBlockN, kBlockM);
                int const m_block_min = get_m_block_min(params, n_block, bidb);
                m_block_max = std::min(m_block_max, m_block_min + n_masking_steps);
            }
        }
        return m_block_max;
    }

    template <typename SchedulerPrefetch, typename SharedStorage>
    CUTLASS_DEVICE void
    load(Params const& params,
         MainloopPipeline pipeline_q,
         MainloopPipeline pipeline_rab,
         MainloopPipeline_dO pipeline_do,
         PipelineState& smem_pipe_write,
         PipelineState_dO& smem_pipe_write_do,
         SharedStorage &shared_storage,
         SchedulerPrefetch const& scheduler_prefetch,
         cute::tuple<int32_t, int32_t, int32_t> block_coord
         ) {

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sRab = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_rab.data()), SmemLayoutRab{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_v.data()), SmemLayoutV{});

        auto [n_block, bidh, bidb] = block_coord;
        int bidh_kv = params.qhead_per_khead_divmod.divide(bidh);
        int bidh_rab = params.qhead_per_rabhead_divmod.divide(bidh);

        // Prepare the TMA loads
        uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
        constexpr uint32_t cluster_shape_x = get<0>(ClusterShape());
        uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};
        Tensor mQ = params.tma_load_Q.get_tma_tensor(params.shape_Q)(_, _, bidh, 0);
        Tensor mRab = params.tma_load_Rab.get_tma_tensor(params.shape_Rab)(_, _, bidh_rab, bidb);
        Tensor mdO = params.tma_load_dO.get_tma_tensor(params.shape_Q)(_, _, bidh, 0);
        Tensor mK = params.tma_load_K.get_tma_tensor(params.shape_K)(_, _, bidh_kv, 0);
        Tensor mV = params.tma_load_V.get_tma_tensor(params.shape_K)(_, _, bidh_kv, 0);

        int const offset_q = params.cu_seqlens_q[bidb];
        int const offset_k = params.cu_seqlens_k[bidb];
        Tensor gQ = local_tile(domain_offset(make_coord(offset_q, _0{}), mQ), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        int offset_rab = Is_delta_q ? get_seqlen_k(params, bidb) - get_seqlen_q(params, bidb) : 0;
        Tensor gRab = local_tile(domain_offset(make_coord(offset_rab, _0{}), mRab), select<0, 1>(TileShape_MNK{}),
                make_coord(_, n_block)); 

        Tensor gdO = local_tile(domain_offset(make_coord(offset_q, _0{}), mdO), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        Tensor gK = local_tile(domain_offset(make_coord(offset_k, _0{}), mK), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)
        Tensor gV = local_tile(domain_offset(make_coord(offset_k, _0{}), mV), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));  // (N, K)

        Tensor sK_x = make_tensor(sK.data(), make_layout(sK.layout(), Layout<_1>{}));
        Tensor gK_x = make_tensor(gK.data(), make_layout(gK.layout(), Layout<_1>{}));
        Tensor sV_x = make_tensor(sV.data(), make_layout(sV.layout(), Layout<_1>{}));
        Tensor gV_x = make_tensor(gV.data(), make_layout(gV.layout(), Layout<_1>{}));

        auto block_tma_Q = params.tma_load_Q.get_slice(cluster_local_block_id.y);
        auto block_tma_Rab = params.tma_load_Rab.get_slice(cluster_local_block_id.y);
        auto block_tma_dO = params.tma_load_dO.get_slice(cluster_local_block_id.y);
        Tensor tQgQ = group_modes<0, 3>(block_tma_Q.partition_S(gQ));
        Tensor tQsQ = group_modes<0, 3>(block_tma_Q.partition_D(sQ));
        Tensor tRabgRab = group_modes<0, 3>(block_tma_Rab.partition_S(gRab));
        Tensor tRabsRab = group_modes<0, 3>(block_tma_Rab.partition_D(sRab));
        Tensor tdOgdO = group_modes<0, 3>(block_tma_dO.partition_S(gdO));
        Tensor tdOsdO = group_modes<0, 3>(block_tma_dO.partition_D(sdO));
        auto [tKgK, tKsK] = tma_partition(params.tma_load_K, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sK_x), group_modes<0, 2>(gK_x));  // (TMA), (TMA)
        auto [tVgV, tVsV] = tma_partition(params.tma_load_V, _0{}, Layout<_1>{},
                                          group_modes<0, 2>(sV_x), group_modes<0, 2>(gV_x));  // (TMA), (TMA)

        uint16_t mcast_mask_qdo = 0;
        if constexpr (cute::is_same_v<GmemTiledCopyQdO, SM90_TMA_LOAD_MULTICAST>) {
            auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
            for (int n = 0; n < size<1>(block_layout); ++n) {
                mcast_mask_qdo |= (uint16_t(1) << block_layout(cluster_local_block_id.x, n, _0{}));
            }
        }

        int m_block_max = get_m_block_max(params, n_block, bidb);
        int m_block_min = get_m_block_min(params, n_block, bidb);
        int m_block = m_block_min;

        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            pipeline_q.producer_acquire(smem_pipe_write);
            copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
                 tQgQ(_, m_block), tQsQ(_, smem_pipe_write.index()));
            if (Has_Rab){
                pipeline_rab.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block), tRabsRab(_, smem_pipe_write.index()));
            }
        }

        // // Wait for the MMA warpgroups to say that smem_k and smem_v are ready
        // cutlass::arch::NamedBarrier::sync(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);

        if (lane_predicate) {
            // Copy K tile and V tile from GMEM to SMEM.
            shared_storage.barrier_KV.arrive_and_expect_tx(TmaTransactionBytesK + TmaTransactionBytesV);
            copy(params.tma_load_K.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tKgK, tKsK);
            copy(params.tma_load_V.with(reinterpret_cast<cutlass::arch::ClusterTransactionBarrier::ValueType&>(shared_storage.barrier_KV), 0 /*mcast_mask*/), tVgV, tVsV);

            #pragma unroll (kHeadDim < 256 ? 2 : 1)
            for (; m_block < m_block_max - 1; ++m_block) {
                // If Q and dO have the same number of stages, we can use the same pipeline state variable
                // to reduce registers
                PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
                pipeline_do.producer_acquire(smem_pipe_write_do_cur);
                copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
                  tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
                if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
                ++smem_pipe_write;
                pipeline_q.producer_acquire(smem_pipe_write);
                copy(params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write), mcast_mask_qdo),
                    tQgQ(_, m_block + 1), tQsQ(_, smem_pipe_write.index()));
                if (Has_Rab){
                    pipeline_rab.producer_acquire(smem_pipe_write);
                    copy(params.tma_load_Rab.with(*pipeline_rab.producer_get_barrier(smem_pipe_write), 0), tRabgRab(_, m_block + 1), tRabsRab(_, smem_pipe_write.index()));
                }
            }
        }
        scheduler_prefetch();
        if (lane_predicate) {
            PipelineState_dO smem_pipe_write_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_write, smem_pipe_write_do);
            pipeline_do.producer_acquire(smem_pipe_write_do_cur);
            copy(params.tma_load_dO.with(*pipeline_do.producer_get_barrier(smem_pipe_write_do_cur), mcast_mask_qdo),
                tdOgdO(_, m_block), tdOsdO(_, smem_pipe_write_do_cur.index()));
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_write_do; }
            ++smem_pipe_write;
        }
        if constexpr (Q_dO_same_stages) { smem_pipe_write_do = smem_pipe_write; }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline pipeline_rab, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write) {
        // Need to copy since pipeline_q.producer_tail(smem_pipe_write) will increment smem_pipe_write
        PipelineState smem_pipe_write_do = smem_pipe_write;
        PipelineState smem_pipe_write_rab = smem_pipe_write;
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (lane_predicate) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
            if (Has_Rab){
                pipeline_rab.producer_tail(smem_pipe_write_rab);
            }
        }
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void
    load_tail(MainloopPipeline pipeline_q, MainloopPipeline pipeline_rab, MainloopPipeline_dO pipeline_do,
              PipelineState& smem_pipe_write, PipelineState_dO& smem_pipe_write_do) {
        PipelineState smem_pipe_write_rab = smem_pipe_write;
        int lane_predicate = cute::elect_one_sync();
        // Issue the epilogue waits
        if (lane_predicate) {
            /* This helps avoid early exit of blocks in Cluster
            * Waits for all stages to either be released (all Consumer UNLOCKs), or if the stage was never used
            * then would just be acquired since the phase was still inverted from make_producer_start_state
            */
            pipeline_q.producer_tail(smem_pipe_write);
            pipeline_do.producer_tail(smem_pipe_write_do);
            if (Has_Rab){
                pipeline_rab.producer_tail(smem_pipe_write_rab);
            }
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_dq(Params const& params,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t> block_coord
             ) {
        if constexpr (!dQacc_use_TMA) { return; }

        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMA{});
        Tensor sdQnoswizzle = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dqacc.data()), SmemLayoutdQaccumTMANoSwizzle{});
        auto [n_block, bidh, bidb] = block_coord;

        int const offset_padded = (params.cu_seqlens_q[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
        // Prepare the TMA loads
        Tensor mdQaccum = params.tma_add_dQ.get_tma_tensor(params.shape_dQaccum)(_, _, bidh, 0);
        Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdQaccum), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)
        auto block_tma_dQ = params.tma_add_dQ.get_slice(_0{});
        Tensor tdQgdQ = block_tma_dQ.partition_D(gdQaccum);  // (TMA, TMA_M, TMA_K)
        Tensor tdQsdQ = block_tma_dQ.partition_S(sdQ); // (TMA, TMA_M, TMA_K)

        int m_block_max = get_m_block_max(params, n_block, bidb);
        int m_block_min = get_m_block_min(params, n_block, bidb);
        int m_block = m_block_min;
        int const num_batch = params.num_batch;
        int const num_head = get<2>(params.shape_Q);
        int *lock_ptr = !Deterministic ? nullptr : params.dq_semaphore + bidb * num_head + bidh;
        using Barrier = cutlass::GenericBarrier<cutlass::detail::SyncwarpSync>;
        int lane_predicate = cute::elect_one_sync();
        #pragma unroll 2
        for (; m_block < m_block_max; ++m_block) {
            if constexpr (Deterministic) {
                Barrier::wait_eq(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head, n_block);
            }
            cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull) /*id*/);  // sdQ full, to be written to gmem
            if (lane_predicate) {
                cute::copy(params.tma_add_dQ, tdQsdQ, tdQgdQ(_, _, _, m_block));
                tma_store_arrive();
            }
            tma_store_wait<0>();
            if constexpr (Deterministic) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
            }
            cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
        }
        if constexpr (Is_local && Deterministic) {
            constexpr int kBlockM = get<0>(TileShape_MNK{});        
            int const seqlen_q = get_seqlen_q(params, bidb);
            int const m_block_global_max = cute::ceil_div(seqlen_q, kBlockM);
            #pragma unroll 2
            for (; m_block < m_block_global_max; ++m_block) {
                Barrier::arrive_inc(lock_ptr, threadIdx.x % cutlass::NumThreadsPerWarp, m_block * num_batch * num_head);
            }
        }
    }

    template <typename SharedStorage>
    CUTLASS_DEVICE void
    store_drab(Params const& params,
             MainloopPipeline_dRab pipeline_drab,
             PipelineState_dRab& smem_pipe_read_drab,
             SharedStorage &shared_storage,
             cute::tuple<int32_t, int32_t, int32_t> block_coord
             ) {
        Tensor sdRab = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_ds.data()), SmemLayoutPdS{});
        Tensor sdRab_pi = cute::as_position_independent_swizzle_tensor(sdRab);
        Tensor sdRabt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_ds.data()), SmemLayoutPdSt{});
        Tensor sdRabt_pi = cute::as_position_independent_swizzle_tensor(sdRabt);
        auto [n_block, bidh, bidb] = block_coord;
        int bidh_drab = params.qhead_per_rabhead_divmod.divide(bidh);

        // Prepare the TMA stores
        int offset_rab = Is_delta_q ? get_seqlen_k(params, bidb) - get_seqlen_q(params, bidb) : 0;
        Tensor mdRab = params.tma_store_dRab.get_tma_tensor(params.shape_dRab)(_, _, bidh_drab, bidb);
        Tensor gdRab = local_tile(domain_offset(make_coord(offset_rab, _0{}), mdRab), select<0, 1>(TileShape_MNK{}),
                make_coord(_, n_block));
        
        auto block_tma_dRab = params.tma_store_dRab.get_slice(_0{});
        Tensor tdRabgdRab = block_tma_dRab.partition_D(gdRab);  // (TMA, TMA_M, TMA_N, _)
        Tensor tdRabsdRab = block_tma_dRab.partition_S(sdRab); // (TMA, TMA_M, TMA_N)
        int m_block_max = get_m_block_max(params, n_block, bidb);
        int m_block_min = get_m_block_min(params, n_block, bidb);
        int m_block = m_block_min;
        #pragma unroll 2
        for (; m_block < m_block_max; ++m_block) {
            pipeline_drab.consumer_wait(smem_pipe_read_drab);
            if (cute::elect_one_sync()) {
                cute::copy(params.tma_store_dRab, tdRabsdRab(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read_drab.index())), tdRabgdRab(_, _, _, m_block));
                tma_store_arrive();
            }
            tma_store_wait<0>();
            pipeline_drab.consumer_release(smem_pipe_read_drab);
            ++smem_pipe_read_drab;
        }
    }

    CUTLASS_DEVICE void
    mma_init() {
        // Tell producer (warp 0) that smem_k and smem_v are ready
        cutlass::arch::NamedBarrier::arrive(NumMmaThreads + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::KVEmpty) /*id*/);
        int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
        if constexpr (dQacc_use_TMA) {
            if (cutlass::canonical_warp_group_idx() == 1 && warp_idx_in_warpgroup == 0) {
                cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
            }
        }
    }

    template <typename SharedStorage, typename FrgTensordKV>
    CUTLASS_DEVICE void
    mma(Params const& params,
        MainloopPipeline pipeline_q,
        MainloopPipeline pipeline_rab,
        MainloopPipeline_dO pipeline_do,
        MainloopPipeline_dRab pipeline_drab,
        PipelineState& smem_pipe_read,
        PipelineState_dO& smem_pipe_read_do,
        PipelineState_dRab& smem_pipe_write_drab,
        FrgTensordKV& tdKrdK,
        FrgTensordKV& tdVrdV,
        int thread_idx,
        int work_idx,
        cute::tuple<int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {
        static_assert(is_rmem<FrgTensordKV>::value, "dK and dV tensor must be rmem resident.");

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sRab = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_rab.data()), SmemLayoutRab{});
        Tensor sRabt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_rab.data()), SmemLayoutRabt{});
        Tensor sdO = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_do.data()), SmemLayoutdO{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sQt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_q.data()), SmemLayoutQt{});
        Tensor sdOt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_do.data()), SmemLayoutdOt{});
        Tensor sKt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_k.data()), SmemLayoutKt{});
        Tensor sP = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_p.data()), SmemLayoutPdS{});
        Tensor sP_pi = cute::as_position_independent_swizzle_tensor(sP);
        Tensor sPt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_p.data()), SmemLayoutPdSt{});
        Tensor sPt_pi = cute::as_position_independent_swizzle_tensor(sPt);
        Tensor sdS = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_ds.data()), SmemLayoutPdS{});
        Tensor sdS_pi = cute::as_position_independent_swizzle_tensor(sdS);
        Tensor sdSt = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_ds.data()), SmemLayoutPdSt{});
        Tensor sdSt_pi = cute::as_position_independent_swizzle_tensor(sdSt);
        Tensor sdQ = make_tensor(make_smem_ptr(shared_storage.mainloop.smem_dqacc.data()), SmemLayoutdQaccum{});

        static_assert(stride<0>(typename TiledMmaSdP::ALayout{}) == 0 and
                      stride<0>(typename TiledMmaSdP::BLayout{}) == 0 and
                      size<0>(typename TiledMmaSdP::ALayout{}) == cutlass::NumThreadsPerWarpGroup and
                      size<0>(typename TiledMmaSdP::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
                      "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");
        constexpr int MmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
        Layout warp_group_thread_layout = make_layout(make_shape(Int<MmaWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));
        Layout warp_group_thread_layout_dq = make_layout(make_shape(Int<NumdQWarpGroups>{}),
                                                      make_stride(Int<cutlass::NumThreadsPerWarpGroup>{}));

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);
        TiledMmaSdP tiled_mma_SdP;
        TiledMmadKV tiled_mma_dKV;
        TiledMmadQ tiled_mma_dQ;

        auto wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx));
        auto thread_mma_SdP = tiled_mma_SdP.get_thread_slice(thread_idx);
        auto wg_mma_dKV = tiled_mma_dKV.get_slice(warp_group_thread_layout(warp_group_idx));
        auto wg_mma_dQ = tiled_mma_dQ.get_slice(thread_idx);
        // auto wg_mma_dQ = tiled_mma_dQ.get_thread_slice(thread_idx);

        auto smem_tiled_copy_PdS = make_tiled_copy_C(SmemCopyAtomPdS{}, tiled_mma_SdP);
        auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(thread_idx);

        R2STiledCopydQaccum r2s_tiled_copy_dQaccum;
        // auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        auto r2s_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_thread_slice(NumdQWarpGroups == 2 ? thread_idx : thread_idx % cutlass::NumThreadsPerWarpGroup);
        Tensor tdQsdQaccum = r2s_thr_copy_dQaccum.partition_D(sdQ);

        // Allocate "fragments/descriptors"
        Tensor tSrQ = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sQ);
        Tensor tSrK = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sK);
        Tensor tdPrdO = mma_partition_fragment_AB</*A=*/!SdP_swapAB>(wg_mma_SdP, sdO);
        Tensor tdPrV = mma_partition_fragment_AB</*A=*/SdP_swapAB>(wg_mma_SdP, sV);
        Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);
        Tensor tdKrQ = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);
        Tensor tdQrdS = mma_partition_fragment_AB</*A=*/!dQ_swapAB>(wg_mma_dQ, sdS);
        Tensor tdQrK = mma_partition_fragment_AB</*A=*/dQ_swapAB>(wg_mma_dQ, sKt);

        Tensor tPsP = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sP_pi, sPt_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        Tensor tdSsdS = smem_thr_copy_PdS.partition_D(cute::conditional_return<!SdP_swapAB>(sdS_pi, sdSt_pi));      // ((Atom,AtomNum),PIPE_M,PIPE_N)
        // For Rab
        auto smem_tiled_copy_rab = make_tiled_copy_C(SmemCopyAtomRab{}, tiled_mma_SdP);
        auto smem_thr_copy_rab = smem_tiled_copy_rab.get_thread_slice(thread_idx);
        Tensor tSsRab = smem_thr_copy_rab.partition_S(cute::conditional_return<!SdP_swapAB>(sRab, sRabt)); // (CPY, CPY_M, CPY_N, PIPE)
        Tensor tSrRab = make_tensor<Element>(partition_shape_C(tiled_mma_SdP, cute::conditional_return<!SdP_swapAB>(select<0, 1>(TileShape_MNK{}), select<1, 0>(TileShape_MNK{}))));
        Tensor tSrRab_copy_view = smem_thr_copy_rab.retile_D(tSrRab); // (CPY, CPY_M, CPY_N)
        Tensor tSrRab_accum = make_tensor_like<ElementAccum>(tSrRab);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        int n_block = get<0>(block_coord);
        int bidb = get<2>(block_coord);
        int const seqlen_q = get_seqlen_q(params, bidb);
        int const seqlen_k = get_seqlen_k(params, bidb);
        int const seqlen_t = get_seqlen_t(params, bidb);
        // int const seqlen_i = seqlen_q - seqlen_t;
        int const seqlen_i = seqlen_k - seqlen_t; // for delta_q

        float param_seqlen_q = static_cast<float>(params.seqlen_q);

        Tensor mdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.ptr_dQaccum)),
                                      params.shape_dQaccum, params.stride_dQaccum)(_, _, get<1>(block_coord), 0);
        int const offset_padded = (params.cu_seqlens_q[bidb] + bidb * kBlockM) / kBlockM * kBlockM;
        Tensor gdQaccum = local_tile(domain_offset(make_coord(offset_padded, _0{}), mdQaccum), select<0, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (M, K, _)

        GmemTiledCopydQaccumAtomic gmem_tiled_copy_dQaccum;
        auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(thread_idx);
        Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

        auto causal_local_mask_fn = [&](auto& tSrS, int const m_block, auto is_causal_type, auto is_local_type, int const is_target_type = 0) {
            static constexpr int Row = !SdP_swapAB ? 0 : 1, Col = !SdP_swapAB ? 1 : 0;
            static constexpr int Mix = 1;
            static constexpr int Target = 2;
            constexpr bool Is_causal = decltype(is_causal_type)::value;
            constexpr bool Is_local = decltype(is_local_type)::value;
            Tensor cS = cute::make_identity_tensor(select<Row, Col>(TileShape_MNK{}));
            Tensor tScS = thread_mma_SdP.partition_C(cS);
            if constexpr (!Is_causal && !Is_local) {
                #pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    bool pred = int(get<Col>(tScS(i))) >= int(seqlen_k - n_block * kBlockN) ||
                        int(get<Row>(tScS(i))) >= int(seqlen_q - m_block * kBlockM);
                    tSrS(i) = pred ? -INFINITY : tSrS(i);
                }
                if (is_target_type == Mix) {
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        bool pred = int(get<Col>(tScS(i))) >= seqlen_i - n_block * kBlockN;
                        tSrS(i) = pred ? -INFINITY : tSrS(i);
                    }
                } // mask local
            } else {
                int causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
                if (is_target_type == Target) {
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        bool pred = int(get<Col>(tScS(i))) != std::min(int(get<Row>(tScS(i))) + causal_row_offset,
                                                                seqlen_k - n_block * kBlockN) - 1 ||
                            int(get<Row>(tScS(i))) >= seqlen_q - m_block * kBlockM;
                        tSrS(i) = pred ? -INFINITY : tSrS(i);
                    }
                } 
                else if constexpr (Is_causal) {
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        bool pred = int(get<Col>(tScS(i))) >= std::min(int(get<Row>(tScS(i))) + causal_row_offset,
                                                                seqlen_k - n_block * kBlockN) ||
                            int(get<Row>(tScS(i))) >= seqlen_q - m_block * kBlockM;
                        tSrS(i) = pred ? -INFINITY : tSrS(i);
                    } // mask upper
                    if (is_target_type == Mix) {
                        #pragma unroll
                        for (int i = 0; i < size(tSrS); ++i) {
                            bool pred = int(get<Col>(tScS(i))) != std::min(int(get<Row>(tScS(i))) + causal_row_offset,
                                                                    seqlen_k - n_block * kBlockN) - 1 &&
                                int(get<Col>(tScS(i))) >= seqlen_i - n_block * kBlockN;
                            tSrS(i) = pred ? -INFINITY : tSrS(i);
                        }
                    }
                } else {
                    int local_row_offset_right = causal_row_offset + params.window_size_right;
                    int local_row_offset_left = causal_row_offset - 1 - params.window_size_left;
                    #pragma unroll
                    for (int i = 0; i < size(tSrS); ++i) {
                        bool pred = int(get<Col>(tScS(i))) >= std::min(int(get<Row>(tScS(i))) + local_row_offset_right, seqlen_k - n_block * kBlockN) ||
                            int(get<Col>(tScS(i))) < int(get<Row>(tScS(i))) + local_row_offset_left ||
                            int(get<Row>(tScS(i))) >= seqlen_q - m_block * kBlockM;
                        tSrS(i) = pred ? -INFINITY : tSrS(i);
                    }
                }
            }
        };
        int m_block_max = get_m_block_max(params, n_block, bidb);
        int m_block_min = get_m_block_min(params, n_block, bidb);
        int m_block = m_block_min;

        clear(tdKrdK);
        clear(tdVrdV);
        // tiled_mma_dKV.accumulate_ = GMMA::ScaleOut::Zero;

        cutlass::ConsumerToken barrier_token = static_cast<cutlass::BarrierStatus>(shared_storage.barrier_KV.try_wait(work_idx % 2));
        if (barrier_token == cutlass::BarrierStatus::WaitAgain) { shared_storage.barrier_KV.wait(work_idx % 2); }

        auto bwd_step = [&](int m_block, auto mask_fn) {
            Tensor tSrS = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            pipeline_q.consumer_wait(smem_pipe_read);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tSrQ(_, _, _, smem_pipe_read.index()), tSrK, tSrS);
            if constexpr (Has_Rab){
                pipeline_rab.consumer_wait(smem_pipe_read);
                cute::copy(smem_tiled_copy_rab, tSsRab(_, _, _, smem_pipe_read.index()), tSrRab_copy_view);
                flash::convert_type_safe(tSrRab, tSrRab_accum);
            }
            Tensor tdPrdP = partition_fragment_C(tiled_mma_SdP, select<!SdP_swapAB ? 0 : 1, !SdP_swapAB ? 1 : 0>(TileShape_MNK{}));
            PipelineState_dO smem_pipe_read_do_cur = cute::conditional_return<Q_dO_same_stages>(smem_pipe_read, smem_pipe_read_do);
            pipeline_do.consumer_wait(smem_pipe_read_do_cur);
            flash::gemm</*zero_init=*/true, /*wg_wait=*/1, /*SwapAB=*/SdP_swapAB>(tiled_mma_SdP, tdPrdO(_, _, _, smem_pipe_read_do_cur.index()), tdPrV, tdPrdP);
            if constexpr (Has_Rab){
                for (int i = 0; i < size(tSrS); ++i) {
                    tSrS(i) += tSrRab_accum(i);
                }
                // Do not need sync here since the sync in later part can do it for us
                // cutlass::arch::NamedBarrier::sync(cutlass::NumThreadsPerWarpGroup, static_cast<int>(BwdNamedBarriers::AddRabWG1) - 1 + cutlass::canonical_warp_group_idx() /*id*/);
                pipeline_rab.consumer_release(smem_pipe_read);
            }
            for (int i = 0; i < size(tSrS); ++i) {
                tSrS(i) *= params.alpha;
            }

            mask_fn(tSrS, m_block);
            auto tSrS_silu = make_fragment_like(tSrS);
            silu_bwd(tSrS, tSrS_silu);
            for (int i = 0; i < size(tSrS_silu); ++i) {
                tSrS_silu(i) /= param_seqlen_q;
            }
            // mask_fn(tSrS_silu, m_block);
            // Convert scores from fp32 to fp16/bf16
            Tensor rP = make_tensor_like<Element>(tSrS_silu);
            flash::convert_type_safe(tSrS_silu, rP);

            if constexpr (!Slice_dQKV_Mma && Mma_dKV_is_RS) {
                Tensor tdVrP = make_tensor(rP.data(), convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdVrP, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
            } else {
                warpgroup_wait<0>();
            }
            for (int i = 0; i < size(tdPrdP); ++i) {
                tdPrdP(i) /= param_seqlen_q;
            }
            dsilu_bwd(tdPrdP, tSrS);
            for (int i = 0; i < size(tdPrdP); ++i) {
                tdPrdP(i) *= params.alpha;
            }

            if constexpr (!Mma_dKV_is_RS) {
                Tensor tPaP = smem_thr_copy_PdS.retile_S(rP);     // ((Atom,AtomNum), MMA_N, MMA_N)
                cute::copy(smem_tiled_copy_PdS, tPaP, tPsP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));
                cutlass::arch::fence_view_async_shared();
            }
            Tensor rdS = make_tensor_like<Element>(tdPrdP);
            flash::convert_type_safe(tdPrdP, rdS);
            Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            // If there's double buffering on dS, we don't need to sync here.
            // Otherwise we might have WG1 writing to dS before WG2 is done reading from it during MmadQ.
            // But because both WGs have to sync at the end of the loop and double buffering,
            // this race condition is not possible.
            // This sync is to ensure (1) P is written in case of !Mma_dKV_is_RS and
            // (2) dS is already read by the Mma in the previous iteration in case of Mma_dKV_is_RS.
            if constexpr (!Mma_dKV_is_RS || (kStages_dS == 1 && Mma_dKV_is_RS)) {
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/);
            }
            // For hdim 64, It's faster to write to smem_dS first before the dV gemm
            // Tensor tdSadS = smem_thr_copy_PdS.retile_S(rdS);     // ((Atom,AtomNum), MMA_N, MMA_N)
            if constexpr (Has_dRab) {
                pipeline_drab.producer_acquire(smem_pipe_write_drab);
            }
            cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index())));

            if constexpr (!Slice_dQKV_Mma) {
                // Most cases take this path, except for hdim256 where we want to slice to reduce register pressure
                if constexpr (Mma_dKV_is_RS) {
                    // If dKV is RS, it's slightly faster to kick off dK Mma before dQ_Mma
                    Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/1>(tiled_mma_dKV, tdKrdS, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                } else {
                    Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
                    Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                }
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                // SMEM fence to make sure sdS is written before it's read by WGMMA
                cutlass::arch::fence_view_async_shared();
                if constexpr (Has_dRab) {
                    pipeline_drab.producer_commit(smem_pipe_write_drab);
                }
                if constexpr (dQacc_use_TMA) {
                    cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
                } else {
                    cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/);
                }
                // cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/); // H100
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                if constexpr (Mma_dKV_is_RS) { pipeline_q.consumer_release(smem_pipe_read); }  // release Q
                if constexpr (dQacc_use_TMA) {
                    // H100
                    // cutlass::arch::NamedBarrier::sync(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQEmpty) /*id*/);  // sdQ empty, ready to be written to
                    Tensor taccdQrdQ = r2s_thr_copy_dQaccum.retile_S(tdQrdQ);        // ((Atom,AtomNum), MMA_M, MMA_N)
                    cute::copy(r2s_tiled_copy_dQaccum, taccdQrdQ, tdQsdQaccum);
                    cutlass::arch::fence_view_async_shared();
                    cutlass::arch::NamedBarrier::arrive(kNThreadsdQ + cutlass::NumThreadsPerWarp, static_cast<int>(BwdNamedBarriers::dQFull) /*id*/);  // sdQ full, to be written to gmem
                } else {
                    Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
                    Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
                    #pragma unroll
                    for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }
                }

                if constexpr (!Mma_dKV_is_RS) {
                    Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
                    Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                    flash::gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                }

            } else {  // Slice_dQKV_Mma

                Tensor tdVrP = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sPt);
                Tensor tdVrP_cur = tdVrP(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/-1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);

                cutlass::arch::fence_view_async_shared();
                if constexpr (Has_dRab) {
                    pipeline_drab.producer_commit(smem_pipe_write_drab);
                }
                cutlass::arch::NamedBarrier::sync(NumMmaThreads, static_cast<int>(BwdNamedBarriers::PdS) /*id*/);
                Tensor tdQrdQ = partition_fragment_C(tiled_mma_dQ, select<!dQ_swapAB ? 0 : 2, !dQ_swapAB ? 2 : 0>(TileShape_MNK{}));
                Tensor tdQrdS_cur = tdQrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/true, /*wg_wait=*/-1, /*SwapAB=*/dQ_swapAB, /*M_slice=*/0>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdVrP_cur, tdVrdO(_, _, _, smem_pipe_read_do_cur.index()), tdVrdV);
                Tensor tdQrdQ_atomic = recast<float4>(tdQrdQ);
                Tensor tdQgdQaccum_atomic = recast<float4>(tdQgdQaccum(_, _, _, m_block));
                #pragma unroll
                for (int i = 0; i < size(tdQrdQ_atomic) / 2; ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                Tensor tdKrdS = mma_partition_fragment_AB</*A=*/!dKV_swapAB>(wg_mma_dKV, sdSt);
                Tensor tdKrdS_cur = tdKrdS(_, _, _, cute::conditional_return<kStages_dS==1>(_0{}, smem_pipe_read.index()));
                flash::gemm</*zero_init=*/false, /*wg_wait=*/1, /*SwapAB=*/dKV_swapAB, /*M_slice=*/0>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
                pipeline_do.consumer_release(smem_pipe_read_do_cur);  // release dO

                flash::gemm</*zero_init=*/true, /*wg_wait=*/0, /*SwapAB=*/dQ_swapAB, /*M_slice=*/1>(tiled_mma_dQ, tdQrdS_cur, tdQrK, tdQrdQ);
                #pragma unroll
                for (int i = size(tdQrdQ_atomic) / 2;  i < size(tdQrdQ_atomic); ++i) { atomicAdd(&tdQgdQaccum_atomic(i), tdQrdQ_atomic(i)); }

                flash::gemm</*zero_init=*/false, /*wg_wait=*/0, /*SwapAB=*/dKV_swapAB, /*M_slice=*/1>(tiled_mma_dKV, tdKrdS_cur, tdKrQ(_, _, _, smem_pipe_read.index()), tdKrdK);
            }

            if constexpr (!Mma_dKV_is_RS) { pipeline_q.consumer_release(smem_pipe_read); }  // release Q
            ++smem_pipe_read;
            ++smem_pipe_write_drab;
            if constexpr (!Q_dO_same_stages) { ++smem_pipe_read_do; }
        };

        // We have separate iterations with causal masking. Not necessary for hdim 128 but for hdim 64
        // this helps quite a bit to not have to do causal masking for most of the iterations.
        if constexpr ((Is_causal || Is_local) && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { causal_local_mask_fn(tSrS, m_block, cute::bool_constant<Is_causal>{}, cute::bool_constant<Is_local>{}, get_target_mask(params, n_block, bidb)); };
            static constexpr int n_masking_steps = cute::ceil_div(kBlockN, kBlockM) + ((Is_delta_q) ? 1 : 0);
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < std::min(m_block_max, m_block_min + n_masking_steps); ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        static constexpr int n_local_bottom_steps = (!Is_local || !SeparateMaskingIterations) ? 0 : cute::ceil_div(kBlockN, kBlockM) + 1;
        // auto mask_fn = [&](auto& tSrS, int m_block) { causal_local_mask_fn(tSrS, m_block, cute::bool_constant<Is_causal && !SeparateMaskingIterations>{}, cute::bool_constant<Is_local && !SeparateMaskingIterations>{}, get_target_mask(params, n_block, bidb)); };
        auto mask_fn = [&](auto& tSrS, int m_block) { causal_local_mask_fn(tSrS, m_block, cute::bool_constant<Is_causal && !SeparateMaskingIterations>{}, cute::bool_constant<Is_local>{}, get_target_mask(params, n_block, bidb)); };
        CUTLASS_PRAGMA_NO_UNROLL
        for (; m_block < m_block_max - n_local_bottom_steps; ++m_block) {
            bwd_step(m_block, mask_fn);
        }

        if constexpr (Is_local && SeparateMaskingIterations) {
            auto mask_fn = [&](auto& tSrS, int m_block) { causal_local_mask_fn(tSrS, m_block, cute::bool_constant<false>{} /*is_causal*/, cute::bool_constant<Is_local>{}, get_target_mask(params, n_block, bidb)); };
            CUTLASS_PRAGMA_NO_UNROLL
            for (; m_block < m_block_max; ++m_block) {
                bwd_step(m_block, mask_fn);
            }
        }

        if constexpr (Q_dO_same_stages) { smem_pipe_read_do = smem_pipe_read; }
    }

};

} // namespace flash
