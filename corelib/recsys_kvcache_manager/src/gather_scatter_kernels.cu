/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Implementation based on FlashInfer library.
# 
******************************************************************************/

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <iostream>
#include "vec_dtypes.cuh"

#define cudaCheck(ans) { cudaSuccesAssert((ans), __FILE__, __LINE__); }
inline void cudaSuccesAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)       \
  switch (head_dim) {                                    \
    case 64: {                                           \
      constexpr size_t HEAD_DIM = 64;                    \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    case 128: {                                          \
      constexpr size_t HEAD_DIM = 128;                   \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    case 256: {                                          \
      constexpr size_t HEAD_DIM = 256;                   \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    case 512: {                                          \
      constexpr size_t HEAD_DIM = 512;                   \
      __VA_ARGS__                                        \
      break;                                             \
    }                                                    \
    default: {                                           \
      std::cerr << "Unsupported head_dim: " << head_dim; \
      return cudaErrorInvalidValue;                      \
    }                                                    \
  }

void get_uint_fastdiv_msa(uint32_t d, uint32_t &m, uint32_t &s, uint32_t &a) {
    unsigned int p, nc, delta, q1, r1, q2, r2;
    a = 0;
    nc = unsigned(-1) - unsigned(-d) % d;
    p = 31;
    q1 = 0x80000000 / nc;
    r1 = 0x80000000 - q1 * nc;
    q2 = 0x7FFFFFFF / d;
    r2 = 0x7FFFFFFF - q2 * d;
    do {
      p++;
      if (r1 >= nc - r1) {
        q1 = 2 * q1 + 1;
        r1 = 2 * r1 - nc;
      } else {
        q1 = 2 * q1;
        r1 = 2 * r1;
      }
      if (r2 + 1 >= d - r2) {
        if (q2 >= 0x7FFFFFFF) a = 1;
        q2 = 2 * q2 + 1;
        r2 = 2 * r2 + 1 - d;
      } else {
        if (q2 >= 0x80000000) a = 1;
        q2 = 2 * q2;
        r2 = 2 * r2 + 1;
      }
      delta = d - 1 - r2;
    } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));
    m = q2 + 1;
    s = p - 32;
}

__host__ __device__ __forceinline__ void divmod(uint32_t n, uint32_t d,
                                                uint32_t m, uint32_t s, uint32_t a, 
                                                uint32_t& q, uint32_t& r) {
    if (d == 1) {
        q = n;
    } else {
#ifdef __CUDA_ARCH__
        q = __umulhi(m, n);
#else
        q = (((unsigned long long)((long long)m * (long long)n)) >> 32);
#endif
        q += a * n;
        q >>= s;
    }
    r = n - q * d;
}


template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void ScatterPagedKVCacheKernel(DType* __restrict__ continuous_kv,
                                         IdType* __restrict__ page_ids,
                                         uint32_t page_size,
                                         uint32_t stride_page,
                                         uint32_t stride_k2v,
                                         uint32_t stride_n,
                                         uint32_t stride_h,
                                         uint32_t nnz,
                                         DType*  kv_cache,
                                         uint32_t m, uint32_t s, uint32_t a) {
    uint32_t tx = threadIdx.x, ty = threadIdx.y;
    uint32_t head_idx = ty;
    uint32_t cta_id = blockIdx.x;
    uint32_t num_ctas = gridDim.x;

    DType* __restrict__ continuous_k = continuous_kv;
    DType* __restrict__ continuous_v = continuous_kv + stride_k2v;
    DType* k_cache = kv_cache;
    DType* v_cache = kv_cache + stride_k2v;

#pragma unroll 4
    for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
        uint32_t page_id_idx, entry_idx;
        divmod(i, page_size, m, s, a,
            page_id_idx, entry_idx);
        size_t inner_page_offset = head_idx * stride_h + entry_idx * stride_n + tx * vec_size;
        size_t dst_offset = __ldg(page_ids + page_id_idx) * stride_page + inner_page_offset;
        size_t src_offset = page_id_idx * stride_page + inner_page_offset;
        vec_t<DType, vec_size>::memcpy(
            k_cache + dst_offset, continuous_k + src_offset);
        vec_t<DType, vec_size>::memcpy(
            v_cache + dst_offset, continuous_v + src_offset);
    }
}

template <typename DType, typename IdType>
cudaError_t ScatterPagedKVCache(DType* continuous_kv,
                                IdType* page_ids,
                                uint32_t num_heads,
                                uint32_t head_dim,
                                uint32_t page_size,
                                uint32_t stride_page,
                                uint32_t stride_k2v,
                                uint32_t stride_n,
                                uint32_t stride_h,
                                DType* kv_cache,
                                uint32_t nnz,
                                int num_sms,
                                cudaStream_t stream) {
  int num_blocks_per_sm = 0;

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    uint32_t num_threads = bdx * bdy;
    uint32_t smem_size = 0;
    auto kernel = ScatterPagedKVCacheKernel<HEAD_DIM, vec_size, DType, IdType>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                  num_threads, smem_size);
    num_blocks_per_sm = min(num_blocks_per_sm, ((int(nnz) + num_sms - 1) / num_sms));
    dim3 nblks(num_blocks_per_sm * num_sms);
    dim3 nthrs(bdx, bdy);

    uint32_t m, s, a;
    get_uint_fastdiv_msa(page_size, m, s, a);

    void* args[] = {(void*)&continuous_kv, (void*)&page_ids,      (void*)&page_size,
                    (void*)&stride_page,   (void*)&stride_k2v,    (void*)&stride_n,
                    (void*)&stride_h,      (void*)&nnz,           (void*)&kv_cache,
                    (void*)&m,             (void*)&s,             (void*)&a};
    cudaCheck(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template 
cudaError_t ScatterPagedKVCache<nv_bfloat16, int32_t>(
    nv_bfloat16* continuous_kv,
    int32_t* page_ids,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t stride_page,
    uint32_t stride_k2v,
    uint32_t stride_n,
    uint32_t stride_h,
    nv_bfloat16* kv_cache,
    uint32_t nnz,
    int num_sms,
    cudaStream_t stream);


template <uint32_t head_dim, uint32_t vec_size, typename DType, typename IdType>
__global__ void GatherPagedKVCacheKernel(DType* gather_kv,
                                         IdType* __restrict__ page_ids,
                                         uint32_t page_size,
                                         uint32_t stride_page,
                                         uint32_t stride_k2v,
                                         uint32_t stride_n,
                                         uint32_t stride_h,
                                         uint32_t nnz,
                                         DType* __restrict__ kv_cache,
                                         uint32_t m, uint32_t s, uint32_t a) {
    uint32_t tx = threadIdx.x, ty = threadIdx.y;
    uint32_t head_idx = ty;
    uint32_t cta_id = blockIdx.x;
    uint32_t num_ctas = gridDim.x;

    DType* gather_k = gather_kv;
    DType* gather_v = gather_kv + stride_k2v;
    DType* __restrict__ k_cache = kv_cache;
    DType* __restrict__ v_cache = kv_cache + stride_k2v;

#pragma unroll 4
    for (uint32_t i = cta_id; i < nnz; i += num_ctas) {
        uint32_t page_id_idx, entry_idx;
        divmod(i, page_size, m, s, a,
            page_id_idx, entry_idx);
        size_t inner_page_offset = head_idx * stride_h + entry_idx * stride_n + tx * vec_size;
        size_t src_offset = __ldg(page_ids + page_id_idx) * stride_page + inner_page_offset;
        size_t dst_offset = page_id_idx * stride_page + inner_page_offset;
        vec_t<DType, vec_size>::memcpy(
            gather_k + dst_offset, k_cache + src_offset);
        vec_t<DType, vec_size>::memcpy(
            gather_v + dst_offset, v_cache + src_offset);
    }
}

template <typename DType, typename IdType>
cudaError_t GatherPagedKVCache(DType* gather_kv,
                               IdType* page_ids,
                               uint32_t num_heads,
                               uint32_t head_dim,
                               uint32_t page_size,
                               uint32_t stride_page,
                               uint32_t stride_k2v,
                               uint32_t stride_n,
                               uint32_t stride_h,
                               DType* kv_cache,
                               uint32_t nnz,
                               int num_sms,
                               cudaStream_t stream) {
  int num_blocks_per_sm = 0;

  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16 / sizeof(DType), HEAD_DIM / 32);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    uint32_t num_threads = bdx * bdy;
    uint32_t smem_size = 0;
    auto kernel = GatherPagedKVCacheKernel<HEAD_DIM, vec_size, DType, IdType>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                  num_threads, smem_size);
    num_blocks_per_sm = min(num_blocks_per_sm, ((int(nnz) + num_sms - 1) / num_sms));
    dim3 nblks(num_blocks_per_sm * num_sms);
    dim3 nthrs(bdx, bdy);

    uint32_t m, s, a;
    get_uint_fastdiv_msa(page_size, m, s, a);

    void* args[] = {(void*)&gather_kv,     (void*)&page_ids,      (void*)&page_size,
                    (void*)&stride_page,   (void*)&stride_k2v,    (void*)&stride_n,
                    (void*)&stride_h,      (void*)&nnz,           (void*)&kv_cache,
                    (void*)&m,             (void*)&s,             (void*)&a};
    cudaCheck(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  });
  return cudaSuccess;
}

template 
cudaError_t GatherPagedKVCache<nv_bfloat16, int32_t>(
  nv_bfloat16* gather_kv,
  int32_t* page_ids,
  uint32_t num_heads,
  uint32_t head_dim,
  uint32_t page_size,
  uint32_t stride_page,
  uint32_t stride_k2v,
  uint32_t stride_n,
  uint32_t stride_h,
  nv_bfloat16* kv_cache,
  uint32_t nnz,
  int num_sms,
  cudaStream_t stream);


__global__ void GetPagedBatchIndicesPositionsKernel(
  int32_t batch_size,
  int32_t* append_indptr,
  int32_t* seq_lens_ptr,
  int32_t* batch_indices_ptr,
  int32_t* positions_ptr) {

  int32_t tx = threadIdx.x;
  int32_t seq_idx = blockIdx.x;
  int32_t seq_start = append_indptr[seq_idx];
  int32_t total_seq_len = seq_lens_ptr[seq_idx];
  int32_t append_per_seq = append_indptr[seq_idx + 1] - seq_start;

  int32_t* batch_indices_ptr_per_seq = batch_indices_ptr + seq_start;
  int32_t* positions_ptr_per_seq = positions_ptr + seq_start;
  int32_t pos_start = total_seq_len - append_per_seq;

#pragma unroll 4
  for (int32_t i = tx; i < append_per_seq; i += blockDim.x) {
    batch_indices_ptr_per_seq[i] = seq_idx;
    positions_ptr_per_seq[i] = pos_start + i;
  }
}

cudaError_t GetPagedBatchIndicesPositions(
  int32_t batch_size,
  int32_t* append_indptr,
  int32_t* seq_lens_ptr,
  int32_t* batch_indices_ptr,
  int32_t* positions_ptr,
  cudaStream_t stream
)
{
  dim3 nblks(batch_size);
  dim3 nthrs(128, 1);

  void* args[] = {(void*)&batch_size,     (void*)&append_indptr,      (void*)&seq_lens_ptr,    
                  (void*)&batch_indices_ptr,   (void*)&positions_ptr};
  auto kernel = GetPagedBatchIndicesPositionsKernel;
  cudaCheck(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, 0, stream));
  return cudaSuccess;
}