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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <atomic>
#include <barrier>
#include <iomanip>
#include <iostream>
#include <list>
#include <memory>
#include <queue>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace kvcache {

class KVOnloadHandle {
public:
    KVOnloadHandle(int num_layers);
    ~KVOnloadHandle();

    void init();
    void reset();
    void complete_host(int layer_idx, cudaStream_t stream);
    void wait_layer(int layer_idx);

public:
    int num_layers;
    std::vector<cudaEvent_t> compl_event;
    std::vector<cudaEvent_t> internal_onload_event;
    std::vector<int> host_complete;
    std::mutex mtx_;
    std::condition_variable cv_;

    bool inited;
    bool no_onload;
};

class KVOffloadHandle {
public:
    KVOffloadHandle(int num_layers);
    ~KVOffloadHandle();

    void init(
        at::Tensor offload_user_ids,
        at::Tensor offload_start_indices,
        std::vector<int>&& offload_lengths);
    // void reset();
    void complete_host(int layer_idx, cudaStream_t stream);
    void complete_host(int layer_idx, cudaStream_t stream, 
                       std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >>&& chunks);
    bool try_wait_layer(int layer_idx);
    float get_launch_time(void) { return this->time_stamp; }
    void set_launch_time(float time) { this->time_stamp = time; }
public:
    at::Tensor get_user_ids() const { return this->offload_user_ids; }
    at::Tensor get_start_indices() const { return this->offload_start_indices; }
    at::Tensor get_lengths() { return 
        at::from_blob(offload_lengths.data(), {offload_lengths.size()}, at::dtype(torch::kInt32)); 
    }

public:
    int num_layers;
    cudaEvent_t inference_event;
    std::vector<cudaEvent_t> ready_event;
    std::vector<cudaEvent_t> internal_gather_event;
    std::vector<std::atomic<int>> host_ready;

    at::Tensor offload_user_ids;
    at::Tensor offload_start_indices;
    std::vector<int> offload_lengths;

    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> chunks;
    
    bool inited;
    bool no_offload;
    float time_stamp;
};


class HostKVStorageImpl
{
public:
    HostKVStorageImpl(
        int num_layers,
        int num_kv_heads,
        int kv_headdim,
        int num_tokens_per_page,
        int64_t num_tokens_per_chunk,
        int64_t capacity_per_layer,
        int64_t max_batch_size,
        int64_t max_sequence_length,
        int device_idx
    );
    ~HostKVStorageImpl();

    void register_gpu_cache_table(std::vector<at::Tensor> table);

    at::Tensor lookup(at::Tensor user_ids);

    int64_t get_kvdata_length(int64_t user_id);
    std::pair<std::vector<void*>, std::vector<int64_t>> get_kvdata(int64_t user_id, int64_t length, int64_t layer_idx);
    std::vector<at::Tensor> get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat = true);

    // public:
    // void init_random_kvdata(int64_t user_id, size_t num_tokens);

private:
    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> get_empty_pinned_chunks(
        at::Tensor& offload_user_ids, 
        at::Tensor& offload_start_indices,
        const std::vector<int>& offload_num_pages_list);

public:
    void onload_kvcache(
        at::Tensor onload_user_ids,  // on host
        const std::vector<at::Tensor> onload_page_indices_list, // on gpu
        KVOnloadHandle& onloadhandle);
    bool offload_kvcache(
        at::Tensor offload_user_ids, // on host
        at::Tensor offload_start_indices,  // on host
        const std::vector<at::Tensor>& offload_page_indices_list, // on host
        KVOffloadHandle& offloadhandle);
    std::vector<int> finish_offload(KVOffloadHandle& offloadhandle);
    std::vector<int> cancel_offload(KVOffloadHandle& offloadhandle);

    void evict(int64_t uid);
    void evict_all();
    bool retain(int64_t uid);

public:
    const int num_layers;
    const int num_kv_heads;
    const int kv_headdim;
    const int num_tokens_per_page;
    const int64_t num_tokens_per_chunk;
    const int64_t capacity_per_layer;

    const int max_batch_size;
    const int max_sequence_length;
    size_t max_numel_per_layer_buffer;

    size_t num_pages_per_chunk;

    size_t unit_chunk_numel;
    size_t page_numel;
    size_t per_token_numel;

    size_t unit_chunk_bytes;
    size_t page_bytes;

    size_t inner_token_stride;
    size_t k2v_stride;
    size_t page_stride;

public:
    // device pointer for the cache table per layer (num_pages_per_layer, 2, num_tokens_per_page, num_kv_heads, kv_headdim)
    int device_idx;
    int device_num_sms;
    std::vector<void*> gpu_cache_table;

public:
    // bookkeeper
    std::list<int64_t> _lru_list;
    std::unordered_map<int64_t, 
                       typename std::list<int64_t>::iterator> _lru_lookup_table;
    std::queue<std::pair<int64_t, int64_t>> _empty_chunks;  // <offset in #unit_chunks, size in #unit_chunks>
    int64_t _num_empty_chunks;
    std::unordered_map<int64_t, std::pair<std::vector<int64_t>, std::vector<int64_t>>> _uid_to_chunks;
    std::unordered_map<int64_t, int64_t> _uid_to_length;

public:
    cudaStream_t onload_stream;
    cudaStream_t offload_stream;
    cudaStream_t scatter_stream;
    cudaStream_t gather_stream;

public:
    // internal device buffer
    std::vector<void*> onload_gpu_buffers;
    std::vector<void*> offload_gpu_buffers;
    std::vector<void*> pinned_kvstorage_buffers;
};


}  // namespace kvcache