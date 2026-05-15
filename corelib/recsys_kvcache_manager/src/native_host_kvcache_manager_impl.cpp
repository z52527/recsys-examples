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

#include "native_host_kvcache_manager_impl.h"
#include <nvtx3/nvtx3.hpp>

#define cudaCheck(ans) { cudaSuccesAssert((ans), __FILE__, __LINE__); }
inline void cudaSuccesAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void scatter_paged_kvcache(
    uint16_t *paged_kvcache_table,    // output
    uint16_t *continuous_gpu_buffer,  // input: gpu onload buffer
    int *page_ids,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t stride_page,
    uint32_t stride_k2v,
    uint32_t stride_n,
    uint32_t stride_h,
    uint32_t num_pages,
    const int num_sms,
    cudaStream_t stream);

void gather_paged_kvcache(
    uint16_t *gather_gpu_buffer,    // output: gpu offload buffer
    uint16_t *paged_kvcache_table,  // input
    int *page_ids,
    uint32_t num_heads,
    uint32_t head_dim,
    uint32_t page_size,
    uint32_t stride_page,
    uint32_t stride_k2v,
    uint32_t stride_n,
    uint32_t stride_h,
    uint32_t num_pages,
    const int num_sms,
    cudaStream_t stream);

namespace kvcache {


KVOnloadHandle::KVOnloadHandle(int num_layers)
: num_layers(num_layers)
, compl_event(std::vector<cudaEvent_t>(num_layers))
, internal_onload_event(std::vector<cudaEvent_t>(num_layers))
, host_complete(num_layers, 0)
, inited(false)
, no_onload(true) {}

KVOnloadHandle::~KVOnloadHandle(){
    if (!inited) return;
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        cudaEventDestroy(compl_event[layer_idx]);
        cudaEventDestroy(internal_onload_event[layer_idx]);
    }
};

void KVOnloadHandle::init() {
    this->no_onload = false;
    if (!inited) {
        for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx ++) {
            cudaCheck(cudaEventCreateWithFlags(&this->compl_event[layer_idx], cudaEventDisableTiming));
            cudaCheck(cudaEventCreateWithFlags(&this->internal_onload_event[layer_idx], cudaEventDisableTiming));
            host_complete[layer_idx] = 0;
        }
        inited = true;
    }
}

void KVOnloadHandle::reset() {
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        host_complete[layer_idx] = 0;
    }
}

void KVOnloadHandle::complete_host(int layer_idx, cudaStream_t stream) {
    cudaCheck(cudaEventRecord(compl_event[layer_idx], stream));
    {
        std::unique_lock<std::mutex> lock(mtx_);
        host_complete[layer_idx] = 1;
    }
    cv_.notify_one();
};

void KVOnloadHandle::wait_layer(int layer_idx) {
    if (no_onload) return;
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this, layer_idx](){ return host_complete[layer_idx] == 1; });
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaCheck(cudaStreamWaitEvent(stream, compl_event[layer_idx], 0));
};


KVOffloadHandle::KVOffloadHandle(
    int num_layers
)
: num_layers(num_layers)
, internal_gather_event(num_layers)
, ready_event(num_layers)
, host_ready(num_layers)
, inited(false)
, no_offload(true) {
    for (int i = 0; i < num_layers; i++) {
        host_ready[i].store(0, std::memory_order_release);
    }
}

KVOffloadHandle::~KVOffloadHandle() {
    if (!inited) return;
    cudaCheck(cudaEventDestroy(inference_event));
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
        cudaCheck(cudaEventDestroy(internal_gather_event[layer_idx]));
        cudaCheck(cudaEventDestroy(ready_event[layer_idx]));
    }
}

void KVOffloadHandle::init(
    at::Tensor offload_user_ids,
    at::Tensor offload_start_indices,
    std::vector<int>&& offload_lengths
) {
    this->no_offload = false;
    if (!inited) {
        cudaCheck(cudaEventCreateWithFlags(&inference_event, cudaEventDisableTiming));
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx ++) {
            cudaCheck(cudaEventCreateWithFlags(&internal_gather_event[layer_idx], cudaEventDisableTiming));
            cudaCheck(cudaEventCreateWithFlags(&ready_event[layer_idx], cudaEventDisableTiming));
        }
        inited = true;
    }
    this->offload_user_ids = offload_user_ids;
    this->offload_start_indices = offload_start_indices;
    this->offload_lengths = std::move(offload_lengths);
}

void KVOffloadHandle::complete_host(int layer_idx, cudaStream_t stream) {
    cudaCheck(cudaEventRecord(this->ready_event[layer_idx], stream));
    this->host_ready[layer_idx].store(1, std::memory_order_release);
}

void KVOffloadHandle::complete_host(int layer_idx, cudaStream_t stream,
                                    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >>&& chunks) {
    this->chunks = std::move(chunks);
    cudaCheck(cudaEventRecord(this->ready_event[layer_idx], stream));
    this->host_ready[layer_idx].store(1, std::memory_order_release);
}

bool KVOffloadHandle::try_wait_layer(int layer_idx) {
    if (layer_idx == -1) layer_idx = this->host_ready.size() - 1;
    if (this->host_ready[layer_idx].load(std::memory_order_acquire) == 0) {
        return false;
    }
    auto status = cudaEventQuery(this->ready_event[layer_idx]);
    if (status == cudaSuccess) {
        return true;
    }
    return false;
}

HostKVStorageImpl::HostKVStorageImpl(
    int num_layers,
    int num_kv_heads,
    int kv_headdim,
    int num_tokens_per_page,
    int64_t num_tokens_per_chunk,
    int64_t capacity_per_layer,
    int64_t max_batch_size,
    int64_t max_sequence_length,
    int device_idx
)
    : num_layers(num_layers)
    , num_kv_heads(num_kv_heads)
    , kv_headdim(kv_headdim)
    , num_tokens_per_page(num_tokens_per_page)
    , num_tokens_per_chunk(num_tokens_per_chunk)
    , capacity_per_layer(capacity_per_layer)
    , max_batch_size(max_batch_size)
    , max_sequence_length(max_sequence_length)
    , device_idx(device_idx)
    , _uid_to_chunks()
{
    size_t padded_pages_per_seq = (max_sequence_length + num_tokens_per_page - 1) / num_tokens_per_page;

    this->num_pages_per_chunk = num_tokens_per_chunk / num_tokens_per_page;

    this->unit_chunk_numel = num_tokens_per_chunk * 2 * num_kv_heads * kv_headdim;
    this->page_numel = 2 * num_tokens_per_page * num_kv_heads * kv_headdim;
    this->per_token_numel = 2 * num_kv_heads * kv_headdim;

    this->unit_chunk_bytes = unit_chunk_numel * sizeof(uint16_t);
    this->page_bytes = page_numel * sizeof(uint16_t);

    this->inner_token_stride = num_kv_heads * kv_headdim;
    this->k2v_stride = num_tokens_per_page * inner_token_stride;
    this->page_stride = 2 * k2v_stride;

    c10::cuda::OptionalCUDAGuard device_guard(this->device_idx);

    cudaCheck(cudaDeviceGetAttribute(&this->device_num_sms, cudaDevAttrMultiProcessorCount, device_idx));
    cudaCheck(cudaStreamCreateWithFlags(&onload_stream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&offload_stream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&scatter_stream, cudaStreamNonBlocking));
    cudaCheck(cudaStreamCreateWithFlags(&gather_stream, cudaStreamNonBlocking));

    this->max_numel_per_layer_buffer = max_batch_size * padded_pages_per_seq * page_numel;
    for (int i = 0; i < 2; i++) {
        void *ptr;
        cudaCheck(cudaMalloc(&ptr, max_numel_per_layer_buffer * sizeof(uint16_t)));
        onload_gpu_buffers.push_back(ptr);
    }
    for (int i = 0; i < 2; i++) {
        void *ptr;
        cudaCheck(cudaMalloc(&ptr, max_numel_per_layer_buffer * sizeof(uint16_t)));
        offload_gpu_buffers.push_back(ptr);
    }

    std::cout << "[INFO] Allocating pinned memory for host kvcache manager ..." << std::endl;
    for (int i = 0; i < num_layers; i++) {
        std::cout << "[INFO]  -- layer " << i << ": " << (capacity_per_layer / 1024. / 1024. / 1024) <<     " GiB ..." <<  std::endl; 
        void *ptr;
        cudaCheck(cudaMallocHost(&ptr, capacity_per_layer));
        pinned_kvstorage_buffers.push_back(ptr);
    }
    std::cout << "[INFO] ... done." << std::endl;

    _num_empty_chunks = capacity_per_layer / unit_chunk_bytes;
    _empty_chunks.push(
        std::make_pair(0, _num_empty_chunks));

};

HostKVStorageImpl::~HostKVStorageImpl()
{
    for (void* ptr : pinned_kvstorage_buffers) {
        cudaCheck(cudaFreeHost(ptr));
    }
    for (void* ptr : offload_gpu_buffers) {
        cudaCheck(cudaFree(ptr));
    }
    for (void* ptr : onload_gpu_buffers) {
        cudaCheck(cudaFree(ptr));
    }

    cudaCheck(cudaStreamDestroy(gather_stream));
    cudaCheck(cudaStreamDestroy(scatter_stream));
    cudaCheck(cudaStreamDestroy(offload_stream));
    cudaCheck(cudaStreamDestroy(onload_stream));
}


void HostKVStorageImpl::register_gpu_cache_table(
    std::vector<at::Tensor> table)
{
    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        this->gpu_cache_table.push_back(
            table[layer_idx].data_ptr()
        );
    }
}

at::Tensor HostKVStorageImpl::lookup(at::Tensor user_ids) {
    at::Tensor cached_lengths = at::empty({user_ids.size(0)}, torch::kInt32);
    int *lengths_ptr = cached_lengths.data_ptr<int>();
    for (auto i = 0; i < user_ids.size(0); i++) {
        int64_t uid = user_ids[i].item<int64_t>();
        auto it = _uid_to_length.find(uid);
        lengths_ptr[i] = (it ==  _uid_to_length.end()) ? 0 : it->second;
    }
    return cached_lengths;
}

int64_t HostKVStorageImpl::get_kvdata_length(int64_t user_id) {
    auto it = _uid_to_length.find(user_id);
    if (it ==  _uid_to_length.end()) return 0;
    return it->second;
};

std::pair<std::vector<void*>, std::vector<int64_t>> HostKVStorageImpl::get_kvdata(int64_t user_id, int64_t length, int64_t layer_idx) {
    // assert(this->get_kvdata_length(user_id) >= length);
    std::vector<void*> chunk_ptrs;
    std::vector<int64_t> chunk_bytes;
    if (length == 0) {
        return std::make_pair(chunk_ptrs, chunk_bytes);
    }

    const auto &chunk_offsets = _uid_to_chunks[user_id].first;
    const auto &chunk_psizes = _uid_to_chunks[user_id].second;

    int64_t tokens_from_chunks = 0;
    for (size_t idx = 0; idx < chunk_offsets.size(); idx++) {
        void* src_ptr = reinterpret_cast<void*>(reinterpret_cast<char*>(this->pinned_kvstorage_buffers[layer_idx]) + chunk_offsets[idx]);
        chunk_ptrs.push_back(src_ptr);
        if (tokens_from_chunks + chunk_psizes[idx] * this->num_tokens_per_page >= length) {
            int64_t last_chunk_tokens = length - tokens_from_chunks;
            chunk_bytes.push_back(last_chunk_tokens * this->per_token_numel * sizeof(uint16_t));
            break;
        }
        tokens_from_chunks += chunk_psizes[idx] * this->num_tokens_per_page;
        chunk_bytes.push_back(chunk_psizes[idx] * this->num_tokens_per_page * this->per_token_numel * sizeof(uint16_t));
    }
    // assert(tokens_from_chunks == length);
    return std::make_pair(chunk_ptrs, chunk_bytes);
};


std::vector<at::Tensor> HostKVStorageImpl::get_kvdata_tensor(std::vector<int64_t> user_ids, bool with_concat) {
    int batch_size = user_ids.size();

    std::vector<int64_t> seqlens(batch_size, 0);
    int64_t total_seqlen = 0;
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        seqlens[seq_idx] = get_kvdata_length(user_ids[seq_idx]);
        total_seqlen += seqlens[seq_idx];
    }
    int64_t num_total_pages = total_seqlen / this->num_tokens_per_page;

    at::Tensor tensor_res = at::empty({
        this->num_layers, num_total_pages, 2, this->num_tokens_per_page, this->num_kv_heads, this->kv_headdim}, torch::kBFloat16);
    
    int64_t seqlen_offset = 0;
    uint16_t *raw_ptr = reinterpret_cast<uint16_t*>(tensor_res.data_ptr());

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        uint16_t *seq_ptr = raw_ptr + (seqlen_offset / this->num_tokens_per_page) * this->page_numel;

        for (int layer_idx = 0 ; layer_idx < this->num_layers; layer_idx++){
            auto [chunk_ptrs, chunk_bytes] = get_kvdata(user_ids[seq_idx], seqlens[seq_idx], layer_idx);
            int64_t running_numel = 0;
            for (size_t chk_idx = 0; chk_idx < chunk_ptrs.size(); chk_idx++) {
                std::memcpy(seq_ptr + layer_idx * tensor_res.stride(0) + running_numel, chunk_ptrs[chk_idx], chunk_bytes[chk_idx]);
                running_numel += chunk_bytes[chk_idx] / sizeof(uint16_t);
            }
        }

        seqlen_offset += seqlens[seq_idx];
    }
    
    std::vector<at::Tensor> res({tensor_res});
    return res;
}

// void HostKVStorageImpl::init_random_kvdata(int64_t user_id, size_t num_tokens) {
//     if (_uid_to_length.find(user_id) != _uid_to_length.end()) return;
//     if (num_tokens == 0) return;
    
//     size_t num_chunks = ((num_tokens + this->num_tokens_per_chunk - 1) / this->num_tokens_per_chunk);

//     uint16_t *host_data_ptr = (uint16_t *)malloc(this->num_layers * num_chunks * this->unit_chunk_numel * sizeof(uint16_t));

//     for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
//         _uid_to_chunk_id[layer_idx][user_id] = std::vector<uintptr_t>();
//     _uid_to_mempool[user_id] = std::vector<uintptr_t>();

//     for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
//         uint16_t* src_ptr = host_data_ptr + layer_idx * num_chunks * this->unit_chunk_numel;
        
//         for (size_t chunk_idx = 0; chunk_idx < num_chunks; chunk_idx ++) {
//             _uid_to_chunk_id[layer_idx][user_id].push_back(reinterpret_cast<uintptr_t>(src_ptr + chunk_idx * this->unit_chunk_numel));
//         }
//     }
//     _uid_to_mempool[user_id].push_back(reinterpret_cast<uintptr_t>(host_data_ptr));
//     _uid_to_length[user_id] = num_chunks * this->num_tokens_per_chunk;
// }


void HostKVStorageImpl::onload_kvcache(
    at::Tensor onload_user_ids,  // on host
    const std::vector<at::Tensor> onload_page_indices_list, // on gpu
    KVOnloadHandle& onloadhandle) {
    const int batch_size = onload_user_ids.size(0);

    at::Device device = onload_page_indices_list[0].device();
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    c10::cuda::CUDAStream c10_onload_stream =
        c10::cuda::getStreamFromExternal(this->onload_stream, device.index());

    onloadhandle.init();
    // The onload_page_indices_list are views from paged_indices for inference attention, which 
    // contains more page_ids than required onload, so we need to do extra gpu memory allocation and memcpy.
    // Note: assume uid & page_ids without onload are already removed.
    // [[ contains allocation from torch gpu memory, and memcpy(concat) on gpu ]]
    c10::cuda::CUDAStreamGuard onload_stream_guard(c10_onload_stream);
    at::Tensor onload_page_indices = at::cat(onload_page_indices_list, 0);

    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        // nvtx3::scoped_range r{"onload_layer_" + std::to_string(layer_idx)};
        char *gpu_onload_buffer = reinterpret_cast<char*>(this->onload_gpu_buffers[layer_idx % 2]);  // single layer for a max batch (multi-chunks)
        if (layer_idx >= 2) cudaCheck(cudaStreamWaitEvent(this->onload_stream, onloadhandle.compl_event[layer_idx - 2], 0));

        size_t bytes_offset_in_buffer = 0;
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            auto [chunk_ptrs, chunk_bytes] = this->get_kvdata(
                onload_user_ids[seq_idx].item<int64_t>(), 
                onload_page_indices_list[seq_idx].size(0) * this->num_tokens_per_page, 
                layer_idx
            );
            for (int chunk_idx = 0; chunk_idx < chunk_ptrs.size(); chunk_idx++) {
                cudaCheck(cudaMemcpyAsync(gpu_onload_buffer + bytes_offset_in_buffer,
                    chunk_ptrs[chunk_idx], chunk_bytes[chunk_idx], cudaMemcpyHostToDevice, this->onload_stream));
                bytes_offset_in_buffer += chunk_bytes[chunk_idx];
            }
        }
        cudaCheck(cudaEventRecord(onloadhandle.internal_onload_event[layer_idx], this->onload_stream));
        cudaCheck(cudaStreamWaitEvent(this->scatter_stream, onloadhandle.internal_onload_event[layer_idx], 0));
        scatter_paged_kvcache(reinterpret_cast<uint16_t*>(this->gpu_cache_table[layer_idx]), 
            reinterpret_cast<uint16_t*>(gpu_onload_buffer), 
            onload_page_indices.data_ptr<int>(), 
            this->num_kv_heads, this->kv_headdim, this->num_tokens_per_page, this->page_stride, this->k2v_stride, 
            this->inner_token_stride, this->kv_headdim, onload_page_indices.size(0),
            this->device_num_sms, this->scatter_stream);
        onloadhandle.complete_host(layer_idx, this->scatter_stream);
    }
    // For next onload, onload stream wait for current onload completion so gpu onload buffers are not in use.
    cudaCheck(cudaStreamWaitEvent(this->onload_stream, onloadhandle.compl_event[this->num_layers-1], 0));
}


std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> HostKVStorageImpl::get_empty_pinned_chunks(
    at::Tensor& offload_user_ids, 
    at::Tensor& offload_start_indices,
    const std::vector<int>& offload_num_pages_list) {
    const auto batch_size = offload_user_ids.size(0);

    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> empty_pinned_chunks;

    // Check if have enough buffers and/or need eviction
    int64_t num_pages_required = 0;
    for (auto idx = 0; idx < batch_size; idx++) {
        int64_t start_pos = offload_start_indices[idx].item<int64_t>();
        int64_t num_unaligned_pages = (start_pos % this->num_tokens_per_chunk) / this->num_tokens_per_page;
        num_unaligned_pages = (num_unaligned_pages == 0) ? this->num_pages_per_chunk : num_unaligned_pages;
        num_pages_required += static_cast<int64_t>((
            offload_num_pages_list[idx] + num_unaligned_pages - 1
        ) / this->num_pages_per_chunk) * this->num_pages_per_chunk;
    }

    std::vector<int64_t> uids_to_evict;
    std::unordered_set<int64_t> freezed_uids(offload_user_ids.data_ptr<int64_t>(), offload_user_ids.data_ptr<int64_t>() + batch_size);

    int64_t num_available_pages = _num_empty_chunks * this->num_pages_per_chunk;
    for (auto it = std::rbegin(_lru_list); it != std::rend(_lru_list); it++) {
        if (num_pages_required <= num_available_pages) break;
        
        const int64_t uid_to_evict = *it;
        if (freezed_uids.find(uid_to_evict) != freezed_uids.end()) continue;

        const auto &chunk_psizes = _uid_to_chunks[uid_to_evict].second;
        for (size_t idx = 0; idx < chunk_psizes.size(); idx++) {
            num_available_pages += chunk_psizes[idx];
        }
        uids_to_evict.push_back(uid_to_evict);
    }
    if (num_available_pages < num_pages_required) return empty_pinned_chunks;

    // Actual eviction
    for (auto uid_to_evict : uids_to_evict) evict(uid_to_evict);
    
    // Actual allocation
    for (auto idx = 0; idx < batch_size; idx++) {
        int64_t user_id = offload_user_ids[idx].item<int64_t>();
        int start_pos = offload_start_indices[idx].item<int64_t>();
        int offload_num_pages = offload_num_pages_list[idx];

        std::vector<int64_t> chunk_offsets;
        std::vector<int64_t> chunk_psizes;

        if (start_pos % num_tokens_per_chunk != 0)  {
            // if start_pos is not aligned to chunk size, reuse the last offloaded chunk.
            auto partial_num_pages = (start_pos % this->num_tokens_per_chunk) / this->num_tokens_per_page;
            int64_t chunk_offset = _uid_to_chunks[user_id].first.back();
            chunk_offset += _uid_to_chunks[user_id].second.back() * this->page_bytes;
            chunk_offset -= this->unit_chunk_bytes - partial_num_pages * this->page_bytes;
            // assert();
            chunk_offsets.push_back(chunk_offset);
            chunk_psizes.push_back(this->num_pages_per_chunk - partial_num_pages);

            start_pos += (this->num_pages_per_chunk - partial_num_pages) * this->num_tokens_per_page;
            offload_num_pages -= (this->num_pages_per_chunk - partial_num_pages);
        }

        int padded_num_pages = static_cast<int>(
            std::ceil(static_cast<float>(offload_num_pages) / this->num_pages_per_chunk) * this->num_pages_per_chunk);
        size_t left_num_pages = padded_num_pages;
        while (left_num_pages > 0) {
            // Should always have empty chunks
            auto [chunk_idx, num_unit_chunks] = _empty_chunks.front();
            if (num_unit_chunks * this->num_pages_per_chunk > left_num_pages) {
                _empty_chunks.front().first += left_num_pages / this->num_pages_per_chunk;
                _empty_chunks.front().second -= left_num_pages / this->num_pages_per_chunk;
                chunk_offsets.push_back(static_cast<int64_t>(chunk_idx * this->unit_chunk_bytes));
                chunk_psizes.push_back(left_num_pages);
                left_num_pages = 0;
                break;
            } else {
                _empty_chunks.pop();
                chunk_offsets.push_back(static_cast<int64_t>(chunk_idx * this->unit_chunk_bytes));
                chunk_psizes.push_back(num_unit_chunks * this->num_pages_per_chunk);
                left_num_pages -= num_unit_chunks * this->num_pages_per_chunk;
            }
        }
        _num_empty_chunks -= padded_num_pages / this->num_pages_per_chunk;

        empty_pinned_chunks.push_back(std::make_pair(chunk_offsets, chunk_psizes));
    }

    return empty_pinned_chunks;
}


bool HostKVStorageImpl::offload_kvcache(
    at::Tensor offload_user_ids, // on host
    at::Tensor offload_start_indices,  // on host
    const std::vector<at::Tensor>& offload_page_indices_list, // on host
    KVOffloadHandle& offloadhandle) {
    const auto batch_size = offload_user_ids.size(0);

    at::Device device(torch::kCUDA, static_cast<c10::DeviceIndex>(this->device_idx));
    const c10::cuda::OptionalCUDAGuard device_guard(device);
    c10::cuda::CUDAStream c10_gather_stream =
        c10::cuda::getStreamFromExternal(this->gather_stream, device.index());

    std::vector<int> num_pages_list(batch_size);
    std::vector<int> num_pages_offsets(batch_size);
    std::vector<int> offload_lengths(batch_size);
    for (int i = 0; i < batch_size; i++) {
        num_pages_list[i] = offload_page_indices_list[i].size(0);
        num_pages_offsets[i] = (i == 0) ? 0 : (num_pages_offsets[i - 1] + num_pages_list[i - 1]);
        offload_lengths[i] = num_pages_list[i] * this->num_tokens_per_page;
    }

    offloadhandle.init(
        offload_user_ids,
        offload_start_indices,
        std::move(offload_lengths)
    );

    // [[ contains allocation from torch gpu memory, and memcpy(concat, h2d) ]]
    int64_t cat_dim = 0;
    for (const auto& t : offload_page_indices_list) cat_dim += t.size(0);
    at::Tensor h_offload_page_indices = at::empty(
        {cat_dim},
        at::TensorOptions().device(torch::kCPU).dtype(torch::kInt32).pinned_memory(true)
    );
    at::cat_out(h_offload_page_indices, offload_page_indices_list, 0);

    cudaCheck(cudaEventRecord(offloadhandle.inference_event, at::cuda::getCurrentCUDAStream()));  // ensure the offload kv on gpu is ready
    cudaCheck(cudaStreamWaitEvent(this->gather_stream, offloadhandle.inference_event, 0));

    c10::cuda::CUDAStreamGuard gather_stream_guard(c10_gather_stream);
    at::Tensor offload_page_indices = h_offload_page_indices.to(device, torch::kInt32, true, true);
    
    // Step 0. Find pinned buffers from the pool: 
    //      [[ In Use ]]: Eviction Policy based on user LRU, without pinned-to-unpinned memcpy
    std::vector<std::pair< std::vector<int64_t>, std::vector<int64_t> >> empty_pinned_chunks = this->get_empty_pinned_chunks(
        offload_user_ids,
        offload_start_indices,
        num_pages_list
    );

    // Launch fails if no enough space for offload, even after eviction.
    // When return 0 chunks, no eviction or allocation really happened. 
    if (empty_pinned_chunks.size() == 0) return false;  

    for (int layer_idx = 0; layer_idx < this->num_layers; layer_idx++) {
        char *gpu_offload_buffer = static_cast<char*>(this->offload_gpu_buffers[layer_idx % 2]);  // single layer for a max batch (multi-chunks)
        if (layer_idx >= 2) cudaCheck(cudaStreamWaitEvent(this->gather_stream, offloadhandle.ready_event[layer_idx - 2], 0));

        // Step 1. Gather pages on GPU
        gather_paged_kvcache(reinterpret_cast<uint16_t*>(gpu_offload_buffer), 
            reinterpret_cast<uint16_t*>(this->gpu_cache_table[layer_idx]), 
            offload_page_indices.data_ptr<int>(), 
            this->num_kv_heads, this->kv_headdim, this->num_tokens_per_page, this->page_stride, this->k2v_stride, 
            this->inner_token_stride, this->kv_headdim, offload_page_indices.size(0),
            this->device_num_sms, this->gather_stream);

        // Step 2. cudaEventRecord(offloadhandle.internal_gather_event, this->offload_stream);
        cudaCheck(cudaEventRecord(offloadhandle.internal_gather_event[layer_idx], this->gather_stream));
        // Step 3. cudaMemcpyAsync from GPU cache to pinned buffer
        cudaCheck(cudaStreamWaitEvent(this->offload_stream, offloadhandle.internal_gather_event[layer_idx], 0));
        for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
            auto [chunk_offsets, chunk_psizes] = empty_pinned_chunks[seq_idx];
            size_t offset = num_pages_offsets[seq_idx] * this->page_bytes;
            for (int idx = 0; idx < chunk_offsets.size(); idx++) {
                cudaCheck(cudaMemcpyAsync(
                    static_cast<char*>(this->pinned_kvstorage_buffers[layer_idx]) + chunk_offsets[idx], 
                    gpu_offload_buffer + offset, 
                    chunk_psizes[idx] * this->page_bytes, 
                    cudaMemcpyDeviceToHost, 
                    this->offload_stream
                ));
                offset += chunk_psizes[idx] * this->page_bytes;
            }
        }
        if (layer_idx < this->num_layers - 1) {
            offloadhandle.complete_host(layer_idx, this->offload_stream);
        } else {
            offloadhandle.complete_host(layer_idx, this->offload_stream, std::move(empty_pinned_chunks));   
        }
    }
    // For next offload, gather stream wait for current offload completion to gpu offload buffers are not in used.
    cudaCheck(cudaStreamWaitEvent(this->gather_stream, offloadhandle.ready_event[this->num_layers-1], 0));
    return true;
}

std::vector<int> HostKVStorageImpl::finish_offload(
    KVOffloadHandle& offloadhandle) {

    auto batch_size = offloadhandle.offload_user_ids.size(0);
    std::vector<int> offload_sucess(batch_size, 1);
    
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        int64_t user_id = offloadhandle.offload_user_ids[seq_idx].item<int64_t>();
        auto [chunk_offsets, chunk_psizes] = offloadhandle.chunks[seq_idx];

        auto offload_start_index = offloadhandle.offload_start_indices[seq_idx].item<int64_t>();
        if (offload_start_index != _uid_to_length[user_id]) {
            // give a warning print
            offload_sucess[seq_idx] = 0;
            for (int idx = 0; idx < chunk_offsets.size(); idx++) {
                if (chunk_psizes[idx] < this->num_pages_per_chunk) continue;
                _empty_chunks.push(std::make_pair(chunk_offsets[idx] / this->unit_chunk_bytes, chunk_psizes[idx] / this->num_pages_per_chunk));
                _num_empty_chunks += chunk_psizes[idx] / this->num_pages_per_chunk;
            }
            continue;
        }

        for (int idx = 0; idx < chunk_offsets.size(); idx++) {
            if (chunk_psizes[idx] < this->num_pages_per_chunk) continue;
            _uid_to_chunks[user_id].first.push_back(chunk_offsets[idx]);
            _uid_to_chunks[user_id].second.push_back(chunk_psizes[idx]);
        }
        _uid_to_length[user_id] = offload_start_index + offloadhandle.offload_lengths[seq_idx];
        retain(user_id);
    }

    return offload_sucess;
}

std::vector<int> HostKVStorageImpl::cancel_offload(
    KVOffloadHandle& offloadhandle) {
    // nop for canceling the launched kernels

    auto batch_size = offloadhandle.offload_user_ids.size(0);
    std::vector<int> offload_sucess(batch_size, 0);

    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
        auto [chunk_offsets, chunk_psizes] = offloadhandle.chunks[seq_idx];
        if (chunk_offsets.size() == 0) continue;
        for (int idx = 0; idx < chunk_offsets.size(); idx++) {
            if (chunk_psizes[idx] < this->num_pages_per_chunk) continue;
            _empty_chunks.push(std::make_pair(chunk_offsets[idx] / this->unit_chunk_bytes, chunk_psizes[idx] / this->num_pages_per_chunk));
            _num_empty_chunks += chunk_psizes[idx] / this->num_pages_per_chunk;
        }
    }

    return offload_sucess;
}


void HostKVStorageImpl::evict(int64_t uid) {
    if (_uid_to_chunks.find(uid) == _uid_to_chunks.end()) return;
    const auto &chunk_offsets = _uid_to_chunks[uid].first;
    const auto &chunk_psizes = _uid_to_chunks[uid].second;

    for (size_t idx = 0; idx < chunk_offsets.size(); idx++) {
        _empty_chunks.push(std::make_pair(chunk_offsets[idx] / this->unit_chunk_bytes, chunk_psizes[idx] / this->num_pages_per_chunk));
        _num_empty_chunks += chunk_psizes[idx] / this->num_pages_per_chunk;
    }

    _uid_to_chunks.erase(uid);
    _uid_to_length.erase(uid);
    
    auto const tableIt = _lru_lookup_table.find(uid);
    if (tableIt == _lru_lookup_table.end()) return;
    _lru_list.erase(tableIt->second);
    _lru_lookup_table.erase(tableIt);
}

void HostKVStorageImpl::evict_all() {
    _num_empty_chunks = capacity_per_layer / unit_chunk_bytes;

    std::queue<std::pair<int64_t, int64_t>> chunks;
    chunks.push(std::make_pair(0, _num_empty_chunks));
    std::swap(_empty_chunks, chunks);

    _uid_to_chunks.clear();
    _uid_to_length.clear();
}

bool HostKVStorageImpl::retain(int64_t uid)
{
    auto const tableIt = _lru_lookup_table.find(uid);
    bool found = (_lru_lookup_table.end() != tableIt);
    if (found) {
        _lru_list.erase(tableIt->second);
    }
    _lru_list.push_front(uid);
    _lru_lookup_table[uid] = _lru_list.begin();
    return found;
};


}  // namespace kvcache