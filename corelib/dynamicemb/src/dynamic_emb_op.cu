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

#include <pybind11/pybind11.h>
#include <torch/extension.h>
//#include <torch/python.h>

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/DeviceUtils.cuh"
#include "check.h"
#include "dynamic_variable_base.h"
#include "index_calculation.h"
#include "lookup_backward.h"
#include "lookup_forward.h"
#include "torch_utils.h"
#include "unique_op.h"
#include "utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <torch/torch.h>
#include <cooperative_groups.h>
#include <optional>

namespace py = pybind11;
using namespace dyn_emb;

template <typename T, class = std::enable_if_t<std::is_integral_v<T>>>
inline bool power2(T v) {

  return v && (v & -v) == v;
}

at::Tensor create_sub_tensor(const at::Tensor &original_tensor,
                             int64_t offset) {
  if (offset < 0 || offset >= original_tensor.numel()) {
    throw std::runtime_error("Invalid offset");
  }

  void *data_ptr =
      original_tensor.data_ptr() + offset * original_tensor.element_size();

  int64_t new_size = original_tensor.numel() - offset;

  at::Tensor new_tensor =
      at::from_blob(data_ptr, {new_size}, original_tensor.options());

  return new_tensor;
}

// REMOVE LATER:check result create_sub_tensor correct
void check_sub_tensor(const at::Tensor &original_tensor,
                      const at::Tensor &new_tensor, int64_t offset) {
  void *original_data_ptr = original_tensor.data_ptr();

  void *new_data_ptr = new_tensor.data_ptr();

  std::cout << "Original tensor data pointer: " << original_data_ptr
            << std::endl;
  std::cout << "New tensor data pointer: " << new_data_ptr << std::endl;

  void *expected_new_data_ptr = static_cast<char *>(original_data_ptr) +
                                offset * original_tensor.element_size();

  if (new_data_ptr == expected_new_data_ptr) {
    std::cout << "The new tensor data pointer is correctly referencing the "
                 "original tensor's memory."
              << std::endl;
  } else {
    std::cout << "The new tensor data pointer is NOT correctly referencing the "
                 "original tensor's memory."
              << std::endl;
  }
}

// Dyn_emb API
// TODO all the API need check datatype and dimension continuous
int64_t dyn_emb_rows(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  return table->rows(stream);
}

int64_t dyn_emb_cols(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  return table->cols();
}

int64_t dyn_emb_capacity(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  return table->capacity();
}

void insert_or_assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                      const size_t n, const at::Tensor keys,
                      const at::Tensor values,
                      const c10::optional<at::Tensor> &score = c10::nullopt,
                      bool unique_key = true,
                      bool ignore_evict_strategy = false) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->insert_or_assign(n, keys.data_ptr(), values.data_ptr(),
                            score_.data_ptr(), stream, unique_key,
                            ignore_evict_strategy);
  } else {
    table->insert_or_assign(n, keys.data_ptr(), values.data_ptr(), nullptr,
                            stream, unique_key, ignore_evict_strategy);
  }
}

// If don't need input scores, `scores` can be set to std::nullopt.
void insert_and_evict(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const size_t n,
    const at::Tensor keys,
    const at::Tensor values,
    const std::optional<uint64_t> score,
    at::Tensor evicted_keys,
    at::Tensor evicted_values,
    at::Tensor evicted_score,
    at::Tensor d_evicted_counter,
    bool unique_key = true,
    bool ignore_evict_strategy = false) {

  if (not score and (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
  }
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    // broadcast scores
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
    bc_scores.fill_(score.value());
    table->insert_and_evict(
      n, keys.data_ptr(), values.data_ptr(), bc_scores.data_ptr(),
      evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
      reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
  } else {
    table->insert_and_evict(
      n, keys.data_ptr(), values.data_ptr(), nullptr, 
      evicted_keys.data_ptr(), evicted_values.data_ptr(), evicted_score.data_ptr(),
      reinterpret_cast<uint64_t*>(d_evicted_counter.data_ptr()), stream, unique_key, ignore_evict_strategy);
  }
}

void accum_or_assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                     const size_t n, const at::Tensor keys,
                     const at::Tensor value_or_deltas,
                     const at::Tensor accum_or_assigns,
                     const c10::optional<at::Tensor> &score = c10::nullopt,
                     bool ignore_evict_strategy = false) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(),
                           accum_or_assigns.data_ptr<bool>(), score_.data_ptr(),
                           stream, ignore_evict_strategy);
  } else {
    table->accum_or_assign(n, keys.data_ptr(), value_or_deltas.data_ptr(),
                           accum_or_assigns.data_ptr<bool>(), nullptr, stream,
                           ignore_evict_strategy);
  }
}


void find_and_initialize(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const size_t n,
    const at::Tensor keys,
    const at::Tensor values,
    std::optional<InitializerArgs> initializer_args) {

  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor vals_ptr_tensor = at::empty({static_cast<int64_t>(n)}, 
    at::TensorOptions().dtype(at::kLong).device(values.device()));
  auto vals_ptr = reinterpret_cast<void**>(vals_ptr_tensor.data_ptr<int64_t>());
  at::Tensor founds_tensor = at::empty({static_cast<int64_t>(n)},
     at::TensorOptions().dtype(at::kBool).device(keys.device()));
  auto founds = founds_tensor.data_ptr<bool>();

  table->find_and_initialize(n, keys.data_ptr(), vals_ptr, values.data_ptr(), founds, initializer_args, stream);
}

void find_or_insert(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                  const size_t n,
                  const at::Tensor keys,
                  const at::Tensor values,
                  const std::optional<uint64_t> score = std::nullopt,
                  bool unique_key = true,
                  bool ignore_evict_strategy = false
                  )
{
  if (not score and (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
  }
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor new_tensor = at::empty({static_cast<int64_t>(n)},
                                    at::TensorOptions().dtype(at::kLong).device(values.device()));

  auto new_tensor_data_ptr = reinterpret_cast<void**>(new_tensor.data_ptr<int64_t>());

  at::Tensor found_tensor = at::empty({static_cast<int64_t>(n)},
                                      at::TensorOptions().dtype(at::kBool).device(keys.device()));

  auto found_tensor_data_ptr = found_tensor.data_ptr<bool>();

  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    // broadcast scores
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
    bc_scores.fill_(score.value());
    table->find_or_insert(n, keys.data_ptr(), new_tensor_data_ptr, values.data_ptr(), found_tensor_data_ptr,
                              bc_scores.data_ptr(), stream, unique_key, ignore_evict_strategy);

  } else {
    table->find_or_insert(n, keys.data_ptr(), new_tensor_data_ptr, values.data_ptr(), found_tensor_data_ptr, nullptr,
                              stream, unique_key, ignore_evict_strategy);
  }
}

void find_or_insert_pointers(
  std::shared_ptr<dyn_emb::DynamicVariableBase> table,
  const size_t n,
  const at::Tensor keys,
  at::Tensor values,
  at::Tensor founds,
  const std::optional<uint64_t> score = std::nullopt,
  bool unique_key = true,
  bool ignore_evict_strategy = false) {
  if (not score and (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu)) {
    throw std::invalid_argument("Must specify the score when evict strategy is customized or LFU.");
  }
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto values_data_ptr = reinterpret_cast<void**>(values.data_ptr<int64_t>());
  auto found_tensor_data_ptr = founds.data_ptr<bool>();

  if (table->evict_strategy() == EvictStrategy::kCustomized || table->evict_strategy() == EvictStrategy::kLfu) {
    auto&& option = at::TensorOptions().dtype(at::kUInt64).device(keys.device());
    // broadcast scores
    at::Tensor bc_scores = at::empty({static_cast<int64_t>(n)}, option);
    bc_scores.fill_(score.value());
    table->find_or_insert_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, 
      bc_scores.data_ptr(), stream, unique_key, ignore_evict_strategy);
  } else {
    table->find_or_insert_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, 
      nullptr, stream, unique_key, ignore_evict_strategy);
  }
  AT_CUDA_CHECK(cudaGetLastError());
}

void find_pointers(
  std::shared_ptr<dyn_emb::DynamicVariableBase> table,
  const size_t n,
  const at::Tensor keys,
  at::Tensor values,
  at::Tensor founds) {

  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto values_data_ptr = reinterpret_cast<void**>(values.data_ptr<int64_t>());
  auto found_tensor_data_ptr = founds.data_ptr<bool>();

  table->find_pointers(n, keys.data_ptr(), values_data_ptr, found_tensor_data_ptr, 
      nullptr, stream);
}

void assign(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
            const at::Tensor keys, const at::Tensor values,
            const c10::optional<at::Tensor> &score = c10::nullopt,
            bool unique_key = true) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->assign(n, keys.data_ptr(), values.data_ptr(), score_.data_ptr(),
                  stream, unique_key);
  } else {
    table->assign(n, keys.data_ptr(), values.data_ptr(), nullptr, stream,
                  unique_key);
  }
}

void find(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
          const at::Tensor keys, const at::Tensor values,
          const at::Tensor founds,
          const c10::optional<at::Tensor> &score = c10::nullopt) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(),
                score_.data_ptr(), stream);
  } else {
    table->find(n, keys.data_ptr(), values.data_ptr(), founds.data_ptr<bool>(),
                nullptr, stream);
    std::cout << "keys" << keys << std::endl;
    std::cout << "values" << values << std::endl;
    std::cout << "founds" << founds << std::endl;
  }
}

void erase(std::shared_ptr<dyn_emb::DynamicVariableBase> table, const size_t n,
           const at::Tensor keys) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  table->erase(n, keys.data_ptr(), stream);
}

void clear(std::shared_ptr<dyn_emb::DynamicVariableBase> table) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->clear(stream);
}

void reserve(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
             const size_t new_capacity) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->reserve(new_capacity, stream);
}

void export_batch(std::shared_ptr<dyn_emb::DynamicVariableBase> table,
                  const size_t n, const size_t offset,
                  const at::Tensor d_counter, const at::Tensor keys,
                  const at::Tensor values,
                  const c10::optional<at::Tensor> &score = c10::nullopt) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  if (score.has_value()) {
    at::Tensor score_ = score.value();
    table->export_batch(n, offset, d_counter.data_ptr<size_t>(),
                        keys.data_ptr(), values.data_ptr(), score_.data_ptr(),
                        stream);
  } else {
    table->export_batch(n, offset, d_counter.data_ptr<size_t>(),
                        keys.data_ptr(), values.data_ptr(), nullptr, stream);
  }
}

void count_matched(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const uint64_t threshold,
    at::Tensor num_matched) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->count_matched(threshold, reinterpret_cast<uint64_t*>(num_matched.data_ptr()), stream);
}

void export_batch_matched(
    std::shared_ptr<dyn_emb::DynamicVariableBase> table,
    const uint64_t threshold,
    const uint64_t n,
    const uint64_t offset,
    at::Tensor num_matched,
    at::Tensor keys,
    at::Tensor values) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  table->export_batch_matched(
    threshold, n, offset, reinterpret_cast<uint64_t*>(num_matched.data_ptr()), 
    keys.data_ptr(), values.data_ptr(), nullptr, stream);
}

template <typename scalar_t>
__global__ void compact_offsets(
  const scalar_t *offsets,
  scalar_t *features_offsets,
  const int64_t num_features,
  const int64_t batch_size
) { 
  for (int tid = threadIdx.x + blockIdx.x * blockDim.x; tid < num_features; tid += blockDim.x * gridDim.x) {
    features_offsets[tid] = offsets[tid * batch_size];
  }
  if (threadIdx.x == 0) {
    features_offsets[num_features] = offsets[num_features * batch_size];
  }
}

std::vector<int64_t> offsets_to_table_features_offsets(const at::Tensor &offsets, const std::vector<int> &table_offsets_in_feature, const int64_t batch_size, cudaStream_t stream) {
  int64_t table_num = table_offsets_in_feature.size() - 1;
  int64_t num_features = (offsets.numel() - 1) / batch_size;
  at::Tensor h_features_offsets =
      at::empty({num_features + 1}, offsets.options().device(at::kCPU).pinned_memory(true));
  if (num_features == 0) {
    return {0, 0};
  }
  AT_DISPATCH_INTEGRAL_TYPES(offsets.scalar_type(), "compact_offsets", [&] {
    compact_offsets<<<num_features / 1024 + 1, 1024, 0, stream>>>(
      offsets.data_ptr<scalar_t>(),
      h_features_offsets.data_ptr<scalar_t>(),
      num_features,
      batch_size
    );
  });
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  std::vector<int64_t> table_features_offsets(table_offsets_in_feature.size(), 0);
  for (int i = 0; i < table_offsets_in_feature.size(); ++i) {
    table_features_offsets[i] = h_features_offsets[table_offsets_in_feature[i]].item<int64_t>();
  }
  return table_features_offsets;
}

void lookup_forward_dense(
    std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>> tables,
    const at::Tensor indices, const at::Tensor offsets, const py::list scores,
    const std::vector<int> &table_offsets_in_feature, at::Tensor table_offsets,
    int table_num, int batch_size, int dim, bool use_index_dedup,
    const at::Tensor unique_idx, const at::Tensor reverse_idx,
    const at::Tensor h_unique_nums, const at::Tensor d_unique_nums,
    const at::Tensor h_unique_offsets, const at::Tensor d_unique_offsets,
    const at::Tensor unique_embs, const at::Tensor output_embs,
    int device_num_sms, std::shared_ptr<dyn_emb::UniqueOpBase> unique_op) {

  if (!offsets.is_cuda() || !indices.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  // Check dtype of h_unique_nums and d_unique_nums
  if (h_unique_nums.scalar_type() != at::kUInt64 ||
      d_unique_nums.scalar_type() != at::kUInt64) {
    throw std::runtime_error(
        "h_unique_nums and d_unique_nums must have dtype uint64_t");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t indices_shape = indices.numel();
  auto unique_num_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_nums.dtype()));
  auto unique_offset_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_offsets.dtype()));

  auto h_table_offsets = offsets_to_table_features_offsets(offsets, table_offsets_in_feature, batch_size, stream);
  
  size_t unique_op_capacity = unique_op->get_capacity();
  if (indices_shape * 2 > unique_op_capacity) {
    at::Tensor new_keys = at::empty({indices_shape * 2}, indices.options());
    at::Tensor new_vals = at::empty(
        {indices_shape * 2},
        at::TensorOptions().dtype(at::kUInt64).device(indices.device()));
    unique_op->reset_capacity(new_keys, new_vals, indices_shape * 2, stream);
  }

  std::vector<at::Tensor> tmp_unique_indices(table_num);
  for (int i = 0; i < table_num; ++i) {
    tmp_unique_indices[i] = at::empty_like(indices);
  }

  for (int i = 0; i < table_num; ++i) {
    int64_t indices_begin = h_table_offsets[i];
    int64_t indices_end = h_table_offsets[i + 1];
    int64_t indices_length = indices_end - indices_begin;

    if (indices_length == 0) {
      DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<uint64_t *>(d_unique_nums.data_ptr()) + i, 0,
          sizeof(uint64_t), stream));
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    } else {
      at::Tensor tmp_indices = create_sub_tensor(indices, indices_begin);
      at::Tensor tmp_reverse_idx =
          create_sub_tensor(reverse_idx, indices_begin);
      at::Tensor tmp_d_unique_num = create_sub_tensor(d_unique_nums, i);

      at::Tensor previous_d_unique_num = create_sub_tensor(d_unique_offsets, i);
      unique_op->unique(tmp_indices, indices_length, tmp_reverse_idx,
                        tmp_unique_indices[i], tmp_d_unique_num, stream,
                        previous_d_unique_num);
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    }
  }

  AT_CUDA_CHECK(
      cudaMemcpyAsync(h_unique_nums.data_ptr(), d_unique_nums.data_ptr(),
                      d_unique_nums.numel() * d_unique_nums.element_size(),
                      cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      h_unique_offsets.data_ptr(), d_unique_offsets.data_ptr(),
      (d_unique_nums.numel() + 1) * d_unique_nums.element_size(),
      cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  AT_CUDA_CHECK(
      cudaMemcpyAsync(table_offsets.data_ptr(), h_table_offsets.data(),
                      table_offsets.numel() * table_offsets.element_size(),
                      cudaMemcpyHostToDevice, stream));

  int64_t unique_embs_offset = 0;
  for (int i = 0; i < table_num; ++i) {
    int64_t tmp_unique_num = h_unique_nums[i].item<int64_t>();
    if (tmp_unique_num != 0) {
      at::Tensor tmp_unique_embs =
          create_sub_tensor(unique_embs, unique_embs_offset * dim);
      auto score = std::make_optional<uint64_t>(py::cast<uint64_t>(scores[i]));
      find_or_insert(tables[i], tmp_unique_num, tmp_unique_indices[i],
                    tmp_unique_embs, score);
      if (use_index_dedup) {
        void *dst_ptr = reinterpret_cast<char *>(unique_idx.data_ptr()) +
                        unique_embs_offset * unique_idx.element_size();
        void *src_ptr = tmp_unique_indices[i].data_ptr();
        size_t copy_size = tmp_unique_num * unique_idx.element_size();
        AT_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, copy_size,
                                      cudaMemcpyDeviceToDevice, stream));
      }
    }
    unique_embs_offset += tmp_unique_num;
  }
  auto src_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(unique_embs.dtype()));
  auto dst_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(output_embs.dtype()));
  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(reverse_idx.dtype()));

  dyn_emb::scatter_fused(unique_embs.data_ptr(), output_embs.data_ptr(),
                         reverse_idx.data_ptr(), indices_shape, dim, src_type,
                         dst_type, offset_type, device_num_sms, stream);
}

at::Tensor lookup_forward_dense_eval(
    std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>> tables,
    const at::Tensor &indices,
    const at::Tensor &offsets,
    const std::vector<int> &table_offsets_in_feature,
    at::ScalarType embedding_dtype,
    int table_num,
    int batch_size,
    int dim,
    const at::Device& device,
    const std::vector<InitializerArgs> &eval_initializers) {

  if (!indices.is_cuda() || !offsets.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_indices = indices.numel();

  at::Tensor output_embs = at::empty({num_indices, dim}, at::TensorOptions().dtype(embedding_dtype).device(device));

  auto table_features_offsets = offsets_to_table_features_offsets(offsets, table_offsets_in_feature, batch_size, stream);
  
  for (int i = 0; i < table_num; ++i) {
    int64_t table_offset_begin = table_features_offsets[i];
    int64_t table_offset_end = table_features_offsets[i + 1];
    int64_t table_offset_length = table_offset_end - table_offset_begin;
    at::Tensor current_indices = create_sub_tensor(indices, table_offset_begin);
    at::Tensor current_output_embs = create_sub_tensor(output_embs, table_offset_begin * dim);
    
    find_and_initialize(tables[i], static_cast<size_t>(table_offset_length), current_indices, current_output_embs, eval_initializers[i]);
  }

  return output_embs;
}

void lookup_backward_dense(const at::Tensor indices, const at::Tensor grads,
                           int32_t dim, const at::Tensor table_offsets,
                           at::Tensor unique_indices, at::Tensor unique_grads) {
  // Doc for dynamic embedding's backward:
  //   Step 1: using SegmentedSortDevice to sort the indices per table.
  //   Step 2: using SegmentedUniqueDevice to dedup the indices per table.
  //   Step 3: using 2-stage reduction to reduce the gradients.

  // Initialization
  if (!indices.is_cuda() || !grads.is_cuda() || !table_offsets.is_cuda() ||
      !table_offsets.is_cuda() || !unique_indices.is_cuda() ||
      !unique_grads.is_cuda()) {
    throw std::runtime_error("All argument tensors should on device");
  }
  auto device_ = indices.device();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  // Number of tables should <= 2^31-1
  int32_t table_num = static_cast<int32_t>(table_offsets.size(0) - 1);
  auto scalar_type = indices.dtype().toScalarType();
  auto key_type = scalartype_to_datatype(scalar_type);
  auto id_stype = table_offsets.dtype().toScalarType(); // scalar type
  auto id_dtype = scalartype_to_datatype(id_stype);     // data type
  auto key_num = indices.size(0);

  // Step 1: using SegmentedSortDevice to sort the indices by table.
  SegmentedSortDevice sort_op =
      SegmentedSortDevice(device_, key_num, table_num, key_type, id_dtype);
  auto original_ids =
      at::empty_like(indices, indices.options().dtype(id_stype));
  auto sorted_keys = at::empty_like(indices, indices.options());
  auto sorted_key_ids =
      at::empty_like(indices, indices.options().dtype(id_stype));
  auto sorted_table_ids =
      at::empty_like(indices, indices.options().dtype(at::kInt));
  sort_op(indices, original_ids, table_offsets, sorted_keys, sorted_key_ids,
          sorted_table_ids, stream, true, true);

  // Step 2: using SegmentedUniqueDevice to dedup the indices by table.
  SegmentedUniqueDevice unique_op =
      SegmentedUniqueDevice(device_, key_num, key_type, id_dtype);
  auto unique_key_ids =
      at::empty_like(indices, indices.options().dtype(id_stype));
  unique_op(sorted_keys, sorted_table_ids, unique_indices, unique_key_ids,
            stream);

  // Step 3: using 2-stage reduction to reduce the gradients.
  LocalReduce localReduceOp(device_, key_num, dim, id_dtype, DataType::Float32);
  localReduceOp.local_reduce(grads, unique_grads, sorted_key_ids,
                             unique_key_ids, stream);
}

void lookup_backward_dense_dedup(const at::Tensor grads,
                                 at::Tensor unique_indices,
                                 at::Tensor reverse_idx, int32_t dim,
                                 at::Tensor unique_grads,
                                 int32_t device_num_sms) {
  // Initialization
  if (!grads.is_cuda() || !unique_indices.is_cuda() || !reverse_idx.is_cuda() ||
      !unique_grads.is_cuda()) {
    throw std::runtime_error("All argument tensors should on device");
  }
  auto device_ = unique_indices.device();
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto rev_idx_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(reverse_idx.dtype()));
  auto grad_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(grads.dtype()));
  auto idx_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(unique_indices.dtype()));
  auto key_num = reverse_idx.size(0);
  auto unique_key_num = unique_indices.size(0);

  dyn_emb::one_to_one_atomic(grads.data_ptr(), unique_indices.data_ptr(),
                             reverse_idx.data_ptr(), unique_grads.data_ptr(),
                             dim, key_num, unique_key_num, rev_idx_type,
                             grad_type, idx_type, device_num_sms, stream);
}

void dedup_input_indices(
    const at::Tensor indices, const at::Tensor offsets,
    const at::Tensor h_table_offsets_in_feature,
    const at::Tensor d_table_offsets_in_feature, int table_num,
    int local_batch_size, const at::Tensor reverse_idx,
    const at::Tensor h_unique_nums, const at::Tensor d_unique_nums,
    const at::Tensor h_unique_offsets, const at::Tensor d_unique_offsets,
    std::vector<at::Tensor> unique_idx, const at::Tensor new_offsets,
    const at::Tensor new_lengths, int device_num_sms,
    std::shared_ptr<dyn_emb::UniqueOpBase> unique_op) {

  if (!offsets.is_cuda() || !indices.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  // Check dtype of h_unique_nums and d_unique_nums
  if (h_unique_nums.scalar_type() != at::kUInt64 ||
      d_unique_nums.scalar_type() != at::kUInt64) {
    throw std::runtime_error(
        "h_unique_nums and d_unique_nums must have dtype uint64_t");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t indices_shape = indices.size(0);
  auto unique_num_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_nums.dtype()));
  auto unique_offset_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_offsets.dtype()));
  int64_t new_lengths_size = new_lengths.size(0);

  at::Tensor h_offset =
      at::empty_like(offsets, offsets.options().device(at::kCPU));
  AT_CUDA_CHECK(cudaMemcpyAsync(h_offset.data_ptr(), offsets.data_ptr(),
                                offsets.numel() * offsets.element_size(),
                                cudaMemcpyDeviceToHost, stream));

  size_t unique_op_capacity = unique_op->get_capacity();
  if (indices_shape * 2 > unique_op_capacity) {
    at::Tensor new_keys = at::empty({indices_shape * 2}, indices.options());
    at::Tensor new_vals = at::empty(
        {indices_shape * 2},
        at::TensorOptions().dtype(at::kUInt64).device(indices.device()));
    unique_op->reset_capacity(new_keys, new_vals, indices_shape * 2, stream);
  }

  std::vector<at::Tensor> tmp_unique_indices(table_num);
  for (int i = 0; i < table_num; ++i) {
    tmp_unique_indices[i] = at::empty_like(indices);
  }

  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int i = 0; i < table_num; ++i) {
    int table_offset_begin = h_table_offsets_in_feature[i].item<int64_t>();
    int table_offset_end = h_table_offsets_in_feature[i + 1].item<int64_t>();
    int offset_begin = table_offset_begin * local_batch_size;
    int offset_end = table_offset_end * local_batch_size;

    int64_t indices_begin = h_offset[offset_begin].item<int64_t>();
    int64_t indices_end = h_offset[offset_end].item<int64_t>();
    int64_t indices_length = indices_end - indices_begin;

    if (indices_length == 0) {
      DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<uint64_t *>(d_unique_nums.data_ptr()) + i, 0,
          sizeof(uint64_t), stream));
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    } else {
      at::Tensor tmp_indices = create_sub_tensor(indices, indices_begin);
      at::Tensor tmp_reverse_idx =
          create_sub_tensor(reverse_idx, indices_begin);
      at::Tensor tmp_d_unique_num = create_sub_tensor(d_unique_nums, i);
      at::Tensor previous_d_unique_num = create_sub_tensor(d_unique_offsets, i);

      unique_op->unique(tmp_indices, indices_length, tmp_reverse_idx,
                        unique_idx[i], tmp_d_unique_num, stream,
                        previous_d_unique_num);
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    }
  }

  AT_CUDA_CHECK(
      cudaMemcpyAsync(h_unique_nums.data_ptr(), d_unique_nums.data_ptr(),
                      d_unique_nums.numel() * d_unique_nums.element_size(),
                      cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaMemcpyAsync(
      h_unique_offsets.data_ptr(), d_unique_offsets.data_ptr(),
      d_unique_offsets.numel() * d_unique_offsets.element_size(),
      cudaMemcpyDeviceToHost, stream));

  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(new_offsets.dtype()));
  auto lengths_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(new_lengths.dtype()));

  get_new_length_and_offsets(
      reinterpret_cast<uint64_t *>(d_unique_offsets.data_ptr()),
      d_table_offsets_in_feature.data_ptr<int64_t>(), table_num,
      new_lengths_size, local_batch_size, lengths_type, offset_type,
      new_offsets.data_ptr(), new_lengths.data_ptr(), stream);
}

void lookup_forward(const at::Tensor src, const at::Tensor dst,
                    const at::Tensor offset, const at::Tensor inverse_idx,
                    int combiner, int total_D, int accum_D, int ev_size,
                    int num_vec, int batch_size, int device_num_sms) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto src_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(src.dtype()));
  auto dst_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(dst.dtype()));
  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(offset.dtype()));
  if (combiner == -1) { // sequence
    auto &&num_emb = inverse_idx.size(0);
    dyn_emb::scatter(src.data_ptr(), dst.data_ptr(), offset.data_ptr(),
                     inverse_idx.data_ptr(), num_emb, ev_size, src_type,
                     dst_type, offset_type, device_num_sms, stream);
  } else {
    dyn_emb::scatter_combine(src.data_ptr(), dst.data_ptr(), offset.data_ptr(),
                             inverse_idx.data_ptr(), combiner, total_D, accum_D,
                             ev_size, num_vec, batch_size, src_type, dst_type,
                             offset_type, stream);
  }
}

void lookup_backward(const at::Tensor grad, const at::Tensor unique_buffer,
                     const at::Tensor unique_indices,
                     const at::Tensor inverse_indices,
                     const at::Tensor biased_offsets, const int dim,
                     const int table_num, int batch_size, int feature_num,
                     int num_key) {

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto value_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(unique_buffer.dtype()));
  auto key_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(unique_indices.dtype()));
  dyn_emb::backward(grad.data_ptr(), unique_buffer.data_ptr(),
                    unique_indices.data_ptr(), inverse_indices.data_ptr(),
                    biased_offsets.data_ptr(), dim, batch_size, feature_num,
                    num_key, key_type, value_type, stream);
}

// PYTHON WARP
void bind_dyn_emb_op(py::module &m) {
  py::class_<dyn_emb::InitializerArgs>(m, "InitializerArgs")
    .def(py::init([] (const std::string& mode, float mean, float std_dev, float lower, float upper, float value) {
      return dyn_emb::InitializerArgs(mode, mean, std_dev, lower, upper, value);
    }
    ))
    .def(py::pickle(
      [](const InitializerArgs &p) { // __getstate__
        return py::make_tuple(p.mode, p.mean, p.std_dev, p.lower, p.upper, p.value);
      },
      [](py::tuple t) { // __setstate__
        if (t.size() != 6)
          throw std::runtime_error("Invalid number args of InitializerArgs!");
        InitializerArgs p(
          t[0].cast<std::string>(),
          t[1].cast<float>(),
          t[2].cast<float>(),
          t[3].cast<float>(),
          t[4].cast<float>(),
          t[5].cast<float>());
        return p;
      }
     ));
    py::class_<dyn_emb::DynamicVariableBase, std::shared_ptr<dyn_emb::DynamicVariableBase>>(m, "DynamicEmbTable")
        .def(py::init([](dyn_emb::DataType key_type,
					dyn_emb::DataType value_type, 
					dyn_emb::EvictStrategy evict_type,
					int64_t dim = 128,
					int64_t init_capaity = 1024,
					int64_t max_capaity = 2048,
					size_t max_hbm_for_vectors = 0, 
					size_t max_bucket_size  = 128,
					float max_load_factor = 0.5,
					int block_size = 128,
					int io_block_size = 1024, 
					int device_id = -1, 
					bool io_by_cpu = false,
					bool use_constant_memory = false,
					int reserved_key_start_bit = 0,
					size_t num_of_buckets_per_alloc = 1,
					const dyn_emb::InitializerArgs & initializer_args = dyn_emb::InitializerArgs(),
          const int safe_check_mode = static_cast<int>(SafeCheckMode::IGNORE),
          const int optimizer_type = static_cast<int>(OptimizerType::Null)) {

            int64_t pow2_max_capaity = power2(max_capaity);
            int64_t pow2_init_capaity = power2(init_capaity);
            auto table = dyn_emb::VariableFactory::create(key_type,value_type,evict_type,dim,init_capaity,max_capaity,max_hbm_for_vectors,max_bucket_size,max_load_factor,
                                 block_size,io_block_size,device_id,io_by_cpu,use_constant_memory,reserved_key_start_bit,num_of_buckets_per_alloc,initializer_args, 
                                 static_cast<SafeCheckMode>(safe_check_mode), static_cast<OptimizerType>(optimizer_type));
            return table; }))
         .def("key_type", &dyn_emb::DynamicVariableBase::key_type,
             "Get Dynamic Emb Table key type")
         .def("value_type", &dyn_emb::DynamicVariableBase::value_type,
             "Get Dynamic Emb Table value type")
          .def("evict_strategy", &dyn_emb::DynamicVariableBase::evict_strategy,
            "Get evict strategy of Dynamic Emb Table.")
          .def("capacity", &dyn_emb::DynamicVariableBase::capacity,
            "Get capacity of Dynamic Emb Table.")
          .def("optstate_dim", &dyn_emb::DynamicVariableBase::optstate_dim,
            "Get dim of all optimizer states.")
          .def("set_initial_optstate", &dyn_emb::DynamicVariableBase::set_initial_optstate,
            "Set initial value of optimizer state.")
          .def("get_initial_optstate", &dyn_emb::DynamicVariableBase::get_initial_optstate,
            "Get initial value of optimizer state.");

  m.def("dyn_emb_rows", &dyn_emb_rows, "Get the number of rows in the table",
        py::arg("table"));

  m.def("dyn_emb_cols", &dyn_emb_cols, "Get the number of columns in the table",
        py::arg("table"));

  m.def("dyn_emb_capacity", &dyn_emb_capacity,
        "Get the capacity in the dynamic table", py::arg("table"));

  m.def("insert_or_assign", &insert_or_assign,
        "Insert or assign a key-value pair in the table", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("score") = c10::nullopt, py::arg("unique_key") = true,
        py::arg("ignore_evict_strategy") = false);

  m.def("insert_and_evict", &insert_and_evict,
        "Insert keys and values, evicting if necessary", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("score"),
        py::arg("evicted_keys"), py::arg("evicted_values"),
        py::arg("evicted_score"), py::arg("d_evicted_counter"),
        py::arg("unique_key") = true, py::arg("ignore_evict_strategy") = false);

  m.def("accum_or_assign", &accum_or_assign,
        "Accumulate or assign values to the table", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("value_or_deltas"),
        py::arg("accum_or_assigns"), py::arg("score") = c10::nullopt,
        py::arg("ignore_evict_strategy") = false);
  
  m.def("find_and_initialize", &find_and_initialize,
        "Find and initialize a key-value pair in the table", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("values"), 
        py::arg("initializer_args") = py::none());

  m.def("find_or_insert", &find_or_insert,
        "Find or insert a key-value pair in the table", py::arg("table"),
        py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("score") = py::none(), py::arg("unique_key") = true, 
        py::arg("ignore_evict_strategy") = false);

  m.def("find_or_insert_pointers", &find_or_insert_pointers,
        "Find or insert a key-value pair in the table , and return every "
        "value's ptr",
        py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"), py::arg("founds"),
        py::arg("score") = py::none(), py::arg("unique_key") = true, 
        py::arg("ignore_evict_strategy") = false);

  m.def("assign", &assign, "Assign values to the table based on keys",
        py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("score") = c10::nullopt, py::arg("unique_key") = true);

  m.def("find", &find, "Find values in the table based on keys",
        py::arg("table"), py::arg("n"), py::arg("keys"), py::arg("values"),
        py::arg("founds"), py::arg("score") = c10::nullopt);

  m.def("erase", &erase, "Erase values from the table based on keys",
        py::arg("table"), py::arg("n"), py::arg("keys"));

  m.def("reserve", &reserve, "reserve hash table capacity", py::arg("table"),
        py::arg("new_capacity"));


  py::enum_<dyn_emb::DataType>(m, "DynamicEmbDataType")
      .value("Float32", dyn_emb::DataType::Float32)
      .value("BFloat16", dyn_emb::DataType::BFloat16)
      .value("Float16", dyn_emb::DataType::Float16)
      .value("Int64", dyn_emb::DataType::Int64)
      .value("UInt64", dyn_emb::DataType::UInt64)
      .value("Int32", dyn_emb::DataType::Int32)
      .value("UInt32", dyn_emb::DataType::UInt32)
      .value("Size_t", dyn_emb::DataType::Size_t)
      .export_values();
    m.def("clear", &clear,
          "Clear all keys in the table",
          py::arg("table"));

    m.def("reserve", &reserve,
          "reserve hash table capacity",
          py::arg("table"),
          py::arg("new_capacity"));

    m.def("export_batch", &export_batch,
          "export key value from table",
          py::arg("table"),
          py::arg("n"),
          py::arg("offset"),
          py::arg("d_counter"),
          py::arg("keys"),
          py::arg("values"),
          py::arg("score") = c10::nullopt);
    
    m.def("count_matched", &count_matched, 
      "Count the KV-pairs whose score > threshold in the whole table.",
      py::arg("table"),
      py::arg("threshold"),
      py::arg("num_matched"));

    m.def("export_batch_matched", &export_batch_matched,
      "Export KV-pairs within [offset, offset + n) whose score > threshold",
      py::arg("table"),
      py::arg("threshold"),
      py::arg("n"),
      py::arg("offset"),
      py::arg("num_matched"),
      py::arg("keys"),
      py::arg("values"));

  py::enum_<dyn_emb::EvictStrategy>(m, "EvictStrategy")
      .value("KLru", dyn_emb::EvictStrategy::kLru)
      .value("KLfu", dyn_emb::EvictStrategy::kLfu)
      .value("KEpochLru", dyn_emb::EvictStrategy::kEpochLru)
      .value("KEpochLfu", dyn_emb::EvictStrategy::kEpochLfu)
      .value("KCustomized", dyn_emb::EvictStrategy::kCustomized)
      .export_values();

  py::enum_<dyn_emb::OptimizerType>(m, "OptimizerType")
    .value("Null", dyn_emb::OptimizerType::Null)
    .value("SGD", dyn_emb::OptimizerType::SGD)
    .value("Adam", dyn_emb::OptimizerType::Adam)
    .value("AdaGrad", dyn_emb::OptimizerType::AdaGrad)
    .value("RowWiseAdaGrad", dyn_emb::OptimizerType::RowWiseAdaGrad)
    .export_values();

  m.def("lookup_forward", &lookup_forward, "scatter and combine",
        py::arg("src"), py::arg("dst"), py::arg("offset"),
        py::arg("inverse_idx"), py::arg("combiner"), py::arg("total_D"),
        py::arg("accum_D"), py::arg("ev_size"), py::arg("num_vec"),
        py::arg("batch_size"), py::arg("device_num_sms"));

  m.def("lookup_backward", &lookup_backward, "backward", py::arg("grad"),
        py::arg("unique_buffer"), py::arg("unique_indices"),
        py::arg("inverse_indices"), py::arg("biased_offsets"), py::arg("dim"),
        py::arg("tables_num"), py::arg("batch_size"), py::arg("num_feature"),
        py::arg("num_key"));

  m.def("lookup_forward_dense", &lookup_forward_dense,
        "lookup forward dense for duplicated keys", py::arg("tables"),
        py::arg("indices"), py::arg("offsets"), py::arg("scores"),
        py::arg("table_offsets_in_feature"), py::arg("table_offsets"),
        py::arg("table_num"), py::arg("batch_size"), py::arg("dim"),
        py::arg("use_index_dedup"), py::arg("unique_idx"),
        py::arg("reverse_idx"), py::arg("h_unique_nums"),
        py::arg("d_unique_nums"), py::arg("h_unique_offsets"),
        py::arg("d_unique_offsets"), py::arg("unique_embs"),
        py::arg("output_embs"), py::arg("device_num_sms"),
        py::arg("unique_op"));

  m.def("lookup_forward_dense_eval", &lookup_forward_dense_eval,
        "lookup forward dense eval for globally deduplicated keys", py::arg("tables"),
        py::arg("indices"), py::arg("offsets"), py::arg("table_offsets_in_feature"),
        py::arg("embedding_dtype"), py::arg("table_num"), py::arg("batch_size"),
        py::arg("dim"), py::arg("device"), py::arg("eval_initializers"));

  m.def("lookup_backward_dense", &lookup_backward_dense,
        "lookup backward for dense/sequence", py::arg("indices"),
        py::arg("grads"), py::arg("dim"), py::arg("table_offsets"),
        py::arg("unique_indices"), py::arg("unique_grads"));

  m.def("lookup_backward_dense_dedup", &lookup_backward_dense_dedup,
        "lookup backward for dedup dense/sequence", py::arg("grads"),
        py::arg("unique_indices"), py::arg("reverse_idx"), py::arg("dim"),
        py::arg("unique_grads"), py::arg("device_num_sms"));

  m.def("dedup_input_indices", &dedup_input_indices,
        "duplicate indices from a given list or array of indices",
        py::arg("indices"), py::arg("offset"),
        py::arg("h_table_offsets_in_feature"),
        py::arg("d_table_offsets_in_feature"), py::arg("table_num"),
        py::arg("local_batch_size"), py::arg("reverse_idx"),
        py::arg("h_unique_nums"), py::arg("d_unique_nums"),
        py::arg("h_unique_offsets"), py::arg("d_unique_offsets"),
        py::arg("unique_idx"), py::arg("new_offsets"), py::arg("new_lengths"),
        py::arg("device_num_sms"), py::arg("unique_op"));
}
