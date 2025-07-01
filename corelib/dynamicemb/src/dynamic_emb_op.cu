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
    const c10::optional<at::Tensor> &output_scores = c10::nullopt) {

  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  at::Tensor vals_ptr_tensor = at::empty({static_cast<int64_t>(n)}, 
    at::TensorOptions().dtype(at::kLong).device(values.device()));
  auto vals_ptr = reinterpret_cast<void**>(vals_ptr_tensor.data_ptr<int64_t>());
  at::Tensor founds_tensor = at::empty({static_cast<int64_t>(n)},
     at::TensorOptions().dtype(at::kBool).device(keys.device()));
  auto founds = founds_tensor.data_ptr<bool>();

  // table->find_and_initialize(n, keys.data_ptr(), vals_ptr, values.data_ptr(), founds, output_scores.data_ptr(), stream);
  if (output_scores.has_value()) {
    at::Tensor output_scores_ = output_scores.value();
    table->find_and_initialize(n, keys.data_ptr(), vals_ptr, values.data_ptr(), founds, output_scores_.data_ptr(), stream);
  } else {
    table->find_and_initialize(n, keys.data_ptr(), vals_ptr, values.data_ptr(), founds, nullptr, stream);
  }
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

// DynamicEmb核心前向传播函数：处理embedding lookup的dense模式
// 这个函数是整个DynamicEmb系统的核心，负责从动态embedding表中查找embedding向量
void lookup_forward_dense(
    std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>> tables,  // 动态embedding表的集合，每个表是一个HKV存储
    const at::Tensor indices,           // 输入的key indices，形状为[total_indices_num]
    const at::Tensor offsets,           // 每批次的offset信息，用于分割indices到不同的样本中
    const py::list scores,              // 每个表的score值，用于动态更新策略
    const std::vector<int> &table_offsets_in_feature, // 每个表在feature中的offset位置
    at::Tensor table_offsets,           // 每个表的累积offset，形状为[table_num+1]
    int table_num,                      // embedding表的数量
    int batch_size,                     // 批次大小
    int dim,                            // embedding维度
    bool use_index_dedup,               // 是否使用索引去重
    const at::Tensor unique_idx,        // 去重后的唯一索引
    const at::Tensor reverse_idx,       // 反向索引，用于将unique结果映射回原始位置
    const at::Tensor h_unique_nums,     // 每个表的unique数量(CPU tensor)
    const at::Tensor d_unique_nums,     // 每个表的unique数量(GPU tensor)
    const at::Tensor h_unique_offsets,  // 每个表的unique offset(CPU tensor)
    const at::Tensor d_unique_offsets,  // 每个表的unique offset(GPU tensor)
    const at::Tensor unique_embs,       // 存储unique embeddings的tensor
    const at::Tensor output_embs,       // 最终输出的embeddings
    int device_num_sms,                 // GPU的SM数量，用于kernel优化
    std::shared_ptr<dyn_emb::UniqueOpBase> unique_op,
    int frequency_threshold = 0, int mask_dims = 0) { // 去重操作的实现

  // ========== 第1步：输入验证 ==========
  // 确保所有tensor都在CUDA设备上
  if (!offsets.is_cuda() || !indices.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  // 验证unique相关tensor的数据类型必须是uint64
  if (h_unique_nums.scalar_type() != at::kUInt64 ||
      d_unique_nums.scalar_type() != at::kUInt64) {
    throw std::runtime_error(
        "h_unique_nums and d_unique_nums must have dtype uint64_t");
  }

  // ========== 第2步：初始化CUDA环境和数据类型 ==========
  auto stream = at::cuda::getCurrentCUDAStream().stream();  // 获取当前CUDA stream
  int64_t indices_shape = indices.size(0);                  // 总的indices数量
  
  // 转换PyTorch数据类型到内部DataType枚举
  auto unique_num_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_nums.dtype()));    // unique_nums的数据类型
  auto unique_offset_type = scalartype_to_datatype(
      convertTypeMetaToScalarType(d_unique_offsets.dtype())); // unique_offsets的数据类型

  // ========== 第3步：准备offset数据 ==========
  // 创建CPU版本的offset tensor，用于后续的CPU计算
  at::Tensor h_offset =
      at::empty_like(offsets, offsets.options().device(at::kCPU));
  // 异步将GPU上的offsets拷贝到CPU
  AT_CUDA_CHECK(cudaMemcpyAsync(h_offset.data_ptr(), offsets.data_ptr(),
                                offsets.numel() * offsets.element_size(),
                                cudaMemcpyDeviceToHost, stream));

  // ========== 第4步：检查和调整unique操作的容量 ==========
  size_t unique_op_capacity = unique_op->get_capacity(); // 获取当前去重操作的容量
  // 如果输入数据量超过容量的一半，则扩容到2倍indices数量
  if (indices_shape * 2 > unique_op_capacity) {
    // 创建新的keys和values tensor
    at::Tensor new_keys = at::empty({indices_shape * 2}, indices.options());
    at::Tensor new_vals = at::empty(
        {indices_shape * 2},
        at::TensorOptions().dtype(at::kUInt64).device(indices.device()));
    // 重置unique操作的容量
    unique_op->reset_capacity(new_keys, new_vals, indices_shape * 2, stream);
  }

  // ========== 第5步：为每个表创建临时unique indices tensor ==========
  std::vector<at::Tensor> tmp_unique_indices(table_num);
  for (int i = 0; i < table_num; ++i) {
    // 为每个表创建与indices相同大小的临时tensor
    tmp_unique_indices[i] = at::empty_like(indices);
  }

  // ========== 第6步：计算每个表的indices范围 ==========
  // 创建CPU版本的table_offsets，用于计算每个表的数据范围
  at::Tensor h_table_offsets =
      at::empty({table_num + 1}, table_offsets.options().device(at::kCPU));
  // 同步等待之前的GPU->CPU内存拷贝完成
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  h_table_offsets[0] = 0; // 第一个offset总是0

  // ========== 第7步：对每个表执行unique操作 ==========
  for (int i = 0; i < table_num; ++i) {
    // 计算当前表在feature中的范围
    int table_offset_begin = table_offsets_in_feature[i];     // 表i的起始feature位置
    int table_offset_end = table_offsets_in_feature[i + 1];   // 表i的结束feature位置
    
    // 将feature offset转换为batch offset
    int offset_begin = table_offset_begin * batch_size;       // 批次起始位置
    int offset_end = table_offset_end * batch_size;           // 批次结束位置

    // 从CPU的offset tensor中获取indices的实际范围
    int64_t indices_begin = h_offset[offset_begin].item<int64_t>(); // 当前表的indices起始位置
    int64_t indices_end = h_offset[offset_end].item<int64_t>();     // 当前表的indices结束位置
    int64_t indices_length = indices_end - indices_begin;           // 当前表的indices数量
    h_table_offsets[i + 1] = indices_end;                          // 累积offset

    if (indices_length == 0) {
      // 如果当前表没有数据，则将对应的unique_nums设置为0
      DEMB_CUDA_CHECK(cudaMemsetAsync(
          reinterpret_cast<uint64_t *>(d_unique_nums.data_ptr()) + i, 0,
          sizeof(uint64_t), stream));
      // 更新unique_offsets
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    } else {
      // 如果有数据，则进行unique操作
      // 创建当前表的indices子tensor
      at::Tensor tmp_indices = create_sub_tensor(indices, indices_begin);
      // 创建当前表的reverse_idx子tensor
      at::Tensor tmp_reverse_idx =
          create_sub_tensor(reverse_idx, indices_begin);
      // 创建当前表的unique_num子tensor
      at::Tensor tmp_d_unique_num = create_sub_tensor(d_unique_nums, i);
      // 获取之前的unique offset
      at::Tensor previous_d_unique_num = create_sub_tensor(d_unique_offsets, i);
      
      // 执行unique操作：去除重复的indices
      unique_op->unique(tmp_indices, indices_length, tmp_reverse_idx,
                        tmp_unique_indices[i], tmp_d_unique_num, stream,
                        previous_d_unique_num);
      // 更新cumulative offset
      dyn_emb::add_offset(d_unique_nums.data_ptr(), d_unique_offsets.data_ptr(),
                          i, unique_num_type, unique_offset_type, stream);
    }
  }

  // ========== 第8步：同步GPU计算结果到CPU ==========
  // 将unique数量从GPU拷贝到CPU
  AT_CUDA_CHECK(
      cudaMemcpyAsync(h_unique_nums.data_ptr(), d_unique_nums.data_ptr(),
                      d_unique_nums.numel() * d_unique_nums.element_size(),
                      cudaMemcpyDeviceToHost, stream));
  // 将unique offsets从GPU拷贝到CPU
  AT_CUDA_CHECK(cudaMemcpyAsync(
      h_unique_offsets.data_ptr(), d_unique_offsets.data_ptr(),
      (d_unique_nums.numel() + 1) * d_unique_nums.element_size(),
      cudaMemcpyDeviceToHost, stream));
  // 同步等待拷贝完成
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));
  // 将table_offsets从CPU拷贝到GPU
  AT_CUDA_CHECK(
      cudaMemcpyAsync(table_offsets.data_ptr(), h_table_offsets.data_ptr(),
                      table_offsets.numel() * table_offsets.element_size(),
                      cudaMemcpyHostToDevice, stream));

  // ========== 第9步：从embedding表中查找embedding向量 ==========
    // Reserve unique_output_scores tensor for all tables if frequency masking is enabled
  at::Tensor unique_output_scores;
  int64_t total_unique_embs = 0;
  for (int i = 0; i < table_num; ++i) {
    total_unique_embs += h_unique_nums[i].item<int64_t>();
  }
// 按unique embeddings总数分配
  unique_output_scores = at::zeros({total_unique_embs}, 
      at::TensorOptions().dtype(at::kUInt64).device(indices.device()));


  int64_t unique_embs_offset = 0; // 在unique_embs tensor中的累积offset
  int64_t scores_offset = 0;  // 新增：scores的offset

  for (int i = 0; i < table_num; ++i) {
    int64_t tmp_unique_num = h_unique_nums[i].item<int64_t>(); // 当前表的unique数量
    if (tmp_unique_num != 0) {
      // 创建当前表在unique_embs中的子tensor
      at::Tensor tmp_unique_embs =
          create_sub_tensor(unique_embs, unique_embs_offset * dim);
    // 创建scores子tensor（新增）
      at::Tensor tmp_unique_scores = create_sub_tensor(unique_output_scores, scores_offset);
    
      if (use_index_dedup) {
        // 如果使用索引去重，则调用find_and_initialize（仅查找，不插入新key）
        find_and_initialize(tables[i], tmp_unique_num, tmp_unique_indices[i], tmp_unique_embs, tmp_unique_scores);
        
        void *dst_ptr = reinterpret_cast<char *>(unique_idx.data_ptr()) +
                        unique_embs_offset * unique_idx.element_size();
        void *src_ptr = tmp_unique_indices[i].data_ptr();
        size_t copy_size = tmp_unique_num * unique_idx.element_size();
        AT_CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, copy_size,
                                      cudaMemcpyDeviceToDevice, stream));
      } else {
        // 如果不使用索引去重，则调用find_or_insert（查找或插入新key）
        auto score = std::make_optional<uint64_t>(py::cast<uint64_t>(scores[i])); // 获取当前表的score
        find_or_insert(tables[i], tmp_unique_num, tmp_unique_indices[i],
                      tmp_unique_embs, score);
        //调用find pointer部分，todo 感觉这样写不太好，但修改目前的存在的find pointer不知道是否存在问题。
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        at::Tensor vals_ptr_tensor = at::empty({static_cast<int64_t>(tmp_unique_num)}, 
          at::TensorOptions().dtype(at::kLong).device(tmp_unique_embs.device()));
        auto vals_ptr = reinterpret_cast<void**>(vals_ptr_tensor.data_ptr<int64_t>());
        at::Tensor founds_tensor = at::empty({static_cast<int64_t>(tmp_unique_num)},
          at::TensorOptions().dtype(at::kBool).device(tmp_unique_indices[i].device()));
        auto founds = founds_tensor.data_ptr<bool>(); 
          tables[i]->find_pointers(tmp_unique_num, tmp_unique_indices[i].data_ptr(), vals_ptr, founds, 
            tmp_unique_scores.data_ptr(), stream);
          }
    }
    scores_offset += tmp_unique_num;  // 更新scores的offset
    unique_embs_offset += tmp_unique_num; // 更新累积offset
  }
  
at::Tensor unique_embeddings_for_scatter;
if (frequency_threshold > 0 && mask_dims > 0) {
  unique_embeddings_for_scatter = unique_embs.clone();
  auto score_type = scalartype_to_datatype(convertTypeMetaToScalarType(unique_output_scores.dtype()));
  
  auto emb_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(unique_embeddings_for_scatter.dtype()));
  dyn_emb::mask_embeddings_by_frequency(
    unique_embeddings_for_scatter.data_ptr(), unique_output_scores.data_ptr(),
    total_unique_embs, dim, frequency_threshold, mask_dims,
    emb_type, score_type, device_num_sms, stream);
} else {
  unique_embeddings_for_scatter = unique_embs;
}
// template <typename T>
// __global__ void mask_embeddings_by_frequency(
//   int batch_size, 
// int dim, 
// int mask_dim, 
// uint64_t* frequencies, 
// uint64_t frequency_threshold, 
// T* unique_embeds_for_scatter) {
// }


  // ========== 第10步：将unique embeddings scatter到最终输出位置 ==========
  // 获取源和目标的数据类型
  auto src_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(unique_embeddings_for_scatter.dtype()));
  auto dst_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(output_embs.dtype()));
  auto offset_type =
      scalartype_to_datatype(convertTypeMetaToScalarType(reverse_idx.dtype()));

     // ========== DEBUG: 简单测试masking ==========
   printf("=== MASKING DEBUG INFO ===\n");
   printf("Frequency threshold: %d, Mask dims: %d\n", frequency_threshold, mask_dims);
   printf("Total unique embeddings: %ld, Embedding dim: %d\n", total_unique_embs, dim);
   
   // 检查两个tensor是否指向同一内存
   bool same_storage = unique_embeddings_for_scatter.data_ptr() == unique_embs.data_ptr();
   printf("unique_embeddings_for_scatter and unique_embs point to same memory: %s\n", 
          same_storage ? "YES" : "NO");
   
   if (!same_storage) {
     printf("Different tensors detected - masking may have been applied!\n");
   } else {
     printf("Same tensor - no masking applied\n");
   }
     if (frequency_threshold > 0 && mask_dims > 0) {
       printf("Tensor shape: [%ld, %ld]\n", total_unique_embs, (int64_t)dim);
       
       at::Tensor h_unique_embeddings_for_scatter = unique_embeddings_for_scatter.cpu();
       at::Tensor h_unique_output_scores = unique_output_scores.cpu();
       int masked_embeddings = 0;
       
               if (h_unique_embeddings_for_scatter.dtype() == at::kFloat) {
          float* data_ptr = h_unique_embeddings_for_scatter.data_ptr<float>();
          uint64_t* score_ptr = h_unique_output_scores.data_ptr<uint64_t>();
          
          int should_be_masked = 0;
          int actually_masked = 0;
          int total_checked = std::min(10L, total_unique_embs);
          
          for(int j = 0; j < total_unique_embs; j++) {
            uint64_t score = score_ptr[j];
            if(score > frequency_threshold) continue;
            bool should_mask = (score < frequency_threshold);
            if (should_mask) should_be_masked++;
            
            int zero_count = 0;
            for (int i = dim - mask_dims; i < dim; i++) {
              float value = data_ptr[j * dim + i];
              if (abs(value) < 1e-6) {
                zero_count++;
              }
            }
            bool is_masked = (zero_count == mask_dims);
            if (is_masked) actually_masked++;
            
            printf("Embedding %d: score=%lu, should_mask=%s, is_masked=%s, last %d values=[", 
                   j, score, should_mask?"YES":"NO", is_masked?"YES":"NO", mask_dims);
            for (int i = dim - mask_dims; i < dim; i++) {
              printf("%.3f ", data_ptr[j * dim + i]);
            }
            printf("]\n");
          }
          
          printf("Summary: checked %d embeddings, %d should be masked, %d actually masked\n", 
                 total_checked, should_be_masked, actually_masked);
        }
       
               // printf("Checked first %d embeddings: %d are properly masked (last %d dims are zero)\n", 
        //        total_unique_embs, masked_embeddings, mask_dims);
     }
   printf("========================\n");
   // ========== END DEBUG ==========

  // 调用scatter_fused kernel将unique_embs根据reverse_idx散列到output_embs中
  // 这一步将去重后的embeddings重新分布到原始的位置，完成最终的lookup操作
  dyn_emb::scatter_fused(unique_embeddings_for_scatter.data_ptr(),    // 源：unique后的embeddings
                         output_embs.data_ptr(),    // 目标：最终输出的embeddings
                         reverse_idx.data_ptr(),    // 索引：如何将unique结果映射回原位置
                         indices_shape,             // 总的indices数量
                         dim,                       // embedding维度
                         src_type,                  // 源数据类型
                         dst_type,                  // 目标数据类型
                         offset_type,               // offset数据类型
                         device_num_sms,            // GPU的SM数量
                         stream);                   // CUDA stream
}

void lookup_forward_dense(
    std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>> tables,
    const at::Tensor indices, const at::Tensor offsets,
    const std::vector<int> &table_offsets_in_feature, int table_num,
    int batch_size, int dim, const at::Tensor h_unique_offsets,
    const at::Tensor unique_embs, const at::Tensor output_embs) {

  if (!offsets.is_cuda() || !indices.is_cuda()) {
    throw std::runtime_error(
        "offsets or indices tensor must be on CUDA device");
  }

  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t indices_shape = indices.size(0);
  auto scalar_type = unique_embs.dtype().toScalarType();
  auto emb_dtype = scalartype_to_datatype(scalar_type);
  scalar_type = output_embs.dtype().toScalarType();
  auto output_dtype = scalartype_to_datatype(scalar_type);
  auto &device_prop = DeviceProp::getDeviceProp(indices.device().index());

  at::Tensor h_offset =
      at::empty_like(offsets, offsets.options().device(at::kCPU));
  AT_CUDA_CHECK(cudaMemcpyAsync(h_offset.data_ptr(), offsets.data_ptr(),
                                offsets.numel() * offsets.element_size(),
                                cudaMemcpyDeviceToHost, stream));
  AT_CUDA_CHECK(cudaStreamSynchronize(stream));

  h_unique_offsets[0] = 0;
  for (int i = 0; i < table_num; ++i) {
    int table_offset_begin = table_offsets_in_feature[i];
    int table_offset_end = table_offsets_in_feature[i + 1];
    int offset_begin = table_offset_begin * batch_size;
    int offset_end = table_offset_end * batch_size;

    int64_t indices_begin = h_offset[offset_begin].item<int64_t>();
    int64_t indices_end = h_offset[offset_end].item<int64_t>();
    int64_t indices_length = indices_end - indices_begin;
    h_unique_offsets[i + 1] = indices_end;
    at::Tensor tmp_indices = create_sub_tensor(indices, indices_begin);
    at::Tensor tmp_unique_embs =
        create_sub_tensor(unique_embs, indices_begin * dim);
    find_or_insert(tables[i], indices_length, tmp_indices, tmp_unique_embs);
    at::Tensor tmp_output_embs =
        create_sub_tensor(output_embs, indices_begin * dim);
    dyn_emb::batched_vector_copy_device(
        tmp_unique_embs.data_ptr(), output_embs.data_ptr(), indices_length, dim,
        emb_dtype, output_dtype, device_prop.num_sms, stream);
  }
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

  m.def("lookup_forward_dense",
        (void (*)(std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>>,
                  const at::Tensor, const at::Tensor, const py::list, 
                  const std::vector<int> &,
                  at::Tensor, int, int, int, bool, const at::Tensor,
                  const at::Tensor, const at::Tensor, const at::Tensor,
                  const at::Tensor, const at::Tensor, const at::Tensor,
                  const at::Tensor, int,
                  std::shared_ptr<dyn_emb::UniqueOpBase>, int, int)) &
            lookup_forward_dense,
        "lookup forward dense for duplicated keys", py::arg("tables"),
        py::arg("indices"), py::arg("offsets"), py::arg("scores"),
        py::arg("table_offsets_in_feature"), py::arg("table_offsets"),
        py::arg("table_num"), py::arg("batch_size"), py::arg("dim"),
        py::arg("use_index_dedup"), py::arg("unique_idx"),
        py::arg("reverse_idx"), py::arg("h_unique_nums"),
        py::arg("d_unique_nums"), py::arg("h_unique_offsets"),
        py::arg("d_unique_offsets"), py::arg("unique_embs"),
        py::arg("output_embs"), py::arg("device_num_sms"),
        py::arg("unique_op"), py::arg("frequency_threshold") = 0, 
        py::arg("mask_dims") = 0);

  m.def("lookup_forward_dense",
        (void (*)(std::vector<std::shared_ptr<dyn_emb::DynamicVariableBase>>,
                  const at::Tensor, const at::Tensor, const std::vector<int> &,
                  int, int, int, const at::Tensor, const at::Tensor,
                  const at::Tensor)) &
            lookup_forward_dense,
        "lookup forward dense for globally deduplicated keys",
        py::arg("tables"), py::arg("indices"), py::arg("offsets"),
        py::arg("table_offsets_in_feature"), py::arg("table_num"),
        py::arg("batch_size"), py::arg("dim"), py::arg("h_unique_offsets"),
        py::arg("unique_embs"), py::arg("output_embs"));

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
