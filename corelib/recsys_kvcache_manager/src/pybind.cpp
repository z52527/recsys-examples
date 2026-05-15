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
#include "gpu_kvcache_manager_impl.h"

PYBIND11_MODULE(kvcache_cpp, m) {
  py::class_<kvcache::HostKVStorageImpl>(m, "HostKVStorageImpl", py::module_local())
    .def(py::init<int, int, int, int, int64_t, int64_t, int64_t, int64_t, int>(), 
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_tokens_per_chunk"),
         py::arg("capacity_per_layer"),
         py::arg("max_batch_size"),
         py::arg("max_sequence_length"),
         py::arg("device_idx"))
    .def("register_gpu_cache_table", &kvcache::HostKVStorageImpl::register_gpu_cache_table)
    .def("lookup", &kvcache::HostKVStorageImpl::lookup)
    .def("get_kvdata_tensor", &kvcache::HostKVStorageImpl::get_kvdata_tensor)
    .def("onload_kvcache", &kvcache::HostKVStorageImpl::onload_kvcache)
    .def("offload_kvcache", &kvcache::HostKVStorageImpl::offload_kvcache)
    .def("finish_offload", &kvcache::HostKVStorageImpl::finish_offload)
    .def("cancel_offload", &kvcache::HostKVStorageImpl::cancel_offload)
    .def("evict", &kvcache::HostKVStorageImpl::evict)
    .def("evict_all", &kvcache::HostKVStorageImpl::evict_all)
  ;

  py::class_<kvcache::GPUKVCacheManagerImpl>(m, "GPUKVCacheManagerImpl", py::module_local())
    .def(py::init<int, int, int, int, int, int, int, int, int, int>(),
         py::arg("num_layers"),
         py::arg("num_kv_heads"),
         py::arg("kv_headdim"),
         py::arg("num_tokens_per_page"),
         py::arg("num_tokens_per_chunk"),
         py::arg("num_primary_cache_pages"),
         py::arg("num_buffer_pages"),
         py::arg("max_batch_size"),
         py::arg("max_sequence_length"),
         py::arg("device_idx"))
    .def("lookup", &kvcache::GPUKVCacheManagerImpl::lookup)
    .def("allocate", &kvcache::GPUKVCacheManagerImpl::allocate)
    .def("evict", &kvcache::GPUKVCacheManagerImpl::evict)
    .def("evict_all", &kvcache::GPUKVCacheManagerImpl::evict_all)
    .def("revoke_onboard_pages", &kvcache::GPUKVCacheManagerImpl::revoke_onboard_pages)
    .def("check_for_offload", &kvcache::GPUKVCacheManagerImpl::check_for_offload)
    .def("acquire_offload_pages", &kvcache::GPUKVCacheManagerImpl::acquire_offload_pages)
    .def("release_offload_pages", &kvcache::GPUKVCacheManagerImpl::release_offload_pages)
  ;

  py::class_<kvcache::KVOnloadHandle>(m, "KVOnloadHandle", py::module_local())
    .def(py::init<int>(), py::arg("num_layers"))
    // .def("reset", &kvcache::KVOnloadHandle::reset)
    .def("wait_layer", &kvcache::KVOnloadHandle::wait_layer)
  ;

  py::class_<kvcache::KVOffloadHandle>(m, "KVOffloadHandle", py::module_local())
    .def(py::init<int>(), py::arg("num_layers"))
    // .def("reset", &kvcache::KVOffloadHandle::reset)
    .def("try_wait_layer", &kvcache::KVOffloadHandle::try_wait_layer)
    .def("get_user_ids", &kvcache::KVOffloadHandle::get_user_ids)
    .def("get_start_indices", &kvcache::KVOffloadHandle::get_start_indices)
    .def("get_lengths", &kvcache::KVOffloadHandle::get_lengths)
  ;
}