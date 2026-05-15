# Recsys KVCache Manager

Recsys KVCache Manager is a Python package that LLM-compatible KV data caching, storage and lookup for generative recommanders models inference.
It supports kvcache management based on **user-id**s from recommender systems, inter-requests kvcache reuse for the same user.

Recsys KVCache Manager is based on the **Pytorch** ecosystem. It contains kv data cache on both GPU memory and host memory backed with lower-tier storage.
It supports lookup, offloading (to low-tier storage), and onboard (to GPU memory for inference), and also easy ways to read/write data to the kvcache.


## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Basic APIs](#basic-apis)
- [Examples](#example)
- [Future Plans](#future-plans)

## Features

- **User ID Based Caching**:
For recommander systems, the behaviour sequences (which generates the KV data) from different users varies greatly, and there is almost not common prefix for the input sequences.
Recsys KVCache Manager supports data lookup based on user-id only, instead of comparing the token values of sequences.

<div style="margin-left: 2em;"> 

**Recsys KVCache Manager** modules:

</div>

```
    + KVCacheManager :  Interface for kvcache operations
    |
    ├---- GPUKVCacheManager :  Manager for GPU kvcache table
    |
    └---- HostKVStorageManagerBase :  Interface of manager for host memory/ssd/remote kvcache
        |
        ├---- NativeHostKVCacheManager :  Wrapper to pinned host memory only kvcache
        |
        └---- FlexKVCacheManager :  Wrapper to FlexKV cache system
```

- **Paged GPU KVCache Table**:
The GPU kvcache table is organized as a paged KV-data table, and supports KV data adding/appending, lookup and eviction. When appending new data to the GPU cache, we will evict data from the oldest users (based on the LRU policy) if there is no empty page. The HSTU attention kernel from FBGEMM-HSTU also load KV data directly from a paged table in HSTU attention kernels, which avoids additional data copy.

- **Asynchronous Onboarding/Offloading**:
By using asynchronous data copy on the side CUDA stream, we overlap the KV data transfer between GPU memory and host storage (onboarding/offloading) with embedding lookup, sequence pre-/post-processing, and inference for other requests (in some cases) to reduce the latency of HSTU inference.
Furthermore, the `NativeHostKVCacheManager` backend supports layerwise KV data onboarding, overlapping the H2D data transfer with computation from the previous HSTU layers.

- **Extension for Multiple Backend**: 
`HostKVStorageManagerBase` is provided as an interface for other LLM-compatible kvcache systems for the host memory, storage and remote data pool.
This can be easily extended to integration other kvcache system. Currently, we provide the integration with [`FlexKV`](https://github.com/taco-project/FlexKV/tree/main) as the low-tier kv storage backend. 


## Installation

To install, please use the following command:

```bash
TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0" pip3 install --no-build-isolation .
```

**Note**: To enable the FlexKV backend, please install FlexKV package according the [doc](https://github.com/taco-project/FlexKV/tree/main#how-to-use).


## Basic APIs

-   `lookup_kvcache`: get the cached sequence length from both GPU cache table and host kv storage.

-   `allocate_kvcache`: assign required cache page in the GPU cache table for infernece.
This generates page ids for read/write data, and also other metadata for appending KV data into the GPU table.
It also evicts used cache pages when running out of empty pages. **No offloading** upon eviction in current implementation.

-   `onboard_launch`: launches async host-to-GPU kvcache transfer.

-   `onboard_try_wait`, `onboard_wait`: performs non-blocking/blocking waiting for KV data onboarding.

-   `offload_launch`: launches async GPU-to-host offload and records the task into ongoing offload queue.

-   `offload_try_wait`: polls ongoing offload tasks, finishes ready tasks, and cancels failed/timed-out tasks, and unlock 

-	`evict`, `evict_all`: explicitly evicts cached data from GPU cache table and/or low-tier storage if supported.

#### Importance Notes:

There are some **limitations** for current implementation. These will be resolved soon for broader use cases, and better performance.

1. API `allocate_kvcache` is host blocking, and cannot overlap with other operations.

2. Allow only **one** GPU kvcache manager per device, and only one inference instance for each GPU kvcache manager.

3. Host backend "native" is limited for at most **one** GPU kvcache manager, and only one inference instance. Recommend to use with `user_id` based routing with inference instance isolation.


## Example

**Typical Usage of KVCache Manager**:
```python
    # Input User IDs
    user_ids: torch.Tensor
    # Sequence lengths per user
    sequence_lengths: torch.Tensor

    # Lookup 
    index_meta, lookup_res = kvcache_mgr.lookup_kvcache(user_ids, sequence_lengths)

    # strip cached tokens from input sequences
    [[ ... strip cached tokens ... ]]

    # Allocate in GPU cache table
    kvcache_mgr.offload_try_wait()  # [Optional] Try to free up GPU cache space by completing offloading.
    kvcache_metadata = kvcache_mgr.allocate_kvcache(index_meta, lookup_res)

    # Onboard to GPU cache table, non-blocking, if necessary
    kvcache_mgr.onboard_launch(index_meta, lookup_res, kvcache_metadata)
    for layer_idx in range(3):
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

    [[ ... embedding lookup ... ]]
    [[ ... preprocess ... ]]


    # Dense Module computation
    # Note: Here we show two possible ways to synchronize with onboard completion:
    #       (1) blocking wait for onboard completion
    #       (2) non-blocking "cuda stream wait" with layerwise onboard events.

    # Case (1): Blocking total wait. [ Note: not supported with "native" backend. ]
    kvcache_mgr.onboard_wait(index_meta, kvcache_metadata.kv_onload_handle)
    for layer_idx in range(num_layers):

        [[ ... write new KV data thru `kvcache_metadata.kv_cache_table` ]]  # See `k`vcache_mgr.gpu_kvcache_mgr.put``

        # Case (2): Layerwise stream wait. [ Note: only with "native" backend. ]
        kvcache_metadata.kv_onload_handle.stream_wait_layer(layer_idx)

        [[ ... attention computation, loading data using `kv_indices`, `kvkv_indptr`, etc. ... ]]
        

    # Offloading to host and lower tiers, non-blocking, if necessary
    kvcache_mgr.offload_try_wait() # [Optional] Try to free up host buffer/sync host caching status by completing offloading.
    kvcache_mgr.offload_launch(index_meta)

    [[ ... postprocessing ... ]]
    [[ ... return inference results ... ]]
```

**Refer to** HSTU model inference in (Recsys-Examples) for details [[InferenceRankingGR](../../examples/hstu/model/inference_ranking_gr.py), [InferenceDenseModule](../../examples/hstu/modules/inference_dense_module.py)]


## Future Plans

1. Support concurrent kvcache operations inference instances.
2. Support torch export and AOT induction compilation with the recommenders models for Torch C++ runtime inference.
