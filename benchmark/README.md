# Dynamic Embedding Benchmark

## Overview

This folder contains benchmarks about dynamicemb.

## 1.Benchmark EmbeddingCollection

In this benchmark, we provide a simple performance test for dynamic embedding using 8 GPUs. The test utilizes the embedding table from DLRM and performs embedding table fusion to create a large embedding table, followed by lookups for 26 features.

### How to run

```bash
bash ./benchmark/benchmark_embedding_collection.sh <use_index_dedup> <use_dynamic_embedding> <batch_size>
```

#### Parameters

- `<use_index_dedup>`: A boolean flag to enable or disable index deduplication before data distribution.
  - **True**: Enables index deduplication, reducing communication overhead.
  - **False**: Disables index deduplication.
  - **Default**: True.

- `<use_dynamic_embedding>`: A boolean flag to enable or disable the use of dynamic embedding tables.
  - **True**: Enables dynamic embedding tables.
  - **False**: Uses static embedding tables from TorchREC.
  - **Default**: True.

- `<batch_size>`: The global batch size for processing during the benchmark.
  - **Default**: 65536.

### Test Results

In this benchmark, we primarily focus on the performance of embedding collection and deduplication. The tests were conducted on a single node with 8 H100 GPUs connected via NVSwitch. Below are the performance results:

| Configuration               | TorchREC Raw Table (ms) | Dynamic Embedding Table (ms) |
|-----------------------------|-------------------------|-------------------------------|
| Open Dedup, Batch Size 65536 | 14.88                   | 21.56                         |
| Close Dedup, Batch Size 65536 | 23.99                   | 28.47                         |

These results indicate the time taken to perform the embedding collection and deduplication operations under the specified configuration.

During the embedding lookup process, dynamic embedding incurs some performance overhead compared to TorchREC's raw table. However, these overheads diminish when considered within the context of the entire end-to-end model.

## 2.Benchmark BatchedDynamicEmbeddingTables

In this benchmark, we test the forward and backward overhead of `BatchedDynamicEmbeddingTables` (torch.nn.Module contains batched embedding tables) on a single GPU.

### How to run

```bash
bash ./benchmark/benchmark_batched_dynamicemb_tables.sh
```

### Test Results

We test the `BatchedDynamicEmbeddingTables` under different capacities with the same HBM consumption for embeddings, leading to different HBM proportion.
When generating indices, we utilize an extremely large range(2^63), so that most indices are unique and need to insert into HKV.

The overhead(ms) on H100 PCIe:

| use_index_dedup | batch_size | num_embeddings_per_feature | hbm_for_embeddings | optimizer_type | forward_overhead | backward_overhead | totoal_overhead |
|-----------------|------------|----------------------------|--------------------|----------------|------------------|-------------------|-----------------|
| TRUE            | 65536      | 8388608                    | 4                  | sgd            | 0.54184          | 0.363057          | 0.904897        |
| TRUE            | 65536      | 8388608                    | 4                  | adam           | 0.601176         | 0.477679          | 1.078855        |
| TRUE            | 65536      | 67108864                   | 4                  | sgd            | 2.746669         | 4.148325          | 6.894995        |
| TRUE            | 65536      | 67108864                   | 4                  | adam           | 3.226324         | 11.76063          | 14.98695        |
| TRUE            | 1048576    | 8388608                    | 4                  | sgd            | 5.158324         | 3.05149           | 8.209814        |
| TRUE            | 1048576    | 8388608                    | 4                  | adam           | 5.170962         | 7.844773          | 13.01574        |
| TRUE            | 1048576    | 67108864                   | 4                  | sgd            | 50.48192         | 56.61244          | 107.0944        |
| TRUE            | 1048576    | 67108864                   | 4                  | adam           | 74.15156         | 186.0786          | 260.2301        |