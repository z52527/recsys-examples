# DynamicEmb

DynamicEmb is a Python package that provides model-parallel dynamic embedding tables and embedding lookup functionalities for TorchREC, specifically targeting the sparse training aspects of recommendation systems. Currently, DynamicEmb utilizes the [HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV) hash table backend, which is designed to store key-value (feature-embedding) pairs in the high-bandwidth memory (HBM) of GPUs as well as in host memory.

The lookup kernel algorithms implemented in DynamicEmb primarily leverage portions of the algorithms from the [EMBark](https://dl.acm.org/doi/abs/10.1145/3640457.3688111) paper (Embedding Optimization for Training Large-scale Deep Learning Recommendation Systems with EMBark).

## Features

- **Dynamic Embedding Table Support**: DynamicEmb supports embedding tables backed by hash tables, allowing for optimal utilization of both GPU memory and host memory within the system. Hash tables can accept any specified `indices` type values, unlike static tables which only support index values.

- **Seamless Integration with TorchREC**: DynamicEmb inherits the API from TorchREC, ensuring that its usage is largely consistent with TorchREC. Users can easily modify their existing code to run recommendation system models with dynamic embedding tables alongside TorchREC.

- **Embedded in Distributed-Recommender Repository Supporting Generative-Recommenders(GR) Models**: Currently, DynamicEmb is integrated into the Distributed-Recommender repository, serving as an embedding backend for GR models.

- Support for creating dynamic embedding tables within `EmbeddingBagCollection` and `EmbeddingCollection` in TorchREC, allowing for embedding storage and lookup, and enabling coexistence with native Torch embedding tables within Torch models.

- Support for optimizer types: `SGD` and `AdamW`.

- Support for automatically parallel `dump`/`load` of embedding weights in dynamic embedding tables.


## Pre-requisites

- TorchREC v0.7(The current version in use is a slightly modified version of TorchREC v0.7, which includes an added feature for registering customized embedding tables. It is not the raw TorchREC v0.7. We are actively communicating with the TorchREC team, with the hope of eventually upstreaming the customized embedding table registration feature into TorchREC)

### Version Compatibility

Relationship between Different Versions of DynamicEmb and Their Pre-requisites

| DynamicEmb Version | Pre-requisites Version |
|--------------------|------------------------|
| v0.1               | TorchREC v0.7          |

## Installation

To install DynamicEmb, use the following command:

```bash
python setup.py install
```

## DynamicEmb APIs

Regarding how to use the DynamicEmb APIs and their parameters, please refer to the `DynamicEmb_APIs.md` file in the same folder as this document.

## Usage Notes

1. Only the following optimizer types are supported: `SGD`, `EXACT_SGD`, and `AdamW` (specified as the string `"adam"`). This behavior is to maintain consistency with TorchREC.
2. The sharding method for dynamic embedding tables is always `row-wise sharding`, which will be evenly distributed across all GPUs within the TorchREC scope, unlike the `table-wise` and other sharding methods in TorchREC.
3. The allocated memory for dynamic embedding tables may have slight differences from the specified `num_embeddings` because each dynamic embedding table must set a capacity as a power of 2. This will be automatically calculated by the code, so please ensure that `num_embeddings` is aligned to a power of 2 when applying.
4. The lookup process for each dynamic embedding table incurs additional overhead from unique or radix sort operations. Therefore, if you request a large number of small dynamic embedding tables for lookup, the performance will be poor. Since the lookup range of dynamic embedding tables is particularly large (using the entire range of `int64_t`), it is recommended to create one large embedding table and perform a fused lookup for multiple features.
5. Although dynamic embedding tables can be trained together with TorchREC tables, they cannot be fused together for embedding lookup. Therefore, it is recommended to select dynamic embedding tables for all model-parallel tables during training.
6. Currently, DynamicEmb supports training with TorchREC's `EmbeddingBagCollection` and `EmbeddingCollection`. However, in version v0.1, the main lookup process of `EmbeddingBagCollection` is implemented using torch's ops, not fuse a lot of cuda kernels, which may result in some performance issues. Will fix this performance problem in future versions.

### DynamicEmb Insertion Behavior Checking Modes

DynamicEmb uses a hashtable as the backend. If the embedding table capacity is small and the number of indices in a single feature is large, it is easy for too many indices to be allocated to the same hash table bucket in one lookup, resulting in the inability to insert indices into the hashtable. DynamicEmb resolves this issue by setting the lookup results of indices that cannot be inserted to 0.

Fortunately, in a hashtable with a large capacity, such insertion failures are very rare and almost never occur. This issue is more frequent in hashtables with small capacities, which can affect training accuracy. Therefore, we do not recommend using dynamic embedding tables for very small embedding tables.

To prevent this behavior from affecting training without user awareness, DynamicEmb provides a safe check mode. Users can set whether to enable safe check when configuring `DynamicEmbTableOptions`. Enabling safe check will add some overhead, but it can provide insights into whether the hash table frequently fails to insert indices. If the number of insertion failures is high and the proportion of affected indices is large, it is recommended to either increase the dynamic embedding capacity or avoid using dynamic embedding tables for small embedding tables.

#### Example

```python
from dynamic_emb import DynamicEmbTableOptions, DynamicEmbCheckMode

# Configure the DynamicEmbTableOptions with safe check mode enabled
table_options = DynamicEmbTableOptions(
    safe_check_mode=DynamicEmbCheckMode.WARNING
)

# Use the table_options in your dynamic embedding setup
# ...
```

## Getting Started

We provide benchmark and unit test code to demonstrate how to use DynamicEmb. Please visit the benchmark and test folders. Below is a pseudocode example demonstrating how to convert TorchREC code to use DynamicEmb.

To get started with DynamicEmb, please prepare a training script with model parallelism, TorchREC's `EmbeddingBagCollection` or `EmbeddingCollection`. follow these steps:

1. **Modify the `EmbeddingShardingPlanner` part of the TorchREC code and replace it with the corresponding DynamicEmb API. Use `DynamicEmbParameterConstraints` to specify which embedding table should use a dynamic embedding table.**:
    ```python
    from dynamicemb.planner import DynamicEmbParameterConstraints,DynamicEmbeddingShardingPlanner
    from dynamicemb.planner import DynamicEmbeddingEnumerator
    from dynamicemb.shard import DynamicEmbeddingCollectionSharder
    from dynamicemb import DynamicEmbInitializerMode, DynamicEmbInitializerArgs, DynamicEmbTableOptions

    dict_const = {}
    for i in range(args.num_embedding_table):
        if args.data_parallel_embeddings is not None and i in args.data_parallel_embeddings:
            const = ParameterConstraints(
                sharding_types=[
                    ShardingType.DATA_PARALLEL.value
                ],
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
            )
        else:
            use_dynamicemb = True if i < args.num_dyn_emb_table else False
            const = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value]
                pooling_factors=[args.multi_hot_sizes[i]],
                num_poolings=[1],
                enforce_hbm=True,
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=use_dynamicemb,
                dynamicemb_options = DynamicEmbTableOptions(
                    global_hbm_for_values=1024 ** 3,
                    initializer_args=DynamicEmbInitializerArgs(
                        mode=DynamicEmbInitializerMode.DEBUG
                    ),
                ),
            )

        dict_const[table_idx_to_name(i)] = const
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
            hbm_cap=args.hbm_cap,
            ddr_cap=1024 * 1024 * 1024 * 1024,
            intra_host_bw=args.intra_host_bw,
            inter_host_bw=args.inter_host_bw,
        )

        enumerator = DynamicEmbeddingEnumerator(
                  topology = topology,
                  constraints=dict_const,
                )

    planner = DynamicEmbeddingShardingPlanner(
        eb_configs = eb_configs,
        topology = topology,
        constraints=dict_const,
        batch_size=args.batch_size,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        debug=True,
    )
    ```

2. **Use the planner generated in the first step to perform TorchREC's sharding plan and module shard. Note: If using TorchREC's `EmbeddingCollectionSharder`, it needs to be replaced with `DynamicEmbeddingCollectionSharder`. However, `EmbeddingBagCollectionSharder` does not need to be replaced**:
    ```python
    qcomm_forward_precision = get_comm_precission(args.fwd_a2a_precision)
    qcomm_backward_precision = get_comm_precission(args.fwd_a2a_precision)
    qcomm_codecs_registry = (
        get_qcomm_codecs_registry(
            qcomms_config=QCommsConfig(
                forward_precision=qcomm_forward_precision,
                backward_precision=qcomm_backward_precision,
            )
        )
        if backend == "nccl"
        else None
    )
    fused_params = {"output_dtype": SparseType.FP32}
    if not args.use_torch_opt:
        fused_params.update(optimizer_kwargs)
    sharder = DynamicEmbeddingCollectionSharder(qcomm_codecs_registry=qcomm_codecs_registry,
                                                fused_params=fused_params, use_index_dedup=True)
    plan = planner.collective_plan(ebc, [sharder], dist.GroupMember.WORLD)


    data_parallel_wrapper = DefaultDataParallelWrapper(
        allreduce_comm_precision=args.allreduce_precision
    )
    model = DistributedModelParallel(
        module=ebc,
        device=device,
        sharders=[sharder],
        plan=plan,
        data_parallel_wrapper=data_parallel_wrapper,
    )
    ```

3. **Train the created torch model with the dynamic embedding table.**:
    ```python
    sparse_feature: KeyedJaggedTensor = generate_sparse_feature(#your dataset)
    ret = model(sparse_feature)
    ```

## Future Plans

1. Support the latest version of TorchREC and continuously follow TorchREC's version updates.
2. Continuously optimize the performance of embedding lookup and embedding bag lookup.
3. Support multiple optimizer types, aligning with the optimizer types supported by TorchREC.
4. Support more configurations for dynamic embedding table eviction mechanisms and incremental dump.
5. Support the separation of backward and optimizer update (required by certain large language model frameworks like Megatron), to better support large-scale GR training.
6. Add more shard types for dynamic embedding tables, including `table-wise`, `table-row-wise` and `column-wise`.