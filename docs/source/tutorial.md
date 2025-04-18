# Tutorial

## Generative Recommender Introduction
Meta's paper ["Actions Speak Louder Than Words"](https://arxiv.org/abs/2402.17152) introduces a novel paradigm for recommendation systems called **Generative Recommenders(GRs)**, which reformulates recommendation tasks as generative modeling problems. The work introduced Hierarchical Sequential Transduction Units (HSTU), a novel architecture designed to handle high-cardinality, non-stationary data streams in large-scale recommendation systems. HSTU enables both retrieval and ranking tasks. As noted in the paper, “HSTU-based GRs, with 1.5 trillion parameters, improve metrics in online A/B tests by 12.4% and have been deployed on multiple surfaces of a large internet platform with billions of users.”
While **distributed-recommender** supports both retrieval and ranking use cases, in this tutorial, we will only guide you through the process of building a generative recommender for ranking tasks and training it effectively.

## Ranking Model Introduction
The model structure of the ranking model can be depicted by the following picture.
![ranking model structure](./ranking_model_structure.png)

### Input
The input to the HSTU model consists solely of pure categorical features, and it does not accommodate numerical features. The model supports three types of tokens:
* Contextual Tokens: Represent the user side info.
* Item Tokens: Represent the items being recommended.
* Action Tokens: Optional. Represent user actions associated with these items. Please note that if a user has multiple actions associated with a single item token, these actions must be merged into a single token during data preprocessing. For further details, please refer to [the related issue](https://github.com/facebookresearch/generative-recommenders/issues/114).

It is crucial that the number of item tokens matches the number of action tokens. This alignment ensures that each item can be effectively paired with its corresponding user action, as the paper said.

### Embedding Table
The embedding mechanism includes three types of distinct tables:
* Contextual Embedding Table: Corresponds to contextual tokens.
* Item Embedding Table: Corresponds to item tokens.
* Action Embedding Table: Corresponds to action tokens if provided.

### HSTU Block
The HSTU block is a core component of the architecture, which modifies traditional attention mechanisms to effectively handle large, non-stationary vocabularies typical in recommendation systems. 
* **Preprocessing**: After retrieving the embedding vectors from the tables, the HSTU preprocessing stage begins. If action embeddings are provided, the model interleaves the item and action embedding vectors. It then concatenates the contextual embeddings with the interleaved item and action embeddings, ensuring that each sample starts with contextual embeddings followed by item and action sequence pairs. Finally, the model applies position encoding.

* **Postprocessing**: If candidate items are specified, the model predicts only these candidates by filtering candidate item embeddings in the postprocessing. Otherwise, all item embeddings will be selected to be used for prediction.

### Prediction Head
The prediction head of the HSTU model employs multiple MLP network structure, enabling multi-task predictions. 

## Parallelism for HSTU-based Generative Recommender
Scaling is a crucial factor for HSTU-based GRs due to their demonstrated superior scalability compared to traditional Deep Learning Recommendation Models (DLRMs). According to the paper, while DLRMs plateau at around 200 billion parameters, GRs can scale up to 1.5 trillion parameters, resulting in improved model accuracy.

However, achieving efficient scaling for GRs presents unique challenges. Existing libraries designed for large-scale training in LLMs or recommendation systems often fail to meet the specific needs of GRs:
* **[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)**, which supports advanced parallelism (e.g Data, Tensor, Sequence, Pipeline, and Context parallelism), is not well-suited for recommendation systems due to their reliance on massive embedding tables that cannot be effectively handled by existing parallelism.
* **[TorchRec](https://github.com/pytorch/torchrec)**, while providing solutions for sharding large embedding tables across GPUs, lacks robust support for dense model parallelism. This makes it difficult for users to combine embedding and dense parallelism without significant design effort

To address these limitations, a hybrid approach combining sparse and dense parallelism is introduced in the **distributed-recommender**. As the pic shows.
**TorchRec** is employed to shard large embedding tables effectively.
**Megatron-LM** is used to support data and tensor parallelism for the dense components of the model. Context parallelism is also planned as part of future development.
This integration ensures efficient training by coordinating sparse (embedding) and dense (tensor/data) parallelisms within a single model.
![parallelism](./parallelism.png)

## Build a HSTU model for ranking
You can use the following command to run the [tutorial.py](../../tutorial.py) with 1GPU.

```
torchrun --nproc_per_node 1 --master_addr localhost --master_port 6000 tutorial.py
```
### Environment Setup
You can refer to Installation in the [tutorial.py](../../README.md) to set up the environment.
```python
import os

import torch

import distributed_recommender
import distributed_recommender.utils.initialize as init
from distributed_recommender.configs import (
    DynamicShardedEmbeddingConfig,
    KernelBackend,
    RankingConfig,
    ShardedEmbeddingConfig,
    get_hstu_config,
)
from distributed_recommender.data.utils import RankingBatch
from distributed_recommender.model.ranking_gr import RankingGR
from distributed_recommender.modules.embedding import EmbeddingOptimizerParam
from distributed_recommender.utils.tensor_initializer import UniformInitializer

init.initialize_distributed()
init.initialize_model_parallel()
init.set_random_seed(1234)
```
### Embedding configuration
**distributed-recommender** provides two primary types of embedding tables: the **Embedding Table** and the **Dynamic Embedding Table**. Each type possesses distinct characteristics and configurations tailored to various use cases in recommendation tasks.
* The **Embedding Table** serves as a wrapper for the embedding table in **TorchRec**. It has a fixed shape and accepts only indices for lookups. Users can choose between data-parallel and row-wise model-parallel sharding methods. To configure this embedding table, users can utilize the ShardedEmbeddingConfig.
* The [Dynamic Embedding Table](./dynamicemb_readme.md) is powered by the [HierarchicalKV](https://github.com/NVIDIA-Merlin/HierarchicalKV), refer to  offering several key features:
  * **Support for Hashing and Eviction**: This capability allows users to train on large feature fields without encountering out-of-memory (OOM) issues. The dynamic nature facilitates efficient management of embeddings by evicting less important embeddings from memory. User can configure different eviction strategy.
  * **CPU Offloading**: Users can leverage CPU memory to store parts of the embedding table, thereby alleviating memory constraints on GPUs.
  * **Row-Wise Sharding Across GPUs**: Since dynamic Embedding Table targets gigantic embedding table use cases, it only supports row-wise sharding.

You can refer [API book](./distributed_recommender) for more detailed configuration use case 
```python
from dynamicemb import DynamicEmbEvictStrategy
user_vocab_size = 1000
item_vocab_size = 1000
action_vocab_size = 10
dim_size = 128

embedding_optimizer_param = EmbeddingOptimizerParam(
    optimizer_str="adam", learning_rate=0.0001
)
emb_configs = [
    ShardedEmbeddingConfig(
        feature_names=["user_feat0"],
        table_name="user_table0",
        vocab_size=user_vocab_size,
        dim=dim_size,
        sharding_type="model_parallel",
        initializer=UniformInitializer(),
        optimizer_param=embedding_optimizer_param,
    ),
    ShardedEmbeddingConfig(
        feature_names=["user_feat1"],
        table_name="user_table1",
        vocab_size=user_vocab_size,
        dim=dim_size,
        sharding_type="model_parallel",
        initializer=UniformInitializer(),
        optimizer_param=embedding_optimizer_param,
    ),
    ShardedEmbeddingConfig(
        feature_names=["act_feat"],
        table_name="act",
        vocab_size=action_vocab_size,
        dim=dim_size,
        sharding_type="data_parallel",
        initializer=UniformInitializer(),
        optimizer_param=embedding_optimizer_param,
    ),
    DynamicShardedEmbeddingConfig(
        feature_names=["item_feat"],
        table_name="item",
        vocab_size=item_vocab_size,
        dim=dim_size,
        initializer=UniformInitializer(),
        optimizer_param=embedding_optimizer_param,
        global_hbm_for_values=0,
        evict_strategy=DynamicEmbEvictStrategy.LRU,
    ),
]
```

### Dense model configuration
In the dense part of the **distributed-recommender**, three different kernel backends are supported for HSTU attention and in this tutorial we use `CUTLASS backend`:
* `PyTorch Backend`: The HSTU attention is implemented entirely using PyTorch operations. This backend supports multiple precisions, including `float32`, `float16`, and `bfloat16`. It can serve as a reference for users who want to verify accuracy before proceeding.
* `Triton Backend`: The HSTU attention is implemented using `Triton`, based on [the official paper's open-sourced repository](https://github.com/facebookresearch/generative-recommenders/tree/main/generative_recommenders/ops/triton). `Triton` offers competitive performance compared to the PyTorch implementation and allows for easier modifications. Users with customized requirements for altering the HSTU attention structure can modify the source code accordingly. This implementation supports only `float16` and `bfloat16` and is provided without further optimization currently.
* `CUTLASS Backend`: This backend utilizes CUTLASS and is designed to deliver the fastest implementation of HSTU attention. We recommend this backend for users conducting real training and it’s used by default. Currently, it only supports the official HSTU attention structure and is limited to `float16` and `bfloat16`.

```python
hstu_config = get_hstu_config(
    hidden_size=dim_size,
    kv_channels=128,
    num_attention_heads=4,
    num_layers=3,
    init_method=UniformInitializer(),
    dtype=torch.bfloat16,
    is_causal=True,
    kernel_backend=KernelBackend.CUTLASS,
)
```

### Build ranking model
To configure the ranking model, users need to set up the `embedding_configs`, and `prediction_head` using the `RankingConfig`. After that, you can create an instance of `RankingGR` and perform forward and backward passes. Note that this tutorial does not include an optimizer. In a training scenario, since the model contains both embedding and dense modules, each component must be optimized separately. Users can refer to the [example](../../examples/hstu/README.md) for a detailed training loop.

```python
task_config = RankingConfig(
    embedding_configs=emb_configs,
    prediction_head_arch=[[128, 10, 1] for _ in range(1)],
)
ranking_model_train = RankingGR(hstu_config=hstu_config, task_config=task_config)

batch = RankingBatch.random(
    batch_size=128,
    feature_configs=[
        distributed_recommender.data.utils.FeatureConfig(
            feature_names=["user_feat0"],
            max_item_ids=[user_vocab_size],
            max_sequence_length=10,
            is_jagged=True,
        ),
        distributed_recommender.data.utils.FeatureConfig(
            feature_names=["user_feat1"],
            max_item_ids=[user_vocab_size],
            max_sequence_length=10,
            is_jagged=True,
        ),
        distributed_recommender.data.utils.FeatureConfig(
            feature_names=["item_feat", "act_feat"],
            max_item_ids=[item_vocab_size, action_vocab_size],
            max_sequence_length=180,
            is_jagged=True,
        ),
    ],
    item_feature_name="item_feat",
    contextual_feature_names=["user_feat0", "user_feat1"],
    action_feature_name="act_feat",
    max_num_candidates=20,
    device=device,
    num_tasks=1,
)
loss, _ = ranking_model_train(batch)
loss.sum().backward()

init.destroy_global_state()
```

### Checkpointing
Users can utilize the `save` and `load` functions from **distributed-recommender** to manage model checkpointing. These APIs will iterate over the specified modules and save them according to the following criteria:
* If the module is a dynamic embedding module, it will be saved in a folder named `dynamicemb_module`. Different ranks will collectively save the embedding table into a single file. This implementation also supports resharding when using different ranks for loading.
* If the module is not a dynamic embedding module, it will be saved in a folder named `torch_module`, with each rank saving to a separate file. During loading, each rank must load its corresponding saved file, and resharding is not supported.
```python
save_path = "./checkpoint"
os.makedirs(save_path, exist_ok=True)
distributed_recommender.checkpoint.save(save_path, ranking_model_train)
distributed_recommender.checkpoint.load(save_path, ranking_model_train)
```
