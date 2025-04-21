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
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)

import os

import torch
from dynamicemb import DynamicEmbEvictStrategy

import distributed_recommender
import utils.initialize as init
from configs import (
    DynamicShardedEmbeddingConfig,
    KernelBackend,
    RankingConfig,
    ShardedEmbeddingConfig,
    get_hstu_config,
)
from data.utils import RankingBatch
from model.ranking_gr import RankingGR
from modules.embedding import EmbeddingOptimizerParam
from commons.utils.tensor_initializer import UniformInitializer

init.initialize_distributed()
init.initialize_model_parallel()
init.set_random_seed(1234)

device = torch.cuda.current_device()

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

task_config = RankingConfig(
    embedding_configs=emb_configs,
    prediction_head_arch=[[128, 10, 1] for _ in range(1)],
)
ranking_model_train = RankingGR(hstu_config=hstu_config, task_config=task_config)

batch = RankingBatch.random(
    batch_size=128,
    feature_configs=[
        data.utils.FeatureConfig(
            feature_names=["user_feat0"],
            max_item_ids=[user_vocab_size],
            max_sequence_length=10,
            is_jagged=True,
        ),
        data.utils.FeatureConfig(
            feature_names=["user_feat1"],
            max_item_ids=[user_vocab_size],
            max_sequence_length=10,
            is_jagged=True,
        ),
        data.utils.FeatureConfig(
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

save_path = "./checkpoint"
os.makedirs(save_path, exist_ok=True)
checkpoint.save(save_path, ranking_model_train)
checkpoint.load(save_path, ranking_model_train)
