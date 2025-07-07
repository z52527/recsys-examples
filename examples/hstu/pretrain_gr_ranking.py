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
import argparse
from dataclasses import dataclass
from functools import partial  # pylint: disable-unused-import
from typing import List, Tuple, cast

import commons.utils.initialize as init
import gin
import torch  # pylint: disable-unused-import
from commons.utils.logger import print_rank_0
from configs import RankingConfig
from distributed.sharding import make_optimizer_and_shard
from megatron.core import parallel_state
from model import get_ranking_model
from modules.metrics import get_multi_event_metric_module
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from training import (
    NetworkArgs,
    OptimizerArgs,
    TensorModelParallelArgs,
    TrainerArgs,
    create_dynamic_optitons_dict,
    create_embedding_config,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader,
    get_dataset_and_embedding_args,
    maybe_load_ckpts,
    train_with_pipeline,
)


@gin.configurable
@dataclass
class RankingArgs:
    prediction_head_arch: List[int] = cast(List[int], None)
    prediction_head_act_type: str = "relu"
    prediction_head_bias: bool = True
    num_tasks: int = 1
    eval_metrics: Tuple[str, ...] = ("AUC",)

    def __post_init__(self):
        assert (
            self.prediction_head_arch is not None
        ), "Please provide prediction head arch for ranking model"
        if isinstance(self.prediction_head_act_type, str):
            assert self.prediction_head_act_type.lower() in [
                "relu",
                "gelu",
            ], "prediction_head_act_type should be in ['relu', 'gelu']"


parser = argparse.ArgumentParser(
    description="Distributed GR Arguments", allow_abbrev=False
)
parser.add_argument("--gin-config-file", type=str)
args = parser.parse_args()
gin.parse_config_file(args.gin_config_file)
trainer_args = TrainerArgs()
dataset_args, embedding_args = get_dataset_and_embedding_args()
network_args = NetworkArgs()
optimizer_args = OptimizerArgs()
tp_args = TensorModelParallelArgs()


def create_ranking_config() -> RankingConfig:
    ranking_args = RankingArgs()

    return RankingConfig(
        embedding_configs=[
            create_embedding_config(network_args.hidden_size, arg)
            for arg in embedding_args
        ],
        prediction_head_arch=ranking_args.prediction_head_arch,
        prediction_head_act_type=ranking_args.prediction_head_act_type,
        prediction_head_bias=ranking_args.prediction_head_bias,
        num_tasks=ranking_args.num_tasks,
        eval_metrics=ranking_args.eval_metrics,
    )


def main():
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"distributed env initialization done. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )
    hstu_config = create_hstu_config(network_args)
    task_config = create_ranking_config()
    model = get_ranking_model(hstu_config=hstu_config, task_config=task_config)

    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args, network_args.hidden_size
    )

    optimizer_param = create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )

    stateful_metric_module = get_multi_event_metric_module(
        num_classes=task_config.prediction_head_arch[-1],
        num_tasks=task_config.num_tasks,
        metric_types=task_config.eval_metrics,
        comm_pg=parallel_state.get_data_parallel_group(
            with_context_parallel=True
        ),  # ranks in the same TP group do the same compute
    )

    train_dataloader, test_dataloader = get_data_loader(
        "ranking", dataset_args, trainer_args, task_config.num_tasks
    )
    free_memory, total_memory = torch.cuda.mem_get_info()
    print_rank_0(
        f"model initialization done, start training. Free cuda memory: {free_memory / (1024 ** 2):.2f} MB"
    )

    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)
    if trainer_args.pipeline_type in ["prefetch", "native"]:
        pipeline_factory = (
            JaggedMegatronPrefetchTrainPipelineSparseDist
            if trainer_args.pipeline_type == "prefetch"
            else JaggedMegatronTrainPipelineSparseDist
        )
        pipeline = pipeline_factory(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    else:
        pipeline = JaggedMegatronTrainNonePipeline(
            model_train,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    train_with_pipeline(
        pipeline,
        stateful_metric_module,
        trainer_args,
        train_dataloader,
        test_dataloader,
        dense_optimizer,
    )
    init.destroy_global_state()


if __name__ == "__main__":
    main()
