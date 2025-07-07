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

# pyre-strict
from typing import Any, Dict, List, Set, Tuple, Type, Union

import torch
import torch.distributed as dist
import torchrec
from configs.task_config import OptimizerParam
from dynamicemb import DynamicEmbTableOptions
from dynamicemb.planner import DynamicEmbeddingEnumerator
from dynamicemb.planner import (
    DynamicEmbeddingShardingPlanner as DynamicEmbeddingShardingPlanner,
)
from dynamicemb.planner import DynamicEmbParameterConstraints
from dynamicemb.shard import (
    DynamicEmbeddingBagCollectionSharder,
    DynamicEmbeddingCollectionSharder,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType, SparseType
from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import (
    DistributedDataParallelConfig,
    finalize_model_grads,
)
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module
from modules.embedding import DataParallelEmbeddingCollection
from torch import distributed as dist
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torch.optim.optimizer import Optimizer
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.composable.table_batched_embedding_slice import (
    TableBatchedEmbeddingSlice,
)
from torchrec.distributed.embedding_types import ShardingType

# from torchrec.distributed import ModuleShardingPlan
from torchrec.distributed.fbgemm_qcomm_codec import (
    CommType,
    QCommsConfig,
    get_qcomm_codecs_registry,
)
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.planner import Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    ShardedTensor,
    ShardingEnv,
    ShardingType,
)
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.optim.optimizers import in_backward_optimizer_filter

TORCHREC_TYPES: Set[Type[Union[EmbeddingBagCollection, EmbeddingCollection]]] = {
    EmbeddingBagCollection,
    EmbeddingCollection,
}

DATA_PARALLEL_EMBEDDING_MODULE_NAME = "_data_parallel_embedding_collection"


def apply_megatron_ddp(
    dmp: DistributedModelParallel,
    config: TransformerConfig,
    dense_optimizer_param: OptimizerParam,
    device: torch.device,
):
    model = dmp._dmp_wrapped_module
    model = model.to(device)
    if config.fp16 or config.bf16:
        model = Float16Module(config, model)

    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
        check_for_nan_in_grad=False,
        bucket_size=True,
    )
    dmp._dmp_wrapped_module = DDP(
        config,
        ddp_config,
        model,
    )

    def broadcast_params_for_non_model_parallel_embedding_modules():
        for p in dmp._dmp_wrapped_module.parameters():
            if not isinstance(p, TableBatchedEmbeddingSlice):
                dist.broadcast(
                    p.data,
                    src=0,
                    group=parallel_state.get_data_parallel_group(
                        with_context_parallel=True
                    ),
                )

    broadcast_params_for_non_model_parallel_embedding_modules()
    config.finalize_model_grads_func = finalize_model_grads

    param_dtype = torch.float32
    if config.bf16:
        param_dtype = torch.bfloat16
    elif config.fp16:
        param_dtype = torch.float16

    dense_optimizer_config = OptimizerConfig(
        optimizer=dense_optimizer_param.optimizer_str,
        lr=dense_optimizer_param.learning_rate,
        adam_beta1=dense_optimizer_param.adam_beta1,
        adam_beta2=dense_optimizer_param.adam_beta2,
        adam_eps=dense_optimizer_param.adam_eps,
        params_dtype=param_dtype,
        bf16=config.bf16,
        fp16=config.fp16,
    )
    dense_optimizer = get_megatron_optimizer(
        dense_optimizer_config, [dmp._dmp_wrapped_module]
    )
    return dmp, dense_optimizer


# refer to https://github.com/pytorch/torchrec/blob/76a0826c6aec07c347f492aed2d4adf25cbdc3d9/torchrec/distributed/embedding_types.py#L75-L91
# compute_kernel is somehow coupled with sharding_type.
_pipeline_type_to_model_parallel_allowed_compute_kernels = {
    "prefetch": ["fused_uvm_caching"],
    "native": ["fused", "fused_uvm"],
    "none": [],  # none does not constrain the compute kernels
}
_pipeline_type_to_data_parallel_allowed_compute_kernels = {
    "prefetch": ["dense"],
    "native": ["dense"],
    "none": [],
}
_sharding_type_to_allowed_compute_kernels = {
    "data_parallel": _pipeline_type_to_data_parallel_allowed_compute_kernels,
    "model_parallel": _pipeline_type_to_model_parallel_allowed_compute_kernels,
}


def get_planner(
    eb_configs: List[EmbeddingConfig],
    data_parallel_embedding_table_names: Set[str],
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions],
    device: torch.device,
    pipeline_type: str = "none",
):
    constraints = {}
    for config in eb_configs:
        if config.name in data_parallel_embedding_table_names:
            compute_kernel_type = _sharding_type_to_allowed_compute_kernels[
                "data_parallel"
            ][pipeline_type]
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.DATA_PARALLEL.value,
                ],
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=False,
                compute_kernels=compute_kernel_type,
            )
        elif config.name in dynamicemb_options_dict:
            # TODO add dynamic embedding compute kernels
            compute_kernel_type = []
            dynamicemb_options = dynamicemb_options_dict[config.name]
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                bounds_check_mode=BoundsCheckMode.NONE,  # dynamic embedding has no bounding!
                enforce_hbm=True,
                use_dynamicemb=True,
                dynamicemb_options=dynamicemb_options,
            )
        else:
            compute_kernel_type = _sharding_type_to_allowed_compute_kernels[
                "model_parallel"
            ][pipeline_type]
            # TODO: save and load does not support table-wise sharding, disable them for now
            constraint = DynamicEmbParameterConstraints(
                sharding_types=[
                    ShardingType.ROW_WISE.value,
                    # ShardingType.TABLE_WISE.value,
                    # ShardingType.TABLE_ROW_WISE.value,
                ],
                bounds_check_mode=BoundsCheckMode.NONE,
                use_dynamicemb=False,
                compute_kernels=compute_kernel_type,
            )
        constraints.update({config.name: constraint})
    hbm_cap = torch.cuda.get_device_properties(0).total_memory
    ddr_cap = 512 * 1024 * 1024 * 1024  # Assume a Node have 512GB memory
    intra_host_bw = 450e9  # Nvlink bandwidth
    inter_host_bw = 25e9  # NIC bandwidth

    topology = Topology(
        local_world_size=get_local_size(),
        world_size=dist.get_world_size(),
        compute_device=device.type,
        hbm_cap=hbm_cap,
        ddr_cap=ddr_cap,  # For HVK  , if we need to put embedding vector into Host memory , it is important set ddr capacity
        intra_host_bw=intra_host_bw,
        inter_host_bw=inter_host_bw,
    )
    enumerator = DynamicEmbeddingEnumerator(
        topology=topology,
        constraints=constraints,
    )
    return DynamicEmbeddingShardingPlanner(
        eb_configs=eb_configs,
        topology=topology,
        constraints=constraints,
        enumerator=enumerator,
        storage_reservation=HeuristicalStorageReservation(percentage=0.05),
    )


_optimizer_str_to_optim_type = {
    "adam": EmbOptimType.ADAM,
    "sgd": EmbOptimType.EXACT_SGD,
    "row_wise_adagrad": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
}


def sparse_optimizer_factory_and_class(
    optimizer_name: str,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    momentum: float,
    learning_rate: float,
) -> Tuple[Type[Optimizer], Dict[str, Any]]:
    kwargs: Dict[str, Any] = {"lr": learning_rate}
    if optimizer_name == "adam":
        optimizer_cls = torchrec.optim.Adam
        beta1, beta2 = betas
        kwargs.update(
            {"beta1": beta1, "beta2": beta2, "eps": eps, "weight_decay": weight_decay}
        )
    elif optimizer_name == "sgd":
        optimizer_cls = torchrec.optim.SGD
        kwargs.update({"weight_decay": weight_decay, "momentum": momentum})
    elif optimizer_name == "row_wise_adagrad":
        optimizer_cls = torchrec.optim.RowWiseAdagrad
        beta1, beta2 = betas
        kwargs.update(
            {
                "eps": eps,
                "beta1": beta1,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    else:
        raise Exception("Unsupported optimizer!")

    return optimizer_cls, kwargs


def apply_dmp(
    model: torch.nn.Module,
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions],
    sparse_optimizer_param: OptimizerParam,
    pg: torch.distributed.ProcessGroup,
    device: torch.device,
    pipeline_type: str = "native",
):
    enable_prefetch_pipeline = pipeline_type == "prefetch"
    (
        sparse_opt_cls,
        sparse_opt_args,
    ) = sparse_optimizer_factory_and_class(
        optimizer_name=sparse_optimizer_param.optimizer_str,
        betas=(sparse_optimizer_param.adam_beta1, sparse_optimizer_param.adam_beta2),
        eps=sparse_optimizer_param.adam_eps,
        weight_decay=0.0,
        momentum=0.0,
        learning_rate=sparse_optimizer_param.learning_rate,
    )
    assert (
        sparse_optimizer_param.optimizer_str in _optimizer_str_to_optim_type
    ), f"embedding optimizer only support {list(_optimizer_str_to_optim_type.keys())}"
    fused_params = {
        "optimizer": _optimizer_str_to_optim_type[sparse_optimizer_param.optimizer_str],
        "learning_rate": sparse_optimizer_param.learning_rate,
        "beta1": sparse_optimizer_param.adam_beta1,
        "beta2": sparse_optimizer_param.adam_beta2,
        "eps": sparse_optimizer_param.adam_eps,
        # 'weight_decay_mode' : WeightDecayMode.NONE,
        # TODO, expose below params to users
        "output_dtype": SparseType.FP32,
        # only when compute kernel is FUSED_UVM_CACHING or KEY_VALUE are the below params effective.
        "cache_precision": SparseType.FP32,
        "stochastic_rounding": False,
        "prefetch_pipeline": enable_prefetch_pipeline,
    }
    eb_configs = []
    data_parallel_embedding_table_names = []
    data_parallel_embedding_module_names = []
    for k, module in model.named_modules():
        if type(module) in TORCHREC_TYPES:
            for _, param in module.named_parameters(prefix=k):
                if param.requires_grad:
                    apply_optimizer_in_backward(
                        sparse_opt_cls, [param], sparse_opt_args
                    )
            eb_configs.extend(module.embedding_configs())
            if DATA_PARALLEL_EMBEDDING_MODULE_NAME in k:
                data_parallel_embedding_module_names.append(k)
                for config in module.embedding_configs():
                    data_parallel_embedding_table_names.append(config.name)

    planner = get_planner(
        eb_configs,
        set(data_parallel_embedding_table_names),
        dynamicemb_options_dict,
        device,
        pipeline_type,
    )
    qcomm_codecs_registry = get_qcomm_codecs_registry(
        qcomms_config=QCommsConfig(
            forward_precision=CommType.FP32,
            backward_precision=CommType.FP32,
        )
    )
    sharders = [
        DynamicEmbeddingBagCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            fused_params=fused_params,
        ),
        DynamicEmbeddingCollectionSharder(
            qcomm_codecs_registry=qcomm_codecs_registry,
            use_index_dedup=True,
            fused_params=fused_params,
        ),
    ]
    plan = planner.collective_plan(model, sharders, pg)
    data_parallel_sharding_plans = []
    for data_parallel_embedding_module_name in data_parallel_embedding_module_names:
        data_parallel_sharding_plans.append(
            plan.plan.pop(data_parallel_embedding_module_name, None)
        )
    # Shard model
    with tensor_parallel.get_cuda_rng_tracker().fork("sharded-embedding-group-seed"):
        model = DistributedModelParallel(
            module=model,
            env=ShardingEnv.from_process_group(pg),
            device=device,
            sharders=sharders,
            plan=plan,
            init_data_parallel=False,
        )

    # Create keyed optimizer
    non_fused_sparse_params = {}
    for k, v in in_backward_optimizer_filter(model.named_parameters()):
        if v.requires_grad:
            if isinstance(v, ShardedTensor):
                non_fused_sparse_params[k] = v
    assert len(non_fused_sparse_params) == 0, "non_fused_sparse_params should be empty"

    if len(data_parallel_sharding_plans) > 0:
        unwrapped_model = model.module
        for dp_module_name, dp_sharding_plan in zip(
            data_parallel_embedding_module_names, data_parallel_sharding_plans
        ):
            data_parallel_embedding_collection_father_module_name = (
                dp_module_name.split(".")[:-1]
            )
            father_module = unwrapped_model
            for name in data_parallel_embedding_collection_father_module_name:
                father_module = getattr(father_module, name)
            data_parallel_embedding_collection = getattr(
                father_module, DATA_PARALLEL_EMBEDDING_MODULE_NAME
            )
            setattr(
                father_module,
                DATA_PARALLEL_EMBEDDING_MODULE_NAME,
                DataParallelEmbeddingCollection(
                    data_parallel_embedding_collection,
                    dp_sharding_plan,
                    ShardingEnv.from_process_group(pg),
                    fused_params,
                    device,
                ),
            )
    return model


def make_optimizer_and_shard(
    model: torch.nn.Module,
    config: TransformerConfig,
    sparse_optimizer_param: OptimizerParam,
    dense_optimizer_param: OptimizerParam,
    dynamicemb_options_dict: Dict[str, DynamicEmbTableOptions] = {},
    pipeline_type: str = "native",
    device: torch.device = None,
    pg: torch.distributed.ProcessGroup = None,
) -> Tuple[DistributedModelParallel, torch.optim.Optimizer]:
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if pg is None:
        pg = dist.group.WORLD

    model = apply_dmp(
        model,
        dynamicemb_options_dict,
        sparse_optimizer_param,
        pg,
        device,
        pipeline_type,
    )
    model, dense_optimizer = apply_megatron_ddp(
        model, config, dense_optimizer_param, device
    )

    return model, dense_optimizer
