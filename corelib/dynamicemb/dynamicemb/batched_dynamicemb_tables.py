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

import enum
import warnings
from dataclasses import dataclass, field
from itertools import accumulate
from typing import List, Optional, Tuple

import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.batched_dynamicemb_function import *
from dynamicemb.dynamicemb_config import *
from dynamicemb.optimizer import *
from dynamicemb.unique_op import UniqueOp
from dynamicemb_extensions import (
    DynamicEmbTable,
    OptimizerType,
    count_matched,
    device_timestamp,
    dyn_emb_capacity,
    dyn_emb_cols,
    export_batch_matched,
)
from torch import Tensor, nn  # usort:skip


@enum.unique
class SparseType(enum.Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    INT2 = "int2"
    BF16 = "bf16"


class BoundsCheckMode(enum.IntEnum):
    # Raise an exception (CPU) or device-side assert (CUDA)
    FATAL = 0
    # Log the first out-of-bounds instance per kernel, and set to zero.
    WARNING = 1
    # Set to zero.
    IGNORE = 2
    # No bounds checks.
    NONE = 3


class WeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2
    COUNTER = 3
    COWCLIP = 4


class CounterWeightDecayMode(enum.IntEnum):
    NONE = 0
    L2 = 1
    DECOUPLE = 2


class LearningRateMode(enum.IntEnum):
    EQUAL = -1
    TAIL_ID_LR_INCREASE = 0
    TAIL_ID_LR_DECREASE = 1
    COUNTER_SGD = 2


class GradSumDecay(enum.IntEnum):
    NO_DECAY = -1
    CTR_DECAY = 0


@dataclass
class TailIdThreshold:
    val: float = 0
    is_ratio: bool = False


@dataclass
class CounterBasedRegularizationDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    adjustment_iter: int = -1
    adjustment_ub: float = 1.0
    learning_rate_mode: LearningRateMode = LearningRateMode.EQUAL
    grad_sum_decay: GradSumDecay = GradSumDecay.NO_DECAY
    tail_id_threshold: TailIdThreshold = field(default_factory=TailIdThreshold)
    max_counter_update_freq: int = 1000


@dataclass
class CowClipDefinition:
    counter_weight_decay_mode: CounterWeightDecayMode = CounterWeightDecayMode.NONE
    counter_halflife: int = -1
    weight_norm_coefficient: float = 0.0
    lower_bound: float = 0.0


def _export_matched_and_gather(
    dynamic_table: DynamicEmbTable,
    threshold: int,
    pg: Optional[dist.ProcessGroup] = None,
    batch_size: int = BATCH_SIZE_PER_DUMP,
) -> Tuple[Tensor, Tensor]:
    # Get the rank of the current process
    rank = dist.get_rank(group=pg)
    world_size = dist.get_world_size(group=pg)
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    d_num_matched = torch.zeros(1, dtype=torch.uint64, device=device)
    count_matched(dynamic_table, threshold, d_num_matched)

    gathered_num_matched = [
        torch.tensor(0, dtype=torch.int64, device=device) for _ in range(world_size)
    ]
    dist.all_gather(gathered_num_matched, d_num_matched.to(dtype=torch.int64), group=pg)

    total_matched = sum([t.item() for t in gathered_num_matched])  # t is on device.
    key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
    value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
    dim: int = dyn_emb_cols(dynamic_table)

    ret_keys = torch.empty(total_matched, dtype=key_dtype, device="cpu")
    ret_vals = torch.empty(total_matched * dim, dtype=value_dtype, device="cpu")
    ret_offset = 0

    search_offset = 0
    search_capacity = dyn_emb_capacity(dynamic_table)
    batch_size = batch_size if batch_size < search_capacity else search_capacity

    d_keys = torch.empty(batch_size, dtype=key_dtype, device=device)
    d_vals = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
    d_count = torch.zeros(1, dtype=torch.uint64, device=device)

    # Gather keys and values for all ranks
    gathered_keys = [torch.empty_like(d_keys) for _ in range(world_size)]
    gathered_vals = [torch.empty_like(d_vals) for _ in range(world_size)]
    gathered_counts = [
        torch.empty_like(d_count, dtype=torch.int64) for _ in range(world_size)
    ]

    while search_offset < search_capacity:
        export_batch_matched(
            dynamic_table, threshold, batch_size, search_offset, d_count, d_keys, d_vals
        )

        dist.all_gather(gathered_keys, d_keys, group=pg)
        dist.all_gather(gathered_vals, d_vals, group=pg)
        dist.all_gather(gathered_counts, d_count.to(dtype=torch.int64), group=pg)

        for d_keys_, d_vals_, d_count_ in zip(
            gathered_keys, gathered_vals, gathered_counts
        ):
            h_count = d_count_.cpu().item()
            ret_keys[ret_offset : ret_offset + h_count] = d_keys_[0:h_count].cpu()
            ret_vals[ret_offset * dim : (ret_offset + h_count) * dim] = d_vals_[
                0 : h_count * dim
            ].cpu()
            ret_offset += h_count

        search_offset += batch_size
        d_count.fill_(0)

    return ret_keys, ret_vals


def _export_matched(
    dynamic_table: DynamicEmbTable,
    threshold: int,
    batch_size: int = BATCH_SIZE_PER_DUMP,
) -> Tuple[Tensor, Tensor]:
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    d_num_matched = torch.zeros(1, dtype=torch.uint64, device=device)
    count_matched(dynamic_table, threshold, d_num_matched)

    total_matched = d_num_matched.cpu().item()
    key_dtype = dyn_emb_to_torch(dynamic_table.key_type())
    value_dtype = dyn_emb_to_torch(dynamic_table.value_type())
    dim: int = dyn_emb_cols(dynamic_table)

    ret_keys = torch.empty(total_matched, dtype=key_dtype, device="cpu")
    ret_vals = torch.empty(total_matched * dim, dtype=value_dtype, device="cpu")
    ret_offset = 0

    search_offset = 0
    search_capacity = dyn_emb_capacity(dynamic_table)
    batch_size = batch_size if batch_size < search_capacity else search_capacity

    d_keys = torch.empty(batch_size, dtype=key_dtype, device=device)
    d_vals = torch.empty(batch_size * dim, dtype=value_dtype, device=device)
    d_count = torch.zeros(1, dtype=torch.uint64, device=device)

    while search_offset < search_capacity:
        export_batch_matched(
            dynamic_table, threshold, batch_size, search_offset, d_count, d_keys, d_vals
        )

        h_count = d_count.cpu().item()
        ret_keys[ret_offset : ret_offset + h_count] = d_keys[0:h_count].cpu()
        ret_vals[ret_offset * dim : (ret_offset + h_count) * dim] = d_vals[
            0 : h_count * dim
        ].cpu()
        ret_offset += h_count

        search_offset += batch_size
        d_count.fill_(0)

    return ret_keys, ret_vals


class BatchedDynamicEmbeddingTables(nn.Module):
    """
    Dynamic Embedding is based on [HKV](https://github.com/NVIDIA-Merlin/HierarchicalKV/tree/master).
    Looks up one or more dynamic embedding tables. The module is application for training.

    Its optional to fuse the optimizer with the backward operator by parameter *update_grads_explicitly*.
    """

    optimizer_args: OptimizerArgs

    def __init__(
        self,
        table_options: List[DynamicEmbTableOptions],
        table_names: Optional[List[str]] = None,
        feature_table_map: Optional[List[int]] = None,  # [T]
        use_index_dedup: bool = False,
        pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.SUM,
        output_dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        enforce_hbm: bool = False,  # place all weights/momentums in HBM when using cache
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        optimizer: EmbOptimType = EmbOptimType.SGD,
        # General Optimizer args
        stochastic_rounding: bool = True,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        max_norm: float = 0.0,
        learning_rate: float = 0.01,
        # used by EXACT_ADAGRAD, EXACT_ROWWISE_ADAGRAD, EXACT_ROWWISE_WEIGHTED_ADAGRAD, LAMB, and ADAM only
        # NOTE that default is different from nn.optim.Adagrad default of 1e-10
        eps: float = 1.0e-8,
        # used by EXACT_ADAGRAD, EXACT_ROWWISE_ADAGRAD, and EXACT_ROWWISE_WEIGHTED_ADAGRAD only
        initial_accumulator_value: float = 0.0,
        momentum: float = 0.9,  # used by LARS-SGD
        # EXACT_ADAGRAD, SGD, EXACT_SGD do not support weight decay
        # LAMB, ADAM, PARTIAL_ROWWISE_ADAM, PARTIAL_ROWWISE_LAMB, LARS_SGD support decoupled weight decay
        # EXACT_ROWWISE_WEIGHTED_ADAGRAD supports L2 weight decay
        # EXACT_ROWWISE_ADAGRAD support both L2 and decoupled weight decay (via weight_decay_mode)
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
        eta: float = 0.001,  # used by LARS-SGD,
        beta1: float = 0.9,  # used by LAMB and ADAM
        beta2: float = 0.999,  # used by LAMB and ADAM
        counter_based_regularization: Optional[
            CounterBasedRegularizationDefinition
        ] = None,  # used by Rowwise Adagrad
        cowclip_regularization: Optional[
            CowClipDefinition
        ] = None,  # used by Rowwise Adagrad
        # Frequency masking parameters
        frequency_threshold: int = 0,  # Frequency threshold for masking
        mask_dims: int = 0,  # Number of dimensions to mask
    ) -> None:
        super().__init__()
        assert len(table_options) >= 1
        table_option = table_options[0]
        for other_option in table_options:
            assert (
                table_option == other_option
            ), "All tables must match in grouped keys."
        self._dynamicemb_options = table_options
        self.initializer_args = table_option.initializer_args
        self.index_type = table_option.index_type
        self.embedding_dtype = table_option.embedding_dtype
        self.output_dtype = output_dtype
        self.pooling_mode = pooling_mode
        self.use_index_dedup = use_index_dedup
        self._table_names = table_names
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self._create_score()
        self.frequency_threshold = frequency_threshold
        self.mask_dims = mask_dims

        if device is not None:
            self.device_id = int(str(device)[-1])
        else:
            assert torch.cuda.is_available(), "No available CUDA device."
            self.device_id = torch.cuda.current_device()

        if table_option.device_id is None:
            for option in self._dynamicemb_options:
                option.device_id = self.device_id
        # get cuda device config
        device_properties = torch.cuda.get_device_properties(self.device_id)
        self._device_num_sms = device_properties.multi_processor_count

        self.dims: List[int] = [option.dim for option in self._dynamicemb_options]
        # mixed D is not supported by sequence embedding.
        mixed_D = False
        D = self.dims[0]
        for d in self.dims:
            if d != D:
                mixed_D = True
                break
        if mixed_D:
            assert (
                self.pooling_mode != DynamicEmbPoolingMode.NONE
            ), "Mixed dimension tables only supported for pooling tables."

        # physical table number.
        T_ = len(self._dynamicemb_options)
        assert T_ > 0
        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        # logical table number.
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        feature_dims = [self.dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(feature_dims))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(self.dims)

        self.feature_num = len(self.feature_table_map)
        # TODO:deal with shuffeld feature_table_map
        self.table_offsets_in_feature: List[int] = []
        old_table_id = -1
        for idx, table_id in enumerate(self.feature_table_map):
            if table_id != old_table_id:
                self.table_offsets_in_feature.append(idx)
                old_table_id = table_id
        self.table_offsets_in_feature.append(self.feature_num)

        for option in self._dynamicemb_options:
            if option.init_capacity is None:
                option.init_capacity = option.max_capacity
        self._optimizer_type = optimizer
        self._tables: List[DynamicEmbTable] = []
        self._create_tables()
        # add placeholder require_grad param tensor to enable autograd with int8 weights
        # self.placeholder_autograd_tensor = nn.Parameter(
        #     torch.zeros(0, device=torch.device(self.device_id), dtype=torch.float)
        # )

        # TODO: review this code block
        self.stochastic_rounding = stochastic_rounding

        self.weight_decay_mode = weight_decay_mode
        if (weight_decay_mode == WeightDecayMode.COUNTER) != (
            counter_based_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COUNTER together with valid counter_based_regularization"
            )
        if (weight_decay_mode == WeightDecayMode.COWCLIP) != (
            cowclip_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COWCLIP together with valid cowclip_regularization"
            )

        self._used_rowwise_adagrad_with_counter: bool = (
            optimizer == EmbOptimType.EXACT_ROWWISE_ADAGRAD
            and (
                weight_decay_mode in (WeightDecayMode.COUNTER, WeightDecayMode.COWCLIP)
            )
        )

        if counter_based_regularization is None:
            counter_based_regularization = CounterBasedRegularizationDefinition()
        if cowclip_regularization is None:
            cowclip_regularization = CowClipDefinition()
        self._max_counter_update_freq: int = -1
        # Extract parameters from CounterBasedRegularizationDefinition or CowClipDefinition
        # which are passed as entries for OptimizerArgs
        if self._used_rowwise_adagrad_with_counter:
            if self.weight_decay_mode == WeightDecayMode.COUNTER:
                self._max_counter_update_freq = (
                    counter_based_regularization.max_counter_update_freq
                )
                opt_arg_weight_decay_mode = (
                    counter_based_regularization.counter_weight_decay_mode
                )
                counter_halflife = counter_based_regularization.counter_halflife
            else:
                opt_arg_weight_decay_mode = (
                    cowclip_regularization.counter_weight_decay_mode
                )
                counter_halflife = cowclip_regularization.counter_halflife
        else:
            opt_arg_weight_decay_mode = weight_decay_mode
            # Default: -1, no decay applied, as a placeholder for OptimizerArgs
            # which should not be effective when CounterBasedRegularizationDefinition
            # and CowClipDefinition are not used
            counter_halflife = -1

        optimizer_args = OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            max_norm=max_norm,
            learning_rate=learning_rate,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            weight_decay_mode=opt_arg_weight_decay_mode.value,
            eta=eta,
            momentum=momentum,
            counter_halflife=counter_halflife,
            adjustment_iter=counter_based_regularization.adjustment_iter,
            adjustment_ub=counter_based_regularization.adjustment_ub,
            learning_rate_mode=counter_based_regularization.learning_rate_mode.value,
            grad_sum_decay=counter_based_regularization.grad_sum_decay.value,
            tail_id_threshold=counter_based_regularization.tail_id_threshold.val,
            is_tail_id_thresh_ratio=int(
                counter_based_regularization.tail_id_threshold.is_ratio
            ),
            total_hash_size=0,
            weight_norm_coefficient=cowclip_regularization.weight_norm_coefficient,
            lower_bound=cowclip_regularization.lower_bound,
            regularization_mode=weight_decay_mode.value,
        )
        self._optimizer: BaseDynamicEmbeddingOptimizer = None
        self._create_optimizer(optimizer, optimizer_args)

        # TODO:1->10
        self._empty_tensor = nn.Parameter(
            torch.empty(
                10,
                requires_grad=True,
                device=torch.device(self.device_id),
                dtype=self.embedding_dtype,
            )
        )

        # new a unique op
        # TODO: in our case maybe we can use torch.uint32
        reserve_keys = torch.tensor(
            2, dtype=self.index_type, device=torch.device(self.device_id)
        )
        reserve_vals = torch.tensor(
            2, dtype=torch.uint64, device=torch.device(self.device_id)
        )
        counter = torch.tensor(
            1, dtype=torch.uint64, device=torch.device(self.device_id)
        )
        self._unique_op = UniqueOp(reserve_keys, reserve_vals, counter, 2)

    def _create_tables(self) -> None:
        for option in self._dynamicemb_options:
            if option.training:
                if self._optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
                    option.optimizer_type = OptimizerType.RowWiseAdaGrad
                elif (
                    self._optimizer_type == EmbOptimType.SGD
                    or self._optimizer_type == EmbOptimType.EXACT_SGD
                ):
                    option.optimizer_type = OptimizerType.SGD
                elif self._optimizer_type == EmbOptimType.ADAM:
                    option.optimizer_type = OptimizerType.Adam
                elif self._optimizer_type == EmbOptimType.EXACT_ADAGRAD:
                    option.optimizer_type = OptimizerType.AdaGrad
                else:
                    raise ValueError(
                        f"Not supported optimizer type ,optimizer type = {self._optimizer_type} {type(self._optimizer_type)} {self._optimizer_type.value}."
                    )
            self._tables.append(create_dynamicemb_table(option))

    def _create_optimizer(
        self,
        optimizer_type: EmbOptimType,
        optimizer_args: OptimizerArgs,
    ) -> None:
        if optimizer_type == EmbOptimType.SGD:
            self._optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                self._tables,
            )
        elif optimizer_type == EmbOptimType.EXACT_SGD:
            self._optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                self._tables,
            )
        elif optimizer_type == EmbOptimType.ADAM:
            self._optimizer = AdamDynamicEmbeddingOptimizer(
                optimizer_args,
                self._dynamicemb_options,
                self._tables,
            )
        elif optimizer_type == EmbOptimType.EXACT_ADAGRAD:
            self._optimizer = AdaGradDynamicEmbeddingOptimizer(
                optimizer_args, self._dynamicemb_options, self._tables
            )
        elif optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            self._optimizer = RowWiseAdaGradDynamicEmbeddingOptimizer(
                optimizer_args, self._dynamicemb_options, self._tables
            )
        else:
            raise ValueError(
                f"Not supported optimizer type ,optimizer type = {optimizer_type} {type(optimizer_type)} {optimizer_type.value}."
            )

    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, _ in enumerate(self._dynamicemb_options):
            splits.append(
                torch.empty(
                    (1, 1), device=torch.device("cuda"), dtype=self.embedding_dtype
                )
            )
        return splits

    def flush(self) -> None:
        return

    def reset_cache_states(self) -> None:
        return

    @property
    def table_names(self) -> List[str]:
        return self._table_names

    @property
    def optimizer(self) -> BaseDynamicEmbeddingOptimizer:
        return self._optimizer

    @property
    def tables(self) -> List[DynamicEmbTable]:
        return self._tables

    def set_learning_rate(self, lr: float) -> None:
        self._optimizer.set_learning_rate(lr)
        return

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        per_sample_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
        # 2D tensor of batch size for each rank and feature.
        # Shape (number of features, number of ranks)
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        total_unique_indices: Optional[int] = None,
    ) -> List[Tensor]:
        if indices.dtype != self.index_type:
            indices = indices.to(self.index_type)

        # offsets is on device, if we want to split the indices, we have to read the offset firstly.
        # Jost forward it to DynamicEmbeddingFunction
        # return DynamicEmbeddingFunction.apply(indices, offsets, self.table_offsets_in_feature, self.tables, self.total_D,
        #                                       self.dims,self.feature_table_map, self.embedding_dtype, self.pooling_mode, torch.device(self.device_id), 1, self._empty_tensor)

        scores = []
        for table_name in self._table_names:
            if table_name not in self._scores.keys():
                raise RuntimeError(
                    f"Must set score for table '{table_name}' whose score_strategy is customized."
                )
            scores.append(self._scores[table_name])

        if self.pooling_mode == DynamicEmbPoolingMode.NONE:
            res = DynamicEmbeddingFunction.apply(
                indices,
                offsets,
                self.use_index_dedup,
                self.table_offsets_in_feature,
                self._tables,
                scores,
                self.total_D,
                self.dims[0],
                self.feature_table_map,
                self.embedding_dtype,
                self.output_dtype,
                self.pooling_mode,
                self._device_num_sms,
                self._unique_op,
                torch.device(self.device_id),
                self._optimizer,
                self.frequency_threshold,
                self.mask_dims,
                self._empty_tensor,
            )
        else:
            res = DynamicEmbeddingBagFunction.apply(
                indices,
                offsets,
                self.use_index_dedup,
                self.table_offsets_in_feature,
                self._tables,
                scores,
                self.total_D,
                self.dims,
                self.feature_table_map,
                self.embedding_dtype,
                self.output_dtype,
                self.pooling_mode,
                self._device_num_sms,
                self._unique_op,
                torch.device(self.device_id),
                self._optimizer,
                self.frequency_threshold,
                self.mask_dims,
                self._empty_tensor,
            )

        self._update_score()
        return res

    def set_score(
        self,
        named_score: Dict[str, int],
    ) -> None:
        table_names: List[str] = named_score.keys()
        table_scores: List[int] = named_score.values()
        for table_name, table_score in zip(table_names, table_scores):
            if not isinstance(table_score, int):
                raise ValueError(
                    f"Table's score is expect to int but got {type(table_score)}"
                )
            if table_score == 0:
                raise ValueError(f"Can't set table's score to 0.")
            index = self._table_names.index(table_name)
            assert (
                self._dynamicemb_options[index].score_strategy
                == DynamicEmbScoreStrategy.CUSTOMIZED
            ), "Can only set score for table whose score_strategy is DynamicEmbScoreStrategy.CUSTOMIZED."

            if table_name in self._scores and self._scores[table_name] > table_score:
                if warning_for_cstm_score():
                    warnings.warn(
                        f"New set score is less than the old one for table '{table_name}': {table_score} < {self._scores[table_name]}",
                        UserWarning,
                    )
            self._scores[table_name] = table_score

    def get_score(self) -> Dict[str, int]:
        return self._scores.copy()

    def _create_score(self):
        self._scores: Dict[str, int] = {}
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                option.evict_strategy = DynamicEmbEvictStrategy.LRU
                self._scores[table_name] = device_timestamp()
            elif option.score_strategy == DynamicEmbScoreStrategy.STEP:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
                self._scores[table_name] = 1
            elif option.score_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
            elif option.score_strategy == DynamicEmbScoreStrategy.LFU:
                option.evict_strategy = DynamicEmbEvictStrategy.LFU
                self._scores[table_name] = 1

    def _update_score(self):
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            old_score = self._scores[table_name]
            if option.score_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                new_score = device_timestamp()
                if new_score < old_score:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is less than old one({old_score}).",
                        UserWarning,
                    )
                self._scores[table_name] = new_score
            elif option.score_strategy == DynamicEmbScoreStrategy.STEP:
                max_uint64 = (2**64) - 1
                new_score = old_score + 1
                if new_score > max_uint64:
                    warnings.warn(
                        f"Table '{table_name}' 's score({new_score}) is out of range, reset to 0.",
                        UserWarning,
                    )
                    self._scores[table_name] = 0
                else:
                    self._scores[table_name] = new_score
            elif option.score_strategy == DynamicEmbScoreStrategy.LFU:
                self._scores[table_name] = 1

    def incremental_dump(
        self,
        named_thresholds: Dict[str, int] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> Tuple[Dict[str, Tuple[Tensor, Tensor]], Dict[str, int]]:
        table_names: List[str] = named_thresholds.keys()
        table_thresholds: List[int] = named_thresholds.values()
        ret_tensors: Dict[str, Tuple[Tensor, Tensor]] = {}
        ret_scores: Dict[str, int] = {}
        for table_name, threshold in zip(table_names, table_thresholds):
            index = self._table_names.index(table_name)
            if not dist.is_initialized() or dist.get_world_size(group=pg) == 1:
                key, value = _export_matched(self._tables[index], threshold)
            else:
                key, value = _export_matched_and_gather(
                    self._tables[index], threshold, pg
                )
            ret_tensors[table_name] = (key, value)
            ret_scores[table_name] = self._scores[table_name]
        return ret_tensors, ret_scores
