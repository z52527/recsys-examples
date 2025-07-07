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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import itertools
import logging
from collections import OrderedDict, defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
from torch import distributed as dist
from torchrec.distributed.types import LazyAwaitable

if not torch._running_with_deploy():
    from torch.distributed._composable.fsdp.fully_shard import FSDPModule as FSDP2
else:

    class FSDP2:
        pass


from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.transformer.module import Float16Module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.fx.immutable_collections import immutable_dict as fx_immutable_dict
from torch.fx.immutable_collections import immutable_list as fx_immutable_list
from torch.fx.node import Node
from torch.nn.modules.module import _IncompatibleKeys
from torch.profiler import record_function
from torchrec.distributed.dist_data import KJTAllToAll, KJTAllToAllTensorsAwaitable
from torchrec.distributed.embedding_sharding import (
    FusedKJTListSplitsAwaitable,
    KJTListSplitsAwaitable,
    KJTSplitsAllToAllMeta,
)
from torchrec.distributed.embedding_types import KJTList
from torchrec.distributed.model_parallel import DistributedModelParallel, ShardedModule
from torchrec.distributed.types import Awaitable, LazyNoWait
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.streamable import Multistreamable, Pipelineable

logger: logging.Logger = logging.getLogger(__name__)


In = TypeVar("In", bound=Pipelineable)
StageOut = TypeVar("StageOut", bound=Pipelineable)
Out = TypeVar("Out")

RunnableType = Callable[..., StageOut]
StageOutputWithEvent = Tuple[Optional[StageOut], Optional[torch.Event]]


@dataclass
class TrainPipelineContext:
    """
    Context information for a `TrainPipelineSparseDist` instance.

    Attributes:
        input_dist_splits_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the splits awaitable stage, which occurs after starting the
            input dist.
        input_dist_tensors_requests (Dict[str, Awaitable[Any]]): Stores input dist
            requests in the tensors awaitable stage, which occurs after calling `wait()`
            on the splits awaitable.
        module_contexts (Dict[str, Multistreamable]): Stores module contexts from the
            input dist for the current batch.
        module_contexts_next_batch (Dict[str, Multistreamable]): Stores module contexts
            from the input dist for the next batch. (only for version 0)
        fused_splits_awaitables (List[Tuple[List[str], FusedKJTListSplitsAwaitable]]):
            List of fused splits input dist awaitable and the corresponding module names
            of each awaitable.
        event: Optional[torch.cuda.Event]: Event to record the completion of this stage
        index: Optional[int]: Index of the current batch.
        version: int = 0; support for backward compatibility
    """

    # pyre-ignore [4]
    input_dist_splits_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    # pyre-ignore [4]
    input_dist_tensors_requests: Dict[str, Awaitable[Any]] = field(default_factory=dict)
    module_contexts: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )  # deprecated: to support legacy code
    fused_splits_awaitables: List[
        Tuple[List[str], FusedKJTListSplitsAwaitable]
    ] = field(default_factory=list)
    events: List[torch.Event] = field(default_factory=list)
    postproc_fwd_results: Dict[str, Any] = field(default_factory=dict)
    index: Optional[int] = None
    version: int = 0  # 1 is current version, 0 is deprecated but supported for backward compatibility


@dataclass
class PrefetchTrainPipelineContext(TrainPipelineContext):
    module_input_post_prefetch: Dict[str, Multistreamable] = field(default_factory=dict)
    module_contexts_post_prefetch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )
    module_input_post_prefetch_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )
    module_contexts_post_prefetch_next_batch: Dict[str, Multistreamable] = field(
        default_factory=dict
    )


@dataclass
class EmbeddingTrainPipelineContext(TrainPipelineContext):
    embedding_a2a_requests: Dict[
        str,
        Union[
            LazyAwaitable[Multistreamable],
            # ManagedCollisionEC/EBC returns tuple of awaitables
            Tuple[
                LazyAwaitable[KeyedTensor], LazyAwaitable[Optional[KeyedJaggedTensor]]
            ],
        ],
    ] = field(default_factory=dict)
    embedding_tensors: List[List[torch.Tensor]] = field(default_factory=list)
    embedding_features: List[List[Union[str, List[str]]]] = field(default_factory=list)
    detached_embedding_tensors: List[List[torch.Tensor]] = field(default_factory=list)


@dataclass
class ArgInfo:
    """
    Representation of args from a node.

    Attributes:
        input_attrs (List[str]): attributes of input batch,
            e.g. `batch.attr1.attr2` will produce ["attr1", "attr2"].
        is_getitems (List[bool]): `batch[attr1].attr2` will produce [True, False].
        postproc_modules (List[Optional[PipelinedPostproc]]): list of torch.nn.Modules that
            transform the input batch.
        constants: constant arguments that are passed to postproc modules.
        name (Optional[str]): name for kwarg of pipelined forward() call or None for a
            positional arg.
    """

    input_attrs: List[str]
    is_getitems: List[bool]
    # recursive dataclass as postproc_modules.args -> arginfo.postproc_modules -> so on
    postproc_modules: List[Optional["PipelinedPostproc"]]
    constants: List[Optional[object]]
    name: Optional[str]


# pyre-ignore
def _build_args_kwargs(
    # pyre-ignore
    initial_input: Any,
    fwd_args: List[ArgInfo],
) -> Tuple[List[Any], Dict[str, Any]]:
    args = []
    kwargs = {}
    for arg_info in fwd_args:
        if arg_info.input_attrs:
            arg = initial_input
            for attr, is_getitem, postproc_mod, obj in zip(
                arg_info.input_attrs,
                arg_info.is_getitems,
                arg_info.postproc_modules,
                arg_info.constants,
            ):
                if obj is not None:
                    if isinstance(obj, list):
                        arg = [
                            (
                                v
                                if not isinstance(v, ArgInfo)
                                else _build_args_kwargs(initial_input, [v])[0][0]
                            )
                            for v in obj
                        ]
                    elif isinstance(obj, dict):
                        arg = {
                            k: (
                                v
                                if not isinstance(v, ArgInfo)
                                else _build_args_kwargs(initial_input, [v])[0][0]
                            )
                            for k, v in obj.items()
                        }
                    else:
                        arg = obj
                    break
                elif postproc_mod is not None:
                    # postproc will internally run the same logic recursively
                    # if its args are derived from other postproc modules
                    # we can get all inputs to postproc mod based on its recorded args_info + arg passed to it
                    arg = postproc_mod(arg)
                else:
                    if is_getitem:
                        arg = arg[attr]
                    elif attr != "":
                        arg = getattr(arg, attr)
                    else:
                        # neither is_getitem nor valid attr, no-op
                        arg = arg
            if arg_info.name:
                kwargs[arg_info.name] = arg
            else:
                args.append(arg)
        else:
            if arg_info.name:
                kwargs[arg_info.name] = None
            else:
                args.append(None)
    return args, kwargs


def recursive_record_stream(
    # pyre-fixme[2]: Parameter `re` must have a type that does not contain `Any`
    res: Union[torch.Tensor, Pipelineable, Iterable[Any], Dict[Any, Any]],
    stream: torch.Stream,
) -> None:
    if isinstance(res, torch.Tensor) and res.device.type in ["cuda", "mtia"]:
        res.record_stream(stream)
    elif isinstance(res, Pipelineable):
        res.record_stream(stream)
    elif isinstance(res, (list, tuple)):
        for v in res:
            recursive_record_stream(v, stream)
    elif isinstance(res, dict):
        for v in res.values():
            recursive_record_stream(v, stream)


class NoOpStream:
    """No-Op Context manager that takes in a stream"""

    def __init__(self, stream: Optional[torch.Stream]) -> None:
        self._stream = stream

    def __enter__(self) -> "NoOpStream":
        """Return `self` upon entering the runtime context."""
        return self

    # pyre-ignore
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class PipelinedPostproc(torch.nn.Module):
    """
    Wrapper around postproc module found during model graph traversal for sparse data dist
    pipelining. In addition to the original module, it encapsulates information needed for
    execution such as list of ArgInfo and the current training pipeline context.

    Args:
        postproc_module (torch.nn.Module): postproc module to run
        fqn (str): fqn of the postproc module in the model being pipelined
        args (List[ArgInfo]): list of ArgInfo for the postproc module
        context (TrainPipelineContext): Training context for the next iteration / batch

    Returns:
        Any

    Example:
        postproc = PipelinedPostproc(postproc_module, fqn, args, context)
        # module-swap with pipeliend postproc
        setattr(model, fqn, postproc)
    """

    _FORCE_STATE_DICT_LOAD = True

    def __init__(
        self,
        postproc_module: torch.nn.Module,
        fqn: str,
        args: List[ArgInfo],
        context: TrainPipelineContext,
        # TODO: make streams non-optional - skipping now to avoid ripple effect
        default_stream: Optional[torch.Stream],
        dist_stream: Optional[torch.Stream],
    ) -> None:
        super().__init__()
        self._postproc_module = postproc_module
        self._fqn = fqn
        self._args = args
        self._context = context
        self._default_stream = default_stream
        self._dist_stream = dist_stream
        if not default_stream:
            logger.warning(
                f"Postproc module {fqn} has no default stream. This may cause race conditions and NaNs during training!"
            )
        if not dist_stream:
            logger.warning(
                f"Postproc module {fqn} has no dist stream. This may cause race conditions and NaNs during training!"
            )

        if self._dist_stream:
            device: torch.device = self._dist_stream.device
            # pyre-ignore
            self._stream_context = (
                torch.get_device_module(device).stream
                if device.type in ["cuda", "mtia"]
                else torch.cuda.stream
            )
        else:
            self._stream_context = NoOpStream

    @property
    def postproc_module(self) -> torch.nn.Module:
        return self._postproc_module

    @property
    def fqn(self) -> str:
        return self._fqn

    # pyre-ignore
    def forward(self, *input, **kwargs) -> Any:
        """
        Args:
            Any args and kwargs during model fwd
            During _start_data_dist, input[0] contains the current data
        Returns:
            Any
        """
        if self._fqn in self._context.postproc_fwd_results:
            # This should only be hit in two cases:
            # 1) During model forward
            # During model forward, avoid duplicate work
            # by returning the cached result from previous
            # iteration's _start_data_dist
            # 2) During _start_data_dist when postproc module is
            # shared by more than one args. e.g. if we have
            # postproc_out_a = postproc_a(input)
            # postproc_out_b = postproc_b(postproc_out_a) <- postproc_a shared
            # postproc_out_c = postproc_c(postproc_out_a) <-^
            # When processing postproc_b, we cache value of postproc_a(input)
            # so when processing postproc_c, we can reuse postproc_a(input)
            res = self._context.postproc_fwd_results[self._fqn]
            return res

        # Everything below should only be called during _start_data_dist stage

        # Build up arg and kwargs from recursive call to pass to postproc module
        # Arguments to postproc module can be also be a derived product
        # of another postproc module call, as long as module is pipelineable

        # Use input[0] as _start_data_dist only passes 1 arg
        args, kwargs = _build_args_kwargs(input[0], self._args)
        with record_function(f"## sdd_input_postproc {self._context.index} ##"):
            # should be no-op as we call this in dist stream
            with self._stream_context(self._dist_stream):
                res = self._postproc_module(*args, **kwargs)

            # Ensure postproc modules output is safe to use from default stream later
            if self._default_stream and self._dist_stream:
                self._default_stream.wait_stream(self._dist_stream)

                if isinstance(res, (torch.Tensor, Pipelineable, Iterable, Dict)):
                    # Result from module forward might be a complex type such as
                    # Tuple[KeyedJaggedTensor, Dict[str, torch.Tensor]]
                    # In this case, we need to first iterate over each element of tuple
                    # and call record_stream on first item as KJT is Pipelineable
                    # for the second item (Dict), we iterate over the values and call
                    # record_stream accordingly.

                    # pyre-ignore[6]
                    recursive_record_stream(res, self._default_stream)
                elif self._context.index == 0:
                    logger.warning(
                        f"Result of postproc module {self._fqn} is of type {type(res)}. We currently expect it to be a Tensor, Pipelineable, Iterable, or Dict to handle memory safety. If your output is not of this type, please add support for it above. Otherwise you might run into NaNs or CUDA Illegal Memory issues during training!"
                    )

            with self._stream_context(self._default_stream):
                # Cache results, only during _start_data_dist
                self._context.postproc_fwd_results[self._fqn] = res

            return res

    @property
    def args(self) -> List[ArgInfo]:
        return self._args

    def set_context(self, context: TrainPipelineContext) -> None:
        self._context = context

    def get_context(self) -> TrainPipelineContext:
        return self._context

    def named_modules(
        self,
        memo: Optional[Set[torch.nn.Module]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ) -> Iterator[Tuple[str, torch.nn.Module]]:
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            # This is needed because otherwise the rewrite won't find the existing postproc, and will create a new one
            # Also, `named_modules` need to include self - see base implementation in the nn.modules.Module
            yield prefix, self
            # Difference from base implementation is here - the child name (_postproc_module) is not added to the prefix
            yield from self._postproc_module.named_modules(
                memo, prefix, remove_duplicate
            )

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.nn.Parameter]]:
        yield from self._postproc_module.named_parameters(
            prefix,
            recurse,
            remove_duplicate,
        )

    def named_buffers(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        yield from self._postproc_module.named_buffers(
            prefix, recurse, remove_duplicate
        )

    # pyre-ignore [14]
    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        # super().state_dict(destination, prefix, keep_vars)
        if destination is None:
            destination = OrderedDict()
            # pyre-ignore [16]
            destination._metadata = OrderedDict()
        self._postproc_module.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        return destination

    # pyre-ignore [14]
    def load_state_dict(
        self,
        state_dict: OrderedDict[str, torch.Tensor],
        strict: bool = True,
    ) -> _IncompatibleKeys:
        return self._postproc_module.load_state_dict(state_dict, strict=strict)


TForwardContext = TypeVar("TForwardContext", bound=TrainPipelineContext)

EmbeddingModuleRetType = Union[Dict[str, JaggedTensor], KeyedTensor]


class BaseForward(Generic[TForwardContext]):
    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule,
        context: TForwardContext,
        stream: Optional[torch.Stream] = None,
    ) -> None:
        self._name = name
        self._args = args
        self._module = module
        self._context = context
        self._stream = stream
        self._device: torch.device = stream.device if stream else torch.device("cuda")

    @property
    def name(self) -> str:
        return self._name

    @property
    def args(self) -> List[ArgInfo]:
        return self._args

    def set_context(self, context: TForwardContext) -> None:
        self._context = context

    def get_context(self) -> TForwardContext:
        return self._context


class PipelinedForward(BaseForward[TrainPipelineContext]):
    """
    This pipeline is used in TrainPipelineSparseDist
    """

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert (
            self._name in self._context.input_dist_tensors_requests
        ), "Invalid PipelinedForward usage, please do not directly call model.forward()"
        request = self._context.input_dist_tensors_requests.pop(self._name)
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with torch.get_device_module(self._device).stream(self._stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        ctx = self._context.module_contexts.pop(self._name)

        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)
            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(ctx, data)


class EmbeddingPipelinedForward(BaseForward[EmbeddingTrainPipelineContext]):
    """
    This pipeline is used in TrainPipelineSemiSync
    """

    def __call__(
        self,
        # pyre-ignore
        *input,
        # pyre-ignore
        **kwargs,
    ) -> Union[
        Awaitable[EmbeddingModuleRetType],
        Tuple[
            Awaitable[EmbeddingModuleRetType], Awaitable[Optional[KeyedJaggedTensor]]
        ],
    ]:
        assert (
            self._name in self._context.embedding_a2a_requests
        ), "Invalid EmbeddingPipelinedForward usage, please do not directly call model.forward()"

        ctx = self._context.module_contexts.pop(self._name)
        cur_stream = torch.get_device_module(self._device).current_stream()

        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            ctx.record_stream(cur_stream)
        awaitable = self._context.embedding_a2a_requests.pop(self._name)
        # in case of MC modules
        is_mc_module: bool = isinstance(awaitable, Iterable)
        remapped_kjts: Optional[KeyedJaggedTensor] = None

        if is_mc_module:
            embeddings = awaitable[0].wait()
            remapped_kjts = awaitable[1].wait()
        else:
            assert isinstance(awaitable, Awaitable)
            embeddings = (
                awaitable.wait()
            )  # trigger awaitable manually for type checking
        tensors = []
        detached_tensors = []
        # in case of EC, embeddings are Dict[str, JaggedTensor]
        if isinstance(embeddings, Dict):
            for jt in embeddings.values():
                assert isinstance(jt, JaggedTensor)
                tensor = jt.values()
                detached_tensor = tensor.detach().requires_grad_()
                detached_tensor.retain_grad()
                jt._values = detached_tensor
                tensors.append(tensor)
                detached_tensors.append(detached_tensor)
            self._context.embedding_tensors.append(tensors)
            self._context.embedding_features.append(list(embeddings.keys()))
            self._context.detached_embedding_tensors.append(detached_tensors)
        else:
            # in case of EBC, embeddings are KeyedTensor
            assert isinstance(embeddings, KeyedTensor)
            embeddings.record_stream(cur_stream)
            tensor = embeddings.values()
            detached_tensor = tensor.detach().requires_grad_()
            detached_tensor.retain_grad()
            embeddings._values = detached_tensor
            tensors.append(tensor)
            detached_tensors.append(detached_tensor)
            self._context.embedding_tensors.append(tensors)
            """
            KeyedTensor is returned by EmbeddingBagCollections and its variants
            KeyedTensor holds dense data from multiple features and .values()
            returns a single concatenated dense tensor. To ensure that
            context.embedding_tensors[i] has the same length as
            context.embedding_features[i], we pass in a list with a single item:
            a list containing all the embedding feature names.
            """
            self._context.embedding_features.append([list(embeddings.keys())])
            self._context.detached_embedding_tensors.append(detached_tensors)

        if is_mc_module:
            return (LazyNoWait(embeddings), LazyNoWait(remapped_kjts))
        else:
            return LazyNoWait(embeddings)


class PrefetchPipelinedForward(BaseForward[PrefetchTrainPipelineContext]):
    """
    This pipeline is used in PrefetchTrainPipelineSparseDist
    """

    def __init__(
        self,
        name: str,
        args: List[ArgInfo],
        module: ShardedModule,
        context: PrefetchTrainPipelineContext,
        prefetch_stream: Optional[torch.Stream] = None,
    ) -> None:
        super().__init__(
            name=name,
            args=args,
            module=module,
            context=context,
            stream=prefetch_stream,
        )

    # pyre-ignore [2, 24]
    def __call__(self, *input, **kwargs) -> Awaitable:
        assert (
            self._name in self._context.module_input_post_prefetch
        ), "Invalid PrefetchPipelinedForward usage, please do not directly call model.forward()"
        data = self._context.module_input_post_prefetch.pop(self._name)
        ctx = self._context.module_contexts_post_prefetch.pop(self._name)

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        if self._stream is not None:
            torch.get_device_module(self._device).current_stream().wait_stream(
                self._stream
            )
            cur_stream = torch.get_device_module(self._device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)

            ctx.record_stream(cur_stream)

        return self._module.compute_and_output_dist(ctx, data)


class KJTAllToAllForward:
    def __init__(
        self, pg: dist.ProcessGroup, splits: List[int], stagger: int = 1
    ) -> None:
        self._pg = pg
        self._splits = splits
        self._stagger = stagger
        self._splits_cumsum: List[int] = [0] + list(itertools.accumulate(splits))

    def __call__(self, input: KeyedJaggedTensor) -> KJTSplitsAllToAllMeta:
        with torch.no_grad():
            assert len(input.keys()) == sum(self._splits)
            rank = dist.get_rank(self._pg)
            local_keys = input.keys()[
                self._splits_cumsum[rank] : self._splits_cumsum[rank + 1]
            ]
            input_splits = input.dist_splits(self._splits)
            device = input.values().device
            splits_tensors = [
                torch.tensor(splits, device=device) for splits in input_splits
            ]
            if not input.variable_stride_per_key():
                splits_tensors.append(
                    torch.tensor([input.stride()] * self._pg.size(), device=device)
                )
            return KJTSplitsAllToAllMeta(
                pg=self._pg,
                _input=input,
                splits=self._splits,
                splits_tensors=splits_tensors,
                input_splits=input_splits,
                input_tensors=input.dist_tensors(),
                labels=input.dist_labels(),
                keys=local_keys,
                device=device,
                stagger=self._stagger,
            )


class Tracer(torch.fx.Tracer):
    """
    Disables proxying buffers during tracing. Ideally, proxying buffers would be
    disabled, but some models are currently mutating buffer values, which causes errors
    during tracing. If those models can be rewritten to not do that, we can likely
    remove this line.
    """

    proxy_buffer_attributes = False

    def __init__(self, leaf_modules: Optional[List[str]] = None) -> None:
        super().__init__()
        self._leaf_modules: List[str] = leaf_modules if leaf_modules is not None else []

    def create_arg(self, a):
        if isinstance(a, torch.Tensor):
            return super().create_arg(a)
        elif isinstance(a, tuple):
            return tuple(self.create_arg(x) for x in a)
        elif isinstance(a, list):
            return [self.create_arg(x) for x in a]
        elif isinstance(a, dict):
            return {k: self.create_arg(v) for k, v in a.items()}
        else:
            return a

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if (
            isinstance(m, ShardedModule)
            or module_qualified_name in self._leaf_modules
            or isinstance(m, FSDP)
            or isinstance(m, FSDP2)
        ):
            return True
        return super().is_leaf_module(m, module_qualified_name)


def _to_device(batch: In, device: torch.device, non_blocking: bool) -> In:
    assert isinstance(
        batch, (torch.Tensor, Pipelineable)
    ), f"{type(batch)} must implement Pipelineable interface"
    return cast(In, batch.to(device=device, non_blocking=non_blocking))


def _wait_for_batch(batch: In, stream: Optional[torch.Stream]) -> None:
    """
    As mentioned in
    https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html, PyTorch
    uses the "caching allocator" for memory allocation for tensors. When a tensor is
    freed, its memory is likely to be reused by newly constructed tenosrs. By default,
    this allocator traces whether a tensor is still in use by only the CUDA stream where
    it was created. When a tensor is used by additional CUDA streams, we need to call
    `record_stream` to tell the allocator about these streams. Otherwise, the allocator
    might free the underlying memory of the tensor once it is no longer used by the
    creator stream. This is a notable programming trick when we write programs using
    multiple CUDA streams.
    """
    if stream is None:
        return

    device = stream.device
    torch.get_device_module(device).current_stream().wait_stream(stream)
    cur_stream = torch.get_device_module(device).current_stream()
    assert isinstance(
        batch, (torch.Tensor, Multistreamable)
    ), f"{type(batch)} must implement Multistreamable interface"
    batch.record_stream(cur_stream)


def _wait_for_events(
    batch: In,
    context: TrainPipelineContext,
    stream: Optional[torch.Stream],
) -> None:
    """
    Wait for any outstanding events for a given context
    """

    for event in context.events:
        event.wait()
    context.events.clear()
    if stream:
        assert isinstance(
            batch, (torch.Tensor, Multistreamable)
        ), f"{type(batch)} must implement Multistreamable interface"
        batch.record_stream(stream)


def _start_data_dist(
    pipelined_modules: List[ShardedModule],
    batch: Pipelineable,
    context: TrainPipelineContext,
) -> None:
    if context.version == 0:
        context.input_dist_splits_requests.clear()
        context.module_contexts_next_batch.clear()
        context.fused_splits_awaitables.clear()

    for module in pipelined_modules:
        forward = module.forward
        assert isinstance(
            forward,
            (
                PipelinedForward,
                PrefetchPipelinedForward,
                EmbeddingPipelinedForward,
            ),
        )

        # Retrieve argument for the input_dist of EBC
        # is_getitem True means this argument could be retrieved by a list
        # False means this argument is getting while getattr
        # and this info was done in the _rewrite_model by tracing the
        # entire model to get the arg_info_list
        args, kwargs = _build_args_kwargs(batch, forward.args)

        # Start input distribution.
        module_ctx = module.create_context()
        if context.version == 0:
            context.module_contexts_next_batch[forward.name] = module_ctx
        else:
            context.module_contexts[forward.name] = module_ctx
        context.input_dist_splits_requests[forward.name] = module.input_dist(
            module_ctx, *args, **kwargs
        )
    _fuse_input_dist_splits(context)


def _start_embedding_lookup(
    module: ShardedModule,
    context: EmbeddingTrainPipelineContext,
    source_stream: Optional[torch.Stream],
    target_stream: Optional[torch.Stream],
    # pyre-ignore[2]
    stream_context: Callable[..., AbstractContextManager[Any, Any]],
) -> None:
    module_context = context.module_contexts[module.forward.name]
    with stream_context(source_stream):
        kjt = context.input_dist_tensors_requests[module.forward.name].wait()

    if target_stream is not None:
        kjt.record_stream(target_stream)
        module_context.record_stream(target_stream)
    output_dist_out = module.compute_and_output_dist(module_context, kjt)
    context.embedding_a2a_requests[module.forward.name] = output_dist_out


def _fuse_input_dist_splits(context: TrainPipelineContext) -> None:
    names_per_pg = defaultdict(list)
    for name, request in context.input_dist_splits_requests.items():
        pg = None
        if isinstance(request, KJTListSplitsAwaitable):
            for awaitable in request.awaitables:
                if isinstance(awaitable, KJTSplitsAllToAllMeta):
                    pg = awaitable.pg
                    break
        names_per_pg[pg].append(name)

    for pg, names in names_per_pg.items():
        context.fused_splits_awaitables.append(
            (
                names,
                FusedKJTListSplitsAwaitable(
                    # pyre-ignore[6]
                    requests=[
                        context.input_dist_splits_requests[name] for name in names
                    ],
                    contexts=[
                        (
                            context.module_contexts_next_batch[name]
                            if context.version == 0
                            else context.module_contexts[name]
                        )
                        for name in names
                    ],
                    pg=pg,
                ),
            )
        )


def _check_args_for_call_module(
    node: torch.fx.Node,
) -> bool:
    """
    Recursively checks if args to a node is the result of a call_module.
    """
    if node.op == "call_module":
        return True

    for arg in node.args:
        if isinstance(arg, torch.fx.Node) and _check_args_for_call_module(arg):
            return True

    return False


def _check_postproc_pipelineable(
    module: torch.nn.Module,
) -> bool:
    for _, _ in module.named_parameters(recurse=True):
        # Cannot have any trainable params for it to be pipelined
        logger.warning(
            f"Module {module} cannot be pipelined as it has trainable parameters"
        )
        return False
    return True


def _find_postproc_module_recursive(
    module: torch.nn.Module,
    postproc_module_fqn: str,
) -> Optional[torch.nn.Module]:
    """
    Finds the postproc module in the model.
    """
    for name, child in module.named_modules():
        if name == postproc_module_fqn:
            return child
    return None


def _swap_postproc_module_recursive(
    module: torch.nn.Module,
    to_swap_module: torch.nn.Module,
    postproc_module_fqn: str,
    path: str = "",
) -> torch.nn.Module:
    """
    Swaps the postproc module in the model.
    """
    if isinstance(module, PipelinedPostproc):
        return module

    if path == postproc_module_fqn:
        return to_swap_module

    for name, child in module.named_children():
        child = _swap_postproc_module_recursive(
            child,
            to_swap_module,
            postproc_module_fqn,
            path + "." + name if path else name,
        )
        setattr(module, name, child)

    return module


def _get_node_args_helper_inner(
    model: torch.nn.Module,
    # pyre-ignore
    arg,
    arg_info: ArgInfo,
    num_found: int,
    pipelined_postprocs: Set[PipelinedPostproc],
    context: TrainPipelineContext,
    pipeline_postproc: bool,
    for_postproc_module: bool = False,
    default_stream: Optional[torch.Stream] = None,
    dist_stream: Optional[torch.Stream] = None,
) -> int:
    num_found = 0
    while True:
        if isinstance(arg, torch.fx.Proxy):
            arg = arg.node
        if not isinstance(arg, torch.fx.Node):
            if pipeline_postproc:
                arg_info.input_attrs.insert(0, "")
                arg_info.is_getitems.insert(0, False)
                arg_info.postproc_modules.insert(0, None)

                if isinstance(arg, fx_immutable_dict):
                    fx_nested_dict = {}

                    for k, v in arg.items():
                        if isinstance(v, torch.fx.Node):
                            arg_info_nested = ArgInfo([], [], [], [], None)
                            _get_node_args_helper_inner(
                                model,
                                v,
                                arg_info_nested,
                                num_found,
                                pipelined_postprocs,
                                context,
                                pipeline_postproc,
                                for_postproc_module,
                                default_stream=default_stream,
                                dist_stream=dist_stream,
                            )
                            fx_nested_dict[k] = arg_info_nested
                        else:
                            fx_nested_dict[k] = v

                    arg_info.constants.insert(0, fx_nested_dict)
                elif isinstance(arg, fx_immutable_list):
                    fx_nested_list = []
                    for v in arg:
                        if isinstance(v, torch.fx.Node):
                            arg_info_nested = ArgInfo([], [], [], [], None)
                            _get_node_args_helper_inner(
                                model,
                                v,
                                arg_info_nested,
                                num_found,
                                pipelined_postprocs,
                                context,
                                pipeline_postproc,
                                for_postproc_module,
                                default_stream=default_stream,
                                dist_stream=dist_stream,
                            )
                            fx_nested_list.append(arg_info_nested)
                        else:
                            fx_nested_list.append(v)

                    arg_info.constants.insert(0, fx_nested_list)
                else:
                    arg_info.constants.insert(0, arg)
                num_found += 1
            break
        child_node = arg
        if child_node.op == "placeholder":
            if hasattr(child_node, "ph_key"):
                # pyre-ignore[16]
                ph_key: str = child_node.ph_key
                # example: ph_key = 'event_id_list_features_seqs[marketplace]'
                ph_key = ph_key.replace("[", ".")
                ph_keys = ph_key.split(".")
                for key in ph_keys:
                    if "]" in key:
                        arg_info.input_attrs.append(key[:-1])
                        arg_info.is_getitems.append(True)
                    else:
                        arg_info.input_attrs.append(key)
                        arg_info.is_getitems.append(False)
                    arg_info.postproc_modules.append(None)
                    arg_info.constants.append(None)
            else:
                # no-op
                arg_info.input_attrs.insert(0, "")
                arg_info.is_getitems.insert(0, False)
                arg_info.postproc_modules.insert(0, None)
                arg_info.constants.insert(0, None)

            num_found += 1
            break
        elif (
            child_node.op == "call_function"
            and child_node.target.__module__ == "builtins"
            # pyre-ignore[16]
            and child_node.target.__name__ == "getattr"
        ):
            # pyre-fixme[6]: For 2nd argument expected `str` but got
            #  `Union[None, Dict[str, typing.Any], List[typing.Any], Node, bool,
            #  complex, float, int, range, slice, str, device, dtype, layout,
            #  memory_format, Tensor, typing.Tuple[typing.Any, ...]]`.
            arg_info.input_attrs.insert(0, child_node.args[1])
            arg_info.is_getitems.insert(0, False)
            arg_info.postproc_modules.insert(0, None)
            arg_info.constants.insert(0, None)
            arg = child_node.args[0]
        elif (
            child_node.op == "call_function"
            and child_node.target.__module__ == "_operator"
            # pyre-ignore[16]
            and child_node.target.__name__ == "getitem"
        ):
            # pyre-fixme[6]: For 2nd argument expected `str` but got
            #  `Union[None, Dict[str, typing.Any], List[typing.Any], Node, bool,
            #  complex, float, int, range, slice, str, device, dtype, layout,
            #  memory_format, Tensor, typing.Tuple[typing.Any, ...]]`.
            arg_info.input_attrs.insert(0, child_node.args[1])
            arg_info.is_getitems.insert(0, True)
            arg_info.postproc_modules.insert(0, None)
            arg_info.constants.insert(0, None)
            arg = child_node.args[0]
        elif (
            child_node.op == "call_function"
            and child_node.target.__module__ == "torch.utils._pytree"
            # pyre-ignore[16]
            and child_node.target.__name__ == "tree_unflatten"
        ):
            """
            This is for the PT2 export path where we unflatten the input to reconstruct
            the structure with the recorded tree spec.
            """
            assert arg_info.is_getitems[0]
            # pyre-fixme[16]
            arg = child_node.args[0][arg_info.input_attrs[0]]
        elif (
            child_node.op == "call_function"
            and child_node.target.__module__ == "torchrec.sparse.jagged_tensor"
            # pyre-fixme[16]
            and child_node.target.__name__ == "KeyedJaggedTensor"
        ):
            call_module_found = False

            for arg_node in chain(child_node.args, child_node.kwargs.values()):
                if isinstance(arg_node, torch.fx.Node) and _check_args_for_call_module(
                    arg_node
                ):
                    call_module_found = True
                    break

            if call_module_found:
                break

            if "values" in child_node.kwargs:
                arg = child_node.kwargs["values"]
            else:
                arg = child_node.args[1]
        elif child_node.op == "call_method" and child_node.target == "get":
            # pyre-ignore[6]
            arg_info.input_attrs.insert(0, child_node.args[1])
            arg_info.is_getitems.insert(0, True)
            arg_info.postproc_modules.insert(0, None)
            arg_info.constants.insert(0, None)
            arg = child_node.args[0]
        elif child_node.op == "call_module":
            postproc_module_fqn = str(child_node.target)
            postproc_module = _find_postproc_module_recursive(
                model, postproc_module_fqn
            )

            if not pipeline_postproc:
                logger.warning(
                    f"Found module {postproc_module} that potentially modifies KJ. Train pipeline initialized with `pipeline_postproc=False` (default), so we assume KJT input modification. To allow torchrec to check if this module can be safely pipelined, please set `pipeline_postproc=True`"
                )
                break

            if not postproc_module:
                # Could not find such module, should not happen
                break

            if isinstance(postproc_module, PipelinedPostproc):
                # Already did module swap and registered args, early exit
                arg_info.input_attrs.insert(0, "")  # dummy value
                arg_info.is_getitems.insert(0, False)
                pipelined_postprocs.add(postproc_module)
                arg_info.postproc_modules.insert(0, postproc_module)
                arg_info.constants.insert(0, None)
                num_found += 1
                break

            if not isinstance(postproc_module, torch.nn.Module):
                logger.warning(
                    f"Expected postproc_module to be nn.Module but was {type(postproc_module)}"
                )
                break

            # check if module is safe to pipeline i.e.no trainable param
            if not _check_postproc_pipelineable(postproc_module):
                break

            # For module calls, `self` isn't counted
            total_num_args = len(child_node.args) + len(child_node.kwargs)
            if total_num_args == 0:
                # module call without any args, assume KJT modified
                break

            # recursive call to check that all inputs to this postproc module
            # is either made of postproc module or non-modifying train batch input
            # transformations
            postproc_args, num_found_safe_postproc_args = _get_node_args(
                model,
                child_node,
                pipelined_postprocs,
                context,
                pipeline_postproc,
                True,
                default_stream=default_stream,
                dist_stream=dist_stream,
            )
            if num_found_safe_postproc_args == total_num_args:
                logger.info(
                    f"""Module {postproc_module} is a valid postproc module (no
                    trainable params and inputs can be derived from train batch input
                        via a series of either valid postproc modules or non-modifying
                        transformations) and will be applied during sparse data dist
                        stage"""
                )

                pipelined_postproc_module = PipelinedPostproc(
                    postproc_module,
                    postproc_module_fqn,
                    postproc_args,
                    context,
                    default_stream=default_stream,
                    dist_stream=dist_stream,
                )

                # module swap
                _swap_postproc_module_recursive(
                    model, pipelined_postproc_module, postproc_module_fqn
                )

                arg_info.input_attrs.insert(0, "")  # dummy value
                arg_info.is_getitems.insert(0, False)
                pipelined_postprocs.add(pipelined_postproc_module)
                arg_info.postproc_modules.insert(0, pipelined_postproc_module)
                arg_info.constants.insert(0, None)

                num_found += 1

            # we cannot set any other `arg` value here
            # break to avoid infinite loop
            break
        else:
            break

    return num_found


def _get_node_args_helper(
    model: torch.nn.Module,
    # pyre-ignore
    arguments,
    num_found: int,
    pipelined_postprocs: Set[PipelinedPostproc],
    context: TrainPipelineContext,
    pipeline_postproc: bool,
    # Add `None` constants to arg info only for postproc modules
    # Defaults to False for backward compatibility
    for_postproc_module: bool = False,
    default_stream: Optional[torch.Stream] = None,
    dist_stream: Optional[torch.Stream] = None,
) -> Tuple[List[ArgInfo], int]:
    """
    Goes through the args/kwargs of a node and arranges them into a list of `ArgInfo`s.
    It also counts the number of (args + kwargs) found.
    """
    arg_info_list = [ArgInfo([], [], [], [], None) for _ in range(len(arguments))]
    for arg, arg_info in zip(arguments, arg_info_list):
        if not for_postproc_module and arg is None:
            num_found += 1
            continue
        num_found += _get_node_args_helper_inner(
            model,
            arg,
            arg_info,
            num_found,
            pipelined_postprocs,
            context,
            pipeline_postproc,
            for_postproc_module,
            default_stream=default_stream,
            dist_stream=dist_stream,
        )
    return arg_info_list, num_found


def _get_node_args(
    model: torch.nn.Module,
    node: Node,
    pipelined_postprocs: Set[PipelinedPostproc],
    context: TrainPipelineContext,
    pipeline_postproc: bool,
    for_postproc_module: bool = False,
    default_stream: Optional[torch.Stream] = None,
    dist_stream: Optional[torch.Stream] = None,
) -> Tuple[List[ArgInfo], int]:
    num_found = 0
    pos_arg_info_list, num_found = _get_node_args_helper(
        model,
        node.args,
        num_found,
        pipelined_postprocs,
        context,
        pipeline_postproc,
        for_postproc_module,
        default_stream=default_stream,
        dist_stream=dist_stream,
    )
    kwargs_arg_info_list, num_found = _get_node_args_helper(
        model,
        node.kwargs.values(),
        num_found,
        pipelined_postprocs,
        context,
        pipeline_postproc,
        for_postproc_module,
        default_stream=default_stream,
        dist_stream=dist_stream,
    )

    # Replace with proper names for kwargs
    for name, arg_info_list in zip(node.kwargs, kwargs_arg_info_list):
        arg_info_list.name = name

    arg_info_list = pos_arg_info_list + kwargs_arg_info_list

    return (arg_info_list, num_found)


def _get_leaf_module_names_helper(
    model: torch.nn.Module,
    path: str,
    leaf_module_names: Set[str],
) -> bool:
    sharded_children = set()
    for name, child in model.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _get_leaf_module_names_helper(
                child,
                curr_path + ".",
                leaf_module_names,
            )
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, child in model.named_children():
            if name in sharded_children:
                continue
            # assume module is leaf node unless annotated otherwise
            if not getattr(child, "_is_pytorch_fx_traceable", False):
                leaf_module_names.add(path + name)
    return len(sharded_children) > 0


def _get_leaf_module_names(model: torch.nn.Module) -> List[str]:
    """
    Returns a list of top level modules to be used as leaf modules for FX tracing.
    This is a shallow FX trace that only goes the minimum depth required to pipeline
    the model unless child modules are explicitly tagged as `_is_pytorch_fx_traceable`.
    """

    leaf_module_names: Set[str] = set()
    _get_leaf_module_names_helper(
        model,
        "",
        leaf_module_names,
    )
    return list(leaf_module_names)


def _jit_modules(module: torch.nn.Module, path: str, optional: bool = True) -> bool:
    sharded_children = set()
    for name, child in module.named_children():
        curr_path = path + name
        if isinstance(child, ShardedModule):
            sharded_children.add(name)
        else:
            child_sharded = _jit_modules(child, curr_path + ".", optional)
            if child_sharded:
                sharded_children.add(name)

    if len(sharded_children) > 0:
        for name, child in module.named_children():
            if name not in sharded_children:
                try:
                    jit_child = torch.jit.script(child)
                    setattr(module, name, jit_child)
                    logger.info(f"jit.script applied to {path + name}.")
                except Exception as error:
                    if not optional:
                        raise
                    else:
                        logger.info(
                            f"Warning: failed to jit.script {path + name}: {error}."
                        )

    return len(sharded_children) > 0


def _pipeline_detach_model(
    model: torch.nn.Module,
    pipelined_modules: List[ShardedModule],
    # pyre-ignore[2]
    original_forwards: List[Callable[..., Any]],
    original_kjt_dist_forwards: List[
        Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]
    ],
    pipelined_postprocs: List[PipelinedPostproc],
) -> None:
    # Replace pipelined module forward and input dist forward with original forward
    kjt_dists = []
    for mod, original_fwd in zip(pipelined_modules, original_forwards):
        # pyre-ignore
        mod.forward = original_fwd

        for _, child_module in mod.named_modules():
            if not hasattr(child_module, "_input_dists"):
                continue
            for input_dist in child_module._input_dists:
                if hasattr(input_dist, "_dist"):
                    kjt_dists.append(input_dist._dist)
    assert len(kjt_dists) == len(
        original_kjt_dist_forwards
    ), f"Number of KJT dists ({len(kjt_dists)}) does not match number of kjt dist forwards provided ({len(original_kjt_dist_forwards)})"

    for kjt_dist, original_kjt_dist_fwd in zip(
        kjt_dists,
        original_kjt_dist_forwards,
    ):
        kjt_dist.forward = original_kjt_dist_fwd

    # Get underlying nn.Module
    if isinstance(model, DistributedModelParallel):
        model = model.module

    # Replace pipelined postproc modules with original postproc modules
    for postproc_mod in pipelined_postprocs:
        setattr(model, postproc_mod.fqn, postproc_mod.postproc_module)


# pyre-ignore[3]
def _rewrite_model(  # noqa C901
    model: torch.nn.Module,
    context: TForwardContext,
    dist_stream: Optional[torch.Stream],
    batch: Optional[In] = None,
    apply_jit: bool = False,
    pipelined_forward: Type[BaseForward[TrainPipelineContext]] = PipelinedForward,
    pipeline_postproc: bool = False,
    default_stream: Optional[torch.Stream] = None,
) -> Tuple[
    List[ShardedModule],
    torch.nn.Module,
    List[Callable[..., Any]],
    List[PipelinedPostproc],
    List[str],
]:
    input_model = model
    # Get underlying nn.Module
    if isinstance(model, DistributedModelParallel):
        model = model.module
    if isinstance(model, DDP):
        model = model.module
    if isinstance(model, Float16Module):
        model = model.module
    # Collect a list of sharded modules.
    sharded_modules = {}
    for name, m in model.named_modules():
        if isinstance(m, ShardedModule):
            sharded_modules[name] = m

    # Trace a model.
    concrete_args = {}
    if batch:
        if hasattr(batch, "to_proxy"):
            # for some special models, it requires using "input"
            # as the key for input
            # pyre-ignore[16]: Variable[In (bound to Pipelineable)] has no attribute to_proxy.
            concrete_args["inputs"] = copy.copy(batch).to_proxy()
        elif hasattr(batch, "to_proxy_tuple"):
            # when the model is pre-fx traced or dynamo exported, the
            # inputs are already flattened, and therefore we use
            # tuple as concrete args that fx.trace will automatically
            # match with the argument names.
            # We pass in the model for the caller side to customize
            # the batch
            # pyre-ignore[16]: Variable[In (bound to Pipelineable)] has no attribute to_proxy_tuple.
            concrete_args = batch.to_proxy_tuple(model)

    # Select sharded modules, which are top-level in the forward call graph,
    # i.e. don't have input transformations, i.e. rely only on 'builtins.getattr'.
    pipelined_forwards = []
    original_forwards = []

    pipelined_postprocs: Set[PipelinedPostproc] = set()
    non_pipelined_sharded_modules = []

    # default path
    def mod_with_trace():
        tracer = Tracer(leaf_modules=_get_leaf_module_names(model))
        graph = tracer.trace(model, concrete_args=concrete_args)
        graph.print_tabular()
        for node in graph.nodes:
            if node.op == "call_module" and node.target in sharded_modules:
                total_num_args = len(node.args) + len(node.kwargs)
                if total_num_args == 0:
                    continue
                arg_info_list, num_found = _get_node_args(
                    model,
                    node,
                    pipelined_postprocs,
                    context,
                    pipeline_postproc,
                    default_stream=default_stream,
                    dist_stream=dist_stream,
                )
                if num_found == total_num_args:
                    logger.info(f"Module '{node.target}' will be pipelined")
                    child = sharded_modules[node.target]
                    original_forwards.append(child.forward)
                    child.forward = pipelined_forward(
                        node.target,
                        arg_info_list,
                        child,
                        context,
                        dist_stream,
                    )
                    pipelined_forwards.append(child)
                else:
                    logger.warning(
                        f"Module '{node.target}'' will not be pipelined, due to input modifications"
                    )
                    non_pipelined_sharded_modules.append(node.target)

        # JIT script unsharded modules if applicable.
        if apply_jit:
            graph_model = torch.fx.GraphModule(model, graph)
            _jit_modules(graph_model, "")
            if isinstance(input_model, DistributedModelParallel):
                input_model.module = graph_model

        if non_pipelined_sharded_modules:
            logger.warn(
                "Sharded modules were not pipelined: %s. "
                + "This should be fixed for pipelining to work to the full extent.",
                ", ".join(non_pipelined_sharded_modules),
            )

    # hack path
    def mod_directly():
        kjt_name = None
        for attr_name, attr_value in batch.__dict__.items():
            if isinstance(attr_value, KeyedJaggedTensor):
                if kjt_name is not None:
                    raise ValueError(
                        f"Multiple KJT tensors found in batch: {kjt_name} and {attr_name}"
                    )
                kjt_name = attr_name
                continue
        if kjt_name is None:
            raise ValueError("No KJT tensor found in batch")
        for n, sharded_module in sharded_modules.items():
            original_forwards.append(sharded_module.forward)
            sharded_module.forward = pipelined_forward(
                n,
                [
                    ArgInfo(
                        input_attrs=["", kjt_name],
                        is_getitems=[False, False],
                        name=None,
                        postproc_modules=[None, None],
                        constants=[None, None],
                    )
                ],
                sharded_module,
                context,
                dist_stream,
            )
            pipelined_forwards.append(sharded_module)

    mod_directly()
    return (
        pipelined_forwards,
        input_model,
        original_forwards,
        list(pipelined_postprocs),
        non_pipelined_sharded_modules,
    )


def _override_input_dist_forwards(
    pipelined_modules: List[ShardedModule],
) -> List[Callable[[KeyedJaggedTensor], Awaitable[KJTAllToAllTensorsAwaitable]]]:
    """
    Overrides each input dist forward to support fusing the splits collective.
    NOTE: this can only be called after the input dists are initialized.
    """
    original_kjt_dist_forwards = []
    for module in pipelined_modules:
        for child_fqn, child_module in module.named_modules():
            if hasattr(child_module, "_has_uninitialized_input_dist"):
                assert (
                    not child_module._has_uninitialized_input_dist
                ), f"{child_fqn} has uninitialized input dist"

            if not hasattr(child_module, "_input_dists"):
                continue

            for input_dist in child_module._input_dists:
                if hasattr(input_dist, "_dist"):
                    assert isinstance(input_dist._dist, KJTAllToAll)
                    original_kjt_dist_forwards.append(input_dist._dist.forward)
                    input_dist._dist.forward = KJTAllToAllForward(
                        pg=input_dist._dist._pg,
                        splits=input_dist._dist._splits,
                        stagger=input_dist._dist._stagger,
                    )
    return original_kjt_dist_forwards


def _prefetch_embeddings(
    batch: In,
    context: PrefetchTrainPipelineContext,
    pipelined_modules: List[ShardedModule],
    device: torch.device,
    stream_context: torch.Stream,
    data_dist_stream: Optional[torch.Stream],
    default_stream: Optional[torch.Stream],
) -> Dict[str, KJTList]:
    data_per_sharded_module = {}
    for sharded_module in pipelined_modules:
        forward = sharded_module.forward
        assert isinstance(forward, PrefetchPipelinedForward)

        assert forward._name in context.input_dist_tensors_requests
        request = context.input_dist_tensors_requests.pop(forward._name)
        assert isinstance(request, Awaitable)
        with record_function("## wait_sparse_data_dist ##"):
            # Finish waiting on the dist_stream,
            # in case some delayed stream scheduling happens during the wait() call.
            with stream_context(data_dist_stream):
                data = request.wait()

        # Make sure that both result of input_dist and context
        # are properly transferred to the current stream.
        module_context = context.module_contexts[forward._name]
        if data_dist_stream is not None:
            torch.get_device_module(device).current_stream().wait_stream(
                data_dist_stream
            )
            cur_stream = torch.get_device_module(device).current_stream()

            assert isinstance(
                data, (torch.Tensor, Multistreamable)
            ), f"{type(data)} must implement Multistreamable interface"
            data.record_stream(cur_stream)
            data.record_stream(default_stream)

            module_context.record_stream(cur_stream)
            module_context.record_stream(default_stream)

        sharded_module.prefetch(
            ctx=module_context,
            dist_input=data,
            forward_stream=default_stream,
        )
        data_per_sharded_module[forward._name] = data
    return data_per_sharded_module
