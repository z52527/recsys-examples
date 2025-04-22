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
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from commons.utils.logging import print_rank_0
from dynamicemb.dump_load import DynamicEmbDump as dynamic_emb_save
from dynamicemb.dump_load import DynamicEmbLoad as dynamic_emb_load
from dynamicemb.dump_load import find_sharded_modules
from megatron.core.distributed import DistributedDataParallel
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.transformer.module import Float16Module, MegatronModule
from torch import nn


def get_unwrapped_megatron_module(module):
    while isinstance(module, Float16Module) or isinstance(
        module, DistributedDataParallel
    ):
        module = module.module
    return module


def filter_megatron_module(module, remove_duplicate: bool = True):
    """
    Traverse the modules to find the Megatron modules.

    Args:
        module (nn.Module): The module to search.
        remove_duplicate (bool): Whether remove duplicated module in return iterator.

    Yields:
        Tuple[str, nn.Module]: The prefix and the Megatron module.
    """
    memo = set()
    search_list = [("", module)]

    while len(search_list) > 0:
        prefix, m = search_list.pop()
        m = get_unwrapped_megatron_module(m)
        if m in memo:
            continue
        if remove_duplicate:
            memo.add(m)
        if isinstance(m, MegatronModule):
            yield (prefix, m)
        else:
            for name, child in m.named_children():
                submodule_prefix = prefix + ("." if prefix else "") + name
                search_list.append((submodule_prefix, child))


def load_megatron_module(state_dict: dict, module: nn.Module):
    megatron_modules = list(filter_megatron_module(module))

    for n, m in megatron_modules:
        m.load_state_dict(state_dict["model_state_dict"][n])


def save_megatron_module(state_dict: dict, module: nn.Module):
    """
    Save the state dict from the Megatron module.

    Args:
        state_dict (dict): The state dict to save.
        module (nn.Module): The module to save the state dict from.
    """
    megatron_modules = list(filter_megatron_module(module))
    for n, m in megatron_modules:
        state_dict["model_state_dict"][n] = m.state_dict()


def save_sharded_module(state_dict: dict, module: nn.Module, include_optim_state=True):
    """
    Save the state dict from the sharded module. Please refer to `sharded module <https://pytorch.org/torchrec/concepts.html#distributed-training-with-torchrec-sharded-modules>`_ for definition of sharded module.

    Args:
        state_dict (dict): The state dict to save.
        module (nn.Module): The module to save the state dict from.
        include_optim_state (bool, optional): Whether to include the optimizer state. Defaults to True.
    """
    sharded_modules = find_sharded_modules(module)

    for p, n, m in sharded_modules:
        state_dict["model_state_dict"][p] = m.state_dict()
        if include_optim_state:
            state_dict["fused_optimizer_state_dict"][p] = m.fused_optimizer.state_dict()
    if include_optim_state:
        for n, m in module.named_modules():
            if hasattr(m, "_nonfused_embedding_optimizer"):
                state_dict["_nonfused_embedding_optimizer_state_dict"][
                    n
                ] = m._nonfused_embedding_optimizer.state_dict()


def load_sharded_module(state_dict: dict, module: nn.Module, include_optim_state=True):
    """
    Load the state dict into the sharded module. Please refer to `sharded module <https://pytorch.org/torchrec/concepts.html#distributed-training-with-torchrec-sharded-modules>`_  for definition of sharded module.

    Args:
        state_dict (dict): The state dict to load.
        module (nn.Module): The module to load the state dict into.
        include_optim_state (bool, optional): Whether to include the optimizer state. Defaults to True.
    """
    sharded_modules = find_sharded_modules(module)

    for p, n, m in sharded_modules:
        m.load_state_dict(state_dict["model_state_dict"][p])
        if include_optim_state and hasattr(m, "fused_optimizer"):
            m.fused_optimizer.load_state_dict(
                state_dict["fused_optimizer_state_dict"][p]
            )
    if include_optim_state:
        for n, m in module.named_modules():
            if hasattr(m, "_nonfused_embedding_optimizer"):
                m._nonfused_embedding_optimizer.load_state_dict(
                    state_dict["_nonfused_embedding_optimizer_state_dict"][n]
                )


def save(
    path: str,
    module: nn.Module,
    dense_optimizer: Optional[MegatronOptimizer] = None,
    include_optim_state=True,
):
    """
    Save the module and optimizer state to the given path.

    Args:
        path (str): The path to save the state.
        module (nn.Module): The module to save.
        dense_optimizer (Optional[MegatronOptimizer], optional): The optimizer to save. Defaults to None.
        include_optim_state (bool, optional): Whether to include the optimizer state. Defaults to True.

    Raises:
        FileExistsError: If the path does not exist or the save file already exists.
    """
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist.")
    save_dir = os.path.join(path, "dynamicemb_module")
    os.makedirs(save_dir, exist_ok=True)
    print_rank_0(f"dynamic module save dir {save_dir}")
    dynamic_emb_save(save_dir, module, optim=include_optim_state)

    save_dir = os.path.join(path, "torch_module")
    print_rank_0(f"torch module save dir {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    state_dict: Dict[str, Any] = defaultdict(dict)
    save_sharded_module(state_dict, module, include_optim_state=include_optim_state)
    save_megatron_module(state_dict, module)

    if dense_optimizer:
        state_dict["optimizer_state_dict"] = dense_optimizer.state_dict()

    save_path = os.path.join(save_dir, "model.{}.pth".format(dist.get_rank()))
    if os.path.exists(save_path):
        raise FileExistsError(f"{save_path} already exists.")
    torch.save(state_dict, save_path)


def load(
    path: str,
    module: nn.Module,
    dense_optimizer: Optional[MegatronOptimizer] = None,
    include_optim_state=True,
):
    """
    Load the module and optimizer state from the given path.

    Args:
        path (str): The path to load the state from.
        module (nn.Module): The module to load the state into.
        dense_optimizer (Optional[MegatronOptimizer], optional): The optimizer to load the state into. Defaults to None.
        include_optim_state (bool, optional): Whether to include the optimizer state. Defaults to True.
    """
    dist.barrier(device_ids=[torch.cuda.current_device()])
    save_dir = os.path.join(path, "dynamicemb_module")
    dynamic_emb_load(save_dir, module, optim=include_optim_state)

    save_path = os.path.join(
        path, "torch_module", "model.{}.pth".format(dist.get_rank())
    )
    state_dict = torch.load(save_path)
    load_sharded_module(state_dict, module, include_optim_state=include_optim_state)
    load_megatron_module(state_dict, module)
    if dense_optimizer:
        dense_optimizer.load_state_dict(state_dict["optimizer_state_dict"])
