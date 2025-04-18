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
from abc import ABC, abstractmethod
from typing import Dict

import torch


class BaseInitializer(ABC):
    """
    A callable class interface that initializes a tensor. The `__call__()` method must be implemented in the child class.
    """

    @abstractmethod
    def __call__(self, tensor: torch.Tensor):
        """
        Initialize the given tensor.

        Args:
            tensor (torch.Tensor): The tensor to initialize.
        """
        ...

    @abstractmethod
    def get_type_str(self):
        ...

    @abstractmethod
    def get_params(self) -> Dict:
        ...


class NormalInitializer(BaseInitializer):
    """
    Initializer that generates tensors with a normal distribution.

    Args:
      mean (float, optional): The mean of the normal distribution. Defaults to 0.0.
      std (float, optional): The standard deviation of the normal distribution. Defaults to 1.0.

    Example:
      >>> import torch
      >>> from tensor_initializer import NormalInitializer
      >>> x = torch.empty(3, 4)
      >>> initializer = NormalInitializer(mean=0.0, std=1.0)
      >>> initializer(x)
      >>> print(x)
      tensor([[-0.2026,  0.6100, -0.0305,  0.6603],
              [ 0.1965,  0.1868, -0.6768, -0.4550],
              [-0.2756, -0.3978,  0.0739,  0.4324])
    """

    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self._type_str = "normal"
        self._mean = mean
        self._std = std

    def __call__(self, x: torch.Tensor):
        """
        Initialize the tensor with a normal distribution.

        Args:
            x (torch.Tensor): The tensor to initialize.
        """
        torch.nn.init.normal_(x, mean=self._mean, std=self._std)

    def get_type_str(self):
        return self._type_str

    def get_params(self):
        return dict(mean=self._mean, std=self._std)


class UniformInitializer(BaseInitializer):
    """
    Initializer that generates tensors with a uniform distribution.

    Args:
      low (float, optional): The lower bound of the uniform distribution. Defaults to 0.0.
      high (float, optional): The upper bound of the uniform distribution. Defaults to 1.0.

    Example:
      >>> import torch
      >>> from tensor_initializer import UniformInitializer
      >>> x = torch.empty(3, 4)
      >>> initializer = UniformInitializer(low=0.0, high=1.0)
      >>> initializer(x)
      >>> print(x)
      tensor([[0.2026, 0.6100, 0.0305, 0.6603],
              [0.1965, 0.1868, 0.6768, 0.4550],
              [0.2756, 0.3978, 0.
    """

    def __init__(self, low=0.0, high=1.0):
        super().__init__()
        self._type_str = "uniform"
        self._low = low
        self._high = high

    def __call__(self, x: torch.Tensor):
        """
        Initialize the tensor with a uniform distribution.

        Args:
            x (torch.Tensor): The tensor to initialize.
        """
        torch.nn.init.uniform_(x, a=self._low, b=self._high)

    def get_type_str(self):
        return self._type_str

    def get_params(self):
        return dict(low=self._low, high=self._high)


class ConstInitializer(BaseInitializer):
    """
    Initializer that generates tensors with a constant value.

    Args:
      value (float, optional): The constant value. Defaults to 0.0.

    Example:
      >>> import torch
      >>> from tensor_initializer import ConstInitializer
      >>> x = torch.empty(3, 4)
      >>> initializer = ConstInitializer(value=1.0)
      >>> initializer(x)
      >>> print(x)
      tensor([[1., 1., 1., 1.],
              [1., 1., 1., 1.],
              [1., 1., 1., 1.])
    """

    def __init__(self, value=0.0):
        super().__init__()
        self._type_str = "constant"
        self._value = value

    def __call__(self, x: torch.Tensor):
        """
        Initialize the tensor with a constant value.

        Args:
            x (torch.Tensor): The tensor to initialize.
        """
        with torch.no_grad():
            x.fill_(self._value)

    def get_type_str(self):
        return self._type_str

    def get_params(self):
        return dict(value=self._value)


class XavierUniformInitializer(BaseInitializer):
    """
    Initializer that generates tensors with a Xavier uniform distribution.

    see https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_

    Args:
      gain (float, optional): The gain of the Xavier uniform distribution. Defaults to 1.0.

    Example:
      >>> import torch
      >>> from tensor_initializer import XavierUniformInitializer
      >>> x = torch.empty(3, 4)
      >>> initializer = XavierUniformInitializer(gain=1.0)
      >>> initializer(x)
      >>> print(x)
      tensor([[ 0.2026,  0.6100, -0.0305,  0.6603],
              [ 0.1965,  0.1868, -0.6768, -0.4550],
              [-0.2756, -0.3978,  0.0739,  0.4324])
    """

    def __init__(self, gain=1.0):
        super().__init__()
        self._type_str = "xavier_uniform"
        self._gain = gain

    def __call__(self, x: torch.Tensor):
        """
        Initialize the tensor with a Xavier uniform distribution.

        Args:
            x (torch.Tensor): The tensor to initialize.
        """
        torch.nn.init.xavier_uniform_(x)

    def get_type_str(self):
        return self._type_str

    def get_params(self):
        return dict(gain=self._gain)


class XavierNormalInitializer(BaseInitializer):
    """
    Initializer that generates tensors with a Xavier normal distribution.

    see https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_normal_

    Args:
      gain (float, optional): The gain of the Xavier normal distribution. Defaults to 1.0.

    Example:
      >>> import torch
      >>> from tensor_initializer import XavierNormalInitializer
      >>> x = torch.empty(3, 4)
      >>> initializer = XavierNormalInitializer(gain=1.0)
      >>> initializer(x)
      >>> print(x)
      tensor([[ 0.2026,  0.6100, -0.0305,  0.6603],

    """

    def __init__(self, gain=1.0):
        super().__init__()
        self._type_str = "xavier_normal"
        self._gain = gain

    def __call__(self, x: torch.Tensor):
        """
        Initialize the tensor with a Xavier normal distribution.

        Args:
            x (torch.Tensor): The tensor to initialize.
        """
        torch.nn.init.xavier_normal_(x)

    def get_type_str(self):
        return self._type_str

    def get_params(self):
        return dict(gain=self._gain)


def init_2D_tensor_as_sequnece(x, dim=0):
    """
    Initialize a 2D tensor as a sequence.

    Args:
        x (torch.Tensor): The tensor to initialize.
        dim (int, optional): The dimension along which to initialize the sequence. Defaults to 0.

    Example:
      >>> import torch
      >>> from tensor_initializer import init_2D_tensor_as_sequnece
      >>> x = torch.empty(3, 4, device='cuda')
      >>> init_2D_tensor_as_sequnece(x, dim=0)
      >>> print(x)
      tensor([[0., 0., 0., 0.],
          [1., 1., 1., 1.],
          [2., 2., 2., 2.]], device='cuda:0')
      >>> init_2D_tensor_as_sequnece(x, dim=1)
      >>> print(x)
      tensor([[0., 1., 2., 3.],
          [0., 1., 2., 3.],
          [0., 1., 2., 3.]], device='cuda:0')
    """
    with torch.no_grad():
        shape = x.size()
        expand_size = [shape[0], shape[1]]
        expand_size[dim] = -1
        load_embedding_from_global_tensor_seq = (
            torch.arange(0, shape[dim]).cuda().to(x.dtype).unsqueeze(1 - dim)
        )
        x.copy_(load_embedding_from_global_tensor_seq.expand(expand_size))


_init_type_to_cls_map = {
    "normal": NormalInitializer,
    "uniform": UniformInitializer,
    "constant": ConstInitializer,
    "xavier_normal": XavierNormalInitializer,
    "xavier_uniform": XavierUniformInitializer,
}


def get_initializer_cls_from_type(type_str: str = "normal") -> type:
    """
    Get the initializer class from the type string.

    Args:
        type_str (str, optional): The type string of the initializer. The valid types are:
          ``'normal'``, ``'uniform'``, ``'constant'``, ``'xavier_normal'``, ``'xavier_uniform'``. Defaults to ``'normal'``.

    Returns:
        type: The initializer class.
    """
    global _init_type_to_cls_map
    return _init_type_to_cls_map[type_str]


def get_initializer_from_type(type_str: str = "normal", *args, **kwargs):
    """
    Get the initializer instance from the type string.

    Args:
        type_str (str, optional): The type string of the initializer. The valid types are:
          ``'normal'``, ``'uniform'``, ``'constant'``, ``'xavier_normal'``, ``'xavier_uniform'``. Defaults to ``'normal'``.

        *args: Additional arguments for the initializer.
        **kwargs: Additional keyword arguments for the initializer.

    Returns:
        BaseInitializer: The initializer instance.
    """
    cls_ = get_initializer_cls_from_type(type_str)
    return cls_(*args, **kwargs)
