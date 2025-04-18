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
from functools import partial, wraps

import torch


def forward_nvtx_push_hook(module, input, msg=""):
    torch.cuda.nvtx.range_push(msg)


def forward_nvtx_pop_hook(module, input, output):
    torch.cuda.nvtx.range_pop()


def backward_nvtx_push_hook(module, grad_output, msg=""):
    torch.cuda.nvtx.range_push(msg)


def backward_nvtx_pop_hook(module, grad_input, grad_output):
    torch.cuda.nvtx.range_pop()


def register_nvtx_for_module(module: torch.nn.Module, msg=""):
    module.register_forward_pre_hook(
        partial(forward_nvtx_push_hook, msg=msg + " forward")
    )
    module.register_forward_hook(forward_nvtx_pop_hook)
    module.register_full_backward_pre_hook(
        partial(backward_nvtx_push_hook, msg=msg + " backward")
    )
    module.register_full_backward_hook(backward_nvtx_pop_hook)


class _NvtxRangePush(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msg="None"):
        ctx.msg = msg
        torch.cuda.nvtx.range_push(msg)
        return input

    @staticmethod
    def backward(ctx, grad_in):
        torch.cuda.nvtx.range_pop()
        return grad_in, None


class _NvtxRangePop(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, msg="None"):
        ctx.msg = msg
        torch.cuda.nvtx.range_pop()
        return input

    @staticmethod
    def backward(ctx, grad_in):
        msg = ctx.msg
        torch.cuda.nvtx.range_push(msg)
        return grad_in, None


class _binary_identity_op(torch.autograd.Function):
    """
    input must be tensor
    """

    @staticmethod
    def forward(ctx, input_l, input_r):
        return input_l, input_r

    @staticmethod
    def backward(ctx, grad_l, grad_r):
        return grad_l, grad_r


## usage
NvtxRangePush = _NvtxRangePush.apply
NvtxRangePop = _NvtxRangePop.apply
binary_identity_op = _binary_identity_op.apply


def output_nvtx_hook(nvtx_tag, hook_tensor_attr_name: str = "", backward=True):
    def decorator_forward_only(module):
        @wraps(module)
        def forward(*args, **kwags):
            torch.cuda.nvtx.range_push(nvtx_tag)
            output = module(*args, **kwags)
            torch.cuda.nvtx.range_pop()
            return output

        return forward

    def jagged_decorator_include_backward(module):
        @wraps(module)
        def forward(*args, **kwags):
            _placeholder = torch.zeros(1, device="cuda").requires_grad_()
            hook_r = NvtxRangePush(_placeholder, nvtx_tag + " forward")
            output = module(*args, **kwags)
            hook_l = getattr(output, hook_tensor_attr_name)
            hook_l, _ = binary_identity_op(hook_l, hook_r)
            hook_l = NvtxRangePop(hook_l, nvtx_tag + " backward")
            setattr(output, hook_tensor_attr_name, hook_l)
            return output

        return forward

    def decorator_include_backward(module):
        @wraps(module)
        def forward(*args, **kwags):
            _placeholder = torch.zeros(1, device="cuda").requires_grad_()
            hook_r = NvtxRangePush(_placeholder, nvtx_tag + " forward")
            output = module(*args, **kwags)
            output, _ = binary_identity_op(output, hook_r)
            output = NvtxRangePop(output, nvtx_tag + " backward")
            return output

        return forward

    if not backward:
        decorator = decorator_forward_only
    elif hook_tensor_attr_name:
        decorator = jagged_decorator_include_backward
    else:
        decorator = decorator_include_backward
    return decorator
