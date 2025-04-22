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
import pytest
import torch

import commons.utils.initialize as init
from ops.pt_ops.pt_norm_mul_dropout import (
    pytorch_norm_mul_dropout,
)
from ops.triton_ops.triton_norm_mul_dropout import (
    triton_norm_mul_dropout,
)


@pytest.mark.parametrize("training", [True])
@pytest.mark.parametrize("concat_ux", [False])
@pytest.mark.parametrize("dropout_ratio", [0.0, 0.5])
@pytest.mark.parametrize("hidden_dim", [128, 512])
@pytest.mark.parametrize("seed", [1234])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_ln_mul_dropout(
    input_dtype, training, concat_ux, dropout_ratio, hidden_dim, seed
):
    torch.backends.cuda.matmul.allow_tf32 = False
    init.initialize_distributed()
    init.set_random_seed(1234)
    device = torch.cuda.current_device()
    eps = 1e-5
    batchsize = 128

    ln_weight = torch.nn.Parameter(torch.ones(hidden_dim, device=device))
    ln_bias = torch.nn.Parameter(torch.zeros(hidden_dim, device=device))

    ref_weight = ln_weight.detach().clone().requires_grad_(True)
    ref_bias = ln_bias.detach().clone().requires_grad_(True)

    x = (
        torch.empty(batchsize, hidden_dim, device=device, dtype=input_dtype)
        .uniform_(-0.1, 0.1)
        .requires_grad_(True)
    )
    u = (
        torch.empty(batchsize, hidden_dim, device=device, dtype=input_dtype)
        .fill_(0.5)
        .requires_grad_(True)
    )

    ref_x = x.detach().clone().requires_grad_(True)
    ref_u = u.detach().clone().requires_grad_(True)

    y = triton_norm_mul_dropout(
        x, u, ln_weight, ln_bias, eps, dropout_ratio, training, concat_ux, seed=seed
    )
    ref_y = pytorch_norm_mul_dropout(
        ref_x, ref_u, ref_weight, ref_bias, eps, dropout_ratio, training, concat_ux
    )
    if dropout_ratio == 0.0:
        torch.testing.assert_close(y, ref_y)

    dout = torch.empty_like(y).uniform_(-0.1, 0.1)

    y.backward(dout)
    ref_y.backward(dout)

    if dropout_ratio == 0.0:
        torch.testing.assert_close(ln_weight.grad, ref_weight.grad)
        torch.testing.assert_close(ln_bias.grad, ref_bias.grad)
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(u.grad, ref_u.grad)
