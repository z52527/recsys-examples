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
# Copyright (c) 2024, NVIDIA Corporation & AFFILIATES.
import torch

sm_major_version = torch.cuda.get_device_properties(0).major
sm_minor_version = torch.cuda.get_device_properties(0).minor
if sm_major_version == 9 and sm_minor_version == 0:
    from hstu_attn_interface import hstu_attn_qkvpacked_func, hstu_attn_varlen_func
elif sm_major_version == 8:
    from hstu_attn import hstu_attn_varlen_func, hstu_attn_qkvpacked_func

import math
from typing import Optional, Tuple

import pytest
import torch.nn.functional as F
from einops import rearrange


def pad_input(unpadded_input, cu_seqlen, batch, seqlen):
    indices = []
    for i in range(batch):
        indices.append(
            torch.arange(seqlen * i, seqlen * i + cu_seqlen[i + 1] - cu_seqlen[i])
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype,
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def pad_input_delta_q(unpadded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    indices = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        indices.append(
            torch.arange(
                seqlen * i + act_seqlen_k - act_seqlen_q, seqlen * i + act_seqlen_k
            )
        )
    indices = torch.cat(indices)
    output = torch.zeros(
        (batch * seqlen),
        *unpadded_input.shape[1:],
        device=unpadded_input.device,
        dtype=unpadded_input.dtype,
    )
    output[indices] = unpadded_input
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(padded_input, cu_seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(len(cu_seqlen) - 1):
        output.append(padded_input[i, : (cu_seqlen[i + 1] - cu_seqlen[i]), :])
    return torch.cat(output, dim=0)


def unpad_input_delta_q(padded_input, cu_seqlen_q, cu_seqlen_k, batch, seqlen):
    padded_input.reshape(padded_input.size(0), padded_input.size(1), -1)
    output = []
    for i in range(batch):
        act_seqlen_q = (cu_seqlen_q[i + 1] - cu_seqlen_q[i]).item()
        act_seqlen_k = (cu_seqlen_k[i + 1] - cu_seqlen_k[i]).item()
        output.append(padded_input[i, act_seqlen_k - act_seqlen_q : act_seqlen_k, :])
    return torch.cat(output, dim=0)


def construct_mask(
    seqlen_c,
    seqlen,
    seqlen_t=0,
    target_group_size=1,
    window_size=(-1, -1),  # -1 means infinite window size
    seq_offsets=None,
    num_contexts=None,
    device=None,
):
    seqlen = seqlen_c + seqlen + seqlen_t
    bs = seq_offsets.size(0) - 1

    mask = torch.zeros((seqlen, seqlen), device=device, dtype=torch.bool)
    if window_size[0] < 0 and window_size[1] == 0:
        # causal mask
        for i in range(seqlen):
            mask[i, : i + 1] = True

        # context mask
        if seqlen_c != 0:
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                mask[i, 0, : num_contexts[i], :target_start] = True

        # target mask
        if seqlen_t != 0:
            mask = (
                mask.unsqueeze(0).unsqueeze(0).repeat(bs, 1, 1, 1)
                if mask.ndim == 2
                else mask
            )
            for i in range(bs):
                target_start = (
                    num_contexts[i] + seq_offsets[i + 1] - seq_offsets[i]
                ).item()
                # target group mask
                if target_group_size > 1:
                    group_num = math.ceil((seqlen - target_start) / target_group_size)
                    for j in range(group_num):
                        for k in range(
                            min(
                                target_group_size,
                                seqlen - target_start - j * target_group_size,
                            )
                        ):
                            mask[
                                i,
                                0,
                                target_start + j * target_group_size + k,
                                target_start : target_start + j * target_group_size,
                            ] = False
                else:
                    for j in range(target_start, seqlen):
                        mask[i, 0, j, target_start:j] = False

    # local mask
    else:
        window_size_0 = window_size[0] if window_size[0] > 0 else seqlen
        window_size_1 = window_size[1] if window_size[1] > 0 else seqlen
        for i in range(seqlen):
            mask[
                i, max(0, i - window_size_0) : min(seqlen, i + window_size_1 + 1)
            ] = True
    return mask


def generate_input(
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    window_size: Tuple[int, int],
    dtype: torch.dtype,
    full_batch: bool,
    has_drab: bool,
    is_delta_q: bool,
):
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    target_group_size > 1
    # Generate lengths for context
    if max_context_len > 0:
        if full_batch:
            num_contexts = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_context_len
            )
        else:
            num_contexts = torch.randint(
                0,
                max_context_len + 1,
                size=(batch_size,),
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
    else:
        num_contexts = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cuda")
        )
    seq_offsets_c = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_c[1:] = torch.cumsum(num_contexts, dim=0)

    # Generate lengths for historial qkv
    if full_batch:
        lengths_k = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_seq_len_k
        )
    else:
        lengths_k = torch.randint(
            1, max_seq_len_k + 1, size=(batch_size,), device=torch.device("cuda")
        )
    seq_offsets_k = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_k[1:] = torch.cumsum(lengths_k, dim=0)

    # Generate lengths for target qkv
    if has_target:
        if full_batch:
            num_targets = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_target_len
            )
        else:
            num_targets = torch.randint(
                0,
                max_target_len + 1,
                size=(batch_size,),
                dtype=torch.int32,
                device=torch.device("cuda"),
            )
    else:
        num_targets = torch.zeros(
            (batch_size,), dtype=torch.int32, device=torch.device("cuda")
        )
    seq_offsets_t = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_t[1:] = torch.cumsum(num_targets, dim=0)

    # Generate lengths for delta q
    if is_delta_q:
        if full_batch:
            lengths_q = (
                torch.ones(
                    (batch_size,), device=torch.device("cuda"), dtype=torch.int32
                )
                * max_seq_len_q
            )
        else:
            # lengths_q[i] is an integer between 1 and min(max_seq_len_q, lengths_k[i])
            lengths_q = torch.zeros(
                (batch_size,), device=torch.device("cuda"), dtype=torch.int32
            )
            for i in range(batch_size):
                lengths_q[i] = torch.randint(
                    1,
                    min(max_seq_len_q, lengths_k[i]) + 1,
                    size=(1,),
                    device=torch.device("cuda"),
                )
        seq_offsets_q = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
        )
        seq_offsets_q[1:] = torch.cumsum(lengths_q, dim=0)
    else:
        seq_offsets_q = seq_offsets_k

    # Lengths for whole q, kv
    seq_offsets_q_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_q_wt = seq_offsets_c + seq_offsets_q + seq_offsets_t
    seq_offsets_k_wt = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_k_wt = seq_offsets_c + seq_offsets_k + seq_offsets_t

    L_q = int(seq_offsets_q_wt[-1].item())
    L_k = int(seq_offsets_k_wt[-1].item())
    if dtype == torch.float8_e4m3fn:
        dtype_init = torch.float16
    else:
        dtype_init = dtype

    # Generate q, k, v for history + target
    q = (
        torch.empty(
            (L_q, heads, attn_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    k = (
        torch.empty(
            (L_k, heads, attn_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)
    v = (
        torch.empty(
            (L_k, heads, hidden_dim), dtype=dtype_init, device=torch.device("cuda")
        )
        .uniform_(-1, 1)
        .requires_grad_()
    ).to(dtype)

    if is_delta_q is False and dtype != torch.float8_e4m3fn:
        qkv = (
            torch.empty(
                (L_q, 3, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
            )
            .uniform_(-1, 1)
            .requires_grad_()
        )
        q = qkv[:, 0, :, :]
        k = qkv[:, 1, :, :]
        v = qkv[:, 2, :, :]
    else:
        qkv = None

    rab = torch.empty(
        (
            batch_size,
            heads if heads_rab is None else heads_rab,
            max_context_len + max_seq_len_k + max_target_len,
            max_context_len + max_seq_len_k + max_target_len,
        ),
        dtype=dtype_init,
        device=torch.device("cuda"),
    ).uniform_(-1, 1)
    if has_drab:
        rab = rab.requires_grad_()
    if window_size[0] == -1 and window_size[1] == -1:
        attn_mask = None
    else:
        attn_mask = (
            construct_mask(
                seqlen_c=max_context_len,
                seqlen=max_seq_len_k,
                seqlen_t=max_target_len,
                target_group_size=target_group_size,
                window_size=window_size,
                num_contexts=num_contexts,
                seq_offsets=seq_offsets_k,
            )
            .cuda()
            .to(torch.float32)
        )

    return (
        L_q,
        L_k,
        num_contexts if has_context else None,
        seq_offsets_q_wt,
        seq_offsets_k_wt,
        num_targets if has_target else None,
        qkv,
        q,
        k,
        v,
        rab,
        attn_mask,
    )


# @torch.compile
def _hstu_attention_maybe_from_cache(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
    is_delta_q: bool = False,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    dtype_out = q.dtype
    if is_delta_q:
        padded_q = pad_input_delta_q(q, q_offsets, k_offsets, B, seqlen_k)
    else:
        padded_q = pad_input(q, q_offsets, B, seqlen_q)
    padded_k = pad_input(k, k_offsets, B, seqlen_k)
    padded_v = pad_input(v, k_offsets, B, seqlen_k)

    padded_q = padded_q.view(B, seqlen_k, num_heads, attention_dim)
    padded_k = padded_k.view(B, seqlen_k, num_heads, attention_dim)
    padded_v = padded_v.view(B, seqlen_k, num_heads, linear_dim)
    if upcast:
        padded_q, padded_k, padded_v = (
            padded_q.float(),
            padded_k.float(),
            padded_v.float(),
        )
        if rab is not None:
            rab = rab.float()
    qk_attn = torch.einsum(
        "bnhd,bmhd->bhnm",
        padded_q,
        padded_k,
    )

    if rab is not None:
        padding = (
            0,
            qk_attn.shape[-1] - rab.shape[-1],
            0,
            qk_attn.shape[-2] - rab.shape[-2],
        )
        rab = F.pad(rab, padding, value=0)
        masked_qk_attn = qk_attn + rab
    else:
        masked_qk_attn = qk_attn
    masked_qk_attn = masked_qk_attn * alpha
    masked_qk_attn = F.silu(masked_qk_attn)
    masked_qk_attn = masked_qk_attn / seqlen_q
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        masked_qk_attn = (
            masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)[:, :, :, :]
        )

    attn_output = torch.einsum(
        "bhnm,bmhd->bnhd",
        masked_qk_attn,
        padded_v,
    )

    attn_output = attn_output.reshape(B, seqlen_k, num_heads * linear_dim)
    if is_delta_q:
        attn_output = unpad_input_delta_q(
            attn_output, q_offsets, k_offsets, B, seqlen_k
        )
    else:
        attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)

    return attn_output.to(dtype_out)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("heads", [2])
@pytest.mark.parametrize(
    "max_seq_len_q, max_seq_len_k, is_delta_q",
    [
        (32, 32, False),
        (99, 99, False),
        (111, 111, False),
        (256, 256, False),
        (1111, 1111, False),
        (27, 32, True),
        (8, 99, True),
        (51, 256, True),
        (1000, 1111, True),
        (160, 2000, True),
    ],
)
@pytest.mark.parametrize("max_context_len", [0, 11, 99, 160, 333])
@pytest.mark.parametrize(
    "max_target_len, window_size, target_group_size",
    [
        (0, (-1, -1), 1),
        (0, (11, 111), 1),
        (0, (111, 11), 1),
        (0, (111, 222), 1),
        (0, (-1, 0), 1),
        (32, (-1, 0), 1),
        (257, (-1, 0), 1),
        (1024, (-1, 0), 1),
        (111, (-1, 0), 11),
        (1111, (-1, 0), 222),
    ],
)
@pytest.mark.parametrize(
    "attn_dim, hidden_dim",
    [
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)  # attn_dim & hidden_dim cannot exceed 256
@pytest.mark.parametrize(
    "has_rab, has_drab, heads_rab",
    [
        (False, False, None),
        (True, False, None),  # None is heads_rab=heads
        (True, True, None),
        (True, False, 1),
        (True, True, 1),
    ],
)
@pytest.mark.parametrize("run_benchmark", [None])
@pytest.mark.parametrize("dtype", [torch.bfloat16])  # torch.float16
@pytest.mark.parametrize("full_batch", [True, False])
@pytest.mark.parametrize("alpha", [1.0, 1.0 / (100**0.5)])
def test_fused_attn(
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_context_len: int,
    max_target_len: int,
    target_group_size: int,
    attn_dim: int,
    hidden_dim: int,
    alpha: float,
    has_rab: bool,
    has_drab: bool,
    window_size: Tuple[int, int],
    run_benchmark: Optional[int],
    dtype: torch.dtype,
    full_batch: bool,
    is_delta_q: bool,
) -> None:
    has_context = max_context_len > 0
    has_target = max_target_len > 0
    group_target = target_group_size > 1
    is_causal = window_size[0] == -1 and window_size[1] == 0
    if dtype == torch.float8_e4m3fn:
        raise ValueError(
            "float8_e4m3fn is not supported, please use test_fused_attn_fp8 instead"
        )
    if has_drab and not has_rab:
        raise ValueError("has_drab is True but has_rab is False")
    if (has_target and not is_causal) or (
        has_target and (window_size[0] > 0 or window_size[1] > 0)
    ):
        raise ValueError(
            "has_target is True but is_causal is False or window_size is not (-1, -1)"
        )
    if (max_seq_len_q != max_seq_len_k) and not is_delta_q:
        raise ValueError("max_seq_len_q != max_seq_len_k but is_delta_q is False")
    if is_delta_q and max_seq_len_q > max_seq_len_k:
        raise ValueError("is_delta_q is True but max_seq_len_q > max_seq_len_k")
    if group_target and not has_target:
        raise ValueError("group_target is True but has_target is False")
    # TODO: find a better way to avoid these combinations
    if is_delta_q and has_target:
        return
    if is_delta_q and has_context:
        return
    if not is_causal and has_context:
        return
    if (window_size[0] > 0 or window_size[1] > 0) and has_context:
        return

    torch.cuda.synchronize()
    if run_benchmark is not None:
        assert run_benchmark in [
            0,
            1,
        ]  # 0 is run hstu benchmark and 1 is run torch benchmark
        iterations = 20
        profiler_step_start = 10

        input_datas = []
        for i in range(2):
            input_data = generate_input(
                batch_size=batch_size,
                heads=heads,
                heads_rab=heads_rab,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                max_context_len=max_context_len,
                max_target_len=max_target_len,
                target_group_size=target_group_size,
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                window_size=window_size,
                dtype=dtype,
                full_batch=full_batch,
                has_drab=has_drab,
                is_delta_q=is_delta_q,
            )
            input_datas.append(input_data)

        output_datas = []
        fwd_event_start = torch.cuda.Event(enable_timing=True)
        fwd_event_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(iterations):
            if i == profiler_step_start:
                fwd_event_start.record()
            (
                L_q,
                L_k,
                num_contexts,
                seq_offsets_q,
                seq_offsets_k,
                num_targets,
                qkv,
                q,
                k,
                v,
                rab,
                attn_mask,
            ) = input_datas[i % 2]
            if run_benchmark == 0:
                fwd_out = hstu_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    seq_offsets_q=seq_offsets_q,
                    seq_offsets_k=seq_offsets_k,
                    max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
                    max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
                    num_contexts=num_contexts if has_context else None,
                    num_targets=num_targets if has_target else None,
                    target_group_size=target_group_size,
                    window_size=window_size,
                    alpha=alpha,
                    rab=rab if has_rab else None,
                    has_drab=has_drab,
                    is_delta_q=is_delta_q,
                )
            else:
                assert run_benchmark == 1
                fwd_out = _hstu_attention_maybe_from_cache(
                    num_heads=heads,
                    attention_dim=attn_dim,
                    linear_dim=hidden_dim,
                    seqlen_q=max_context_len + max_seq_len_q + max_target_len,
                    seqlen_k=max_context_len + max_seq_len_k + max_target_len,
                    q=q.view(L_q, -1),
                    k=k.view(L_k, -1),
                    v=v.view(L_k, -1),
                    q_offsets=seq_offsets_q,
                    k_offsets=seq_offsets_k,
                    rab=rab if has_rab else None,
                    invalid_attn_mask=attn_mask.to(torch.float32)
                    if attn_mask is not None
                    else None,
                    alpha=alpha,
                    upcast=False,
                    is_delta_q=is_delta_q,
                )
            output_datas.append(fwd_out)
        fwd_event_stop.record()
        torch.cuda.synchronize()
        fwd_time = fwd_event_start.elapsed_time(fwd_event_stop) / (
            iterations - profiler_step_start
        )

        grads = []
        for i in range(iterations):
            grad = torch.rand_like(output_datas[i])
            grads.append(grad)

        bwd_event_start = torch.cuda.Event(enable_timing=True)
        bwd_event_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(iterations):
            if i == profiler_step_start:
                bwd_event_start.record()
            (
                L_q,
                L_k,
                num_contexts,
                seq_offsets_q,
                seq_offsets_k,
                num_targets,
                qkv,
                q,
                k,
                v,
                rab,
                attn_mask,
            ) = input_datas[i % 2]
            g = grads[i]
            fwd_out = output_datas[i]
            if not has_drab:
                if run_benchmark == 0:
                    (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
                        fwd_out,
                        (q, k, v),
                        g,
                        retain_graph=True,
                    )
                else:
                    (dq_torch, dk_torch, dv_torch) = torch.autograd.grad(
                        fwd_out,
                        (q, k, v),
                        g,
                        retain_graph=True,
                    )
            else:
                if run_benchmark == 0:
                    (dq_hstu, dk_hstu, dv_hstu, drab_hstu) = torch.autograd.grad(
                        fwd_out,
                        (q, k, v, rab),
                        g,
                        retain_graph=True,
                    )
                else:
                    (dq_torch, dk_torch, dv_torch, drab_torch) = torch.autograd.grad(
                        fwd_out,
                        (q, k, v, rab),
                        g,
                        retain_graph=True,
                    )
        bwd_event_stop.record()
        torch.cuda.synchronize()
        bwd_time = bwd_event_start.elapsed_time(bwd_event_stop) / (
            iterations - profiler_step_start
        )
        return fwd_time, bwd_time

    (
        L_q,
        L_k,
        num_contexts,
        seq_offsets_q,
        seq_offsets_k,
        num_targets,
        qkv,
        q,
        k,
        v,
        rab,
        attn_mask,
    ) = generate_input(
        batch_size=batch_size,
        heads=heads,
        heads_rab=heads_rab,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        max_context_len=max_context_len,
        max_target_len=max_target_len,
        target_group_size=target_group_size,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        window_size=window_size,
        dtype=dtype,
        full_batch=full_batch,
        has_drab=has_drab,
        is_delta_q=is_delta_q,
    )
    out_ref = _hstu_attention_maybe_from_cache(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        seqlen_q=max_context_len + max_seq_len_q + max_target_len,
        seqlen_k=max_context_len + max_seq_len_k + max_target_len,
        q=q.view(L_q, -1),
        k=k.view(L_k, -1),
        v=v.view(L_k, -1),
        q_offsets=seq_offsets_q,
        k_offsets=seq_offsets_k,
        rab=rab if has_rab else None,
        invalid_attn_mask=attn_mask.to(torch.float32)
        if attn_mask is not None
        else None,
        alpha=alpha,
        is_delta_q=is_delta_q,
    )

    torch_out = _hstu_attention_maybe_from_cache(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        seqlen_q=max_context_len + max_seq_len_q + max_target_len,
        seqlen_k=max_context_len + max_seq_len_k + max_target_len,
        q=q.view(L_q, -1),
        k=k.view(L_k, -1),
        v=v.view(L_k, -1),
        q_offsets=seq_offsets_q,
        k_offsets=seq_offsets_k,
        rab=rab if has_rab else None,
        invalid_attn_mask=attn_mask.to(torch.float32)
        if attn_mask is not None
        else None,
        alpha=alpha,
        upcast=False,
        is_delta_q=is_delta_q,
    )
    if qkv is None:
        hstu_out = hstu_attn_varlen_func(
            q=q.to(dtype),
            k=k.to(dtype),
            v=v.to(dtype),
            seq_offsets_q=seq_offsets_q,
            seq_offsets_k=seq_offsets_k,
            max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            alpha=alpha,
            rab=rab if has_rab else None,
            has_drab=has_drab,
            is_delta_q=is_delta_q,
        )
    else:
        hstu_out = hstu_attn_qkvpacked_func(
            qkv=qkv.to(dtype),
            seq_offsets_q=seq_offsets_q,
            seq_offsets_k=seq_offsets_k,
            max_seqlen_q=max_context_len + max_seq_len_q + max_target_len,
            max_seqlen_k=max_context_len + max_seq_len_k + max_target_len,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            alpha=alpha,
            rab=rab if has_rab else None,
            has_drab=has_drab,
            is_delta_q=is_delta_q,
        )

    print(f"Output max diff: {(hstu_out.view(L_q, -1) - out_ref).abs().max().item()}")
    print(f"Pytorch max diff: {(torch_out - out_ref).abs().max().item()}")

    print(f"Output mean diff: {(hstu_out.view(L_q, -1) - out_ref).abs().mean().item()}")
    print(f"Pytorch mean diff: {(torch_out - out_ref).abs().mean().item()}")

    assert (hstu_out.view(L_q, -1) - out_ref).abs().max().item() <= 2 * (
        torch_out - out_ref
    ).abs().max().item()

    g = torch.rand_like(torch_out)
    if not has_drab:
        (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
            out_ref, (q, k, v), g, retain_graph=True
        )
        (dq_torch, dk_torch, dv_torch) = torch.autograd.grad(
            torch_out, (q, k, v), g, retain_graph=True
        )
        if qkv is None:
            (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
                hstu_out,
                (q, k, v),
                g.view(-1, heads, hidden_dim),
                retain_graph=True,
            )
        else:
            (dqkv_hstu) = torch.autograd.grad(
                hstu_out,
                qkv,
                g.view(-1, heads, hidden_dim),
                retain_graph=True,
            )
            dq_hstu = dqkv_hstu[0][:, 0, :, :]
            dk_hstu = dqkv_hstu[0][:, 1, :, :]
            dv_hstu = dqkv_hstu[0][:, 2, :, :]
    else:
        (dq_ref, dk_ref, dv_ref, drab_ref) = torch.autograd.grad(
            out_ref, (q, k, v, rab), g, retain_graph=True
        )
        (dq_torch, dk_torch, dv_torch, drab_torch) = torch.autograd.grad(
            torch_out, (q, k, v, rab), g, retain_graph=True
        )
        if qkv is None:
            (dq_hstu, dk_hstu, dv_hstu, drab_hstu) = torch.autograd.grad(
                hstu_out,
                (q, k, v, rab),
                g.view(-1, heads, hidden_dim),
                retain_graph=True,
            )
        else:
            (dqkv_hstu, drab_hstu) = torch.autograd.grad(
                hstu_out,
                (qkv, rab),
                g.view(-1, heads, hidden_dim),
                retain_graph=True,
            )
            dq_hstu = dqkv_hstu[:, 0, :, :]
            dk_hstu = dqkv_hstu[:, 1, :, :]
            dv_hstu = dqkv_hstu[:, 2, :, :]

    print(f"dV max diff: {(dv_hstu - dv_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_torch - dv_ref).abs().max().item()}")

    print(f"dK max diff: {(dk_hstu - dk_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_torch - dk_ref).abs().max().item()}")

    print(f"dQ max diff: {(dq_hstu - dq_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_torch - dq_ref).abs().max().item()}")

    if has_drab:
        print(f"dRab max diff: {(drab_hstu - drab_ref).abs().max().item()}")
        print(f"dRab Pytorch max diff: {(drab_torch - drab_ref).abs().max().item()}")
    assert (dv_hstu - dv_ref).abs().max().item() <= 5 * (
        dv_torch - dv_ref
    ).abs().max().item()
    assert (dk_hstu - dk_ref).abs().max().item() <= 5 * (
        dk_torch - dk_ref
    ).abs().max().item()
    assert (dq_hstu - dq_ref).abs().max().item() <= 5 * (
        dq_torch - dq_ref
    ).abs().max().item()
    if has_drab:
        assert (drab_hstu - drab_ref).abs().max().item() <= 5 * (
            drab_torch - drab_ref
        ).abs().max().item()
    torch.cuda.synchronize()


def generate_paged_kv_input(
    batch_size: int,
    heads: int,
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_target_len: int,
    attn_dim: int,
    hidden_dim: int,
    page_size: int,
    dtype: torch.dtype,
    full_batch: bool,
):
    # Generate lengths for new history qkv
    if full_batch:
        lengths_q = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_seq_len_q
        )
    else:
        lengths_q = torch.randint(
            1, max_seq_len_q + 1, size=(batch_size,), device=torch.device("cuda")
        )
    seq_offsets_q = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_q[1:] = torch.cumsum(lengths_q, dim=0)

    # Generate lengths for target qkv
    if full_batch:
        num_targets = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_target_len
        )
    else:
        num_targets = torch.randint(
            0,
            max_target_len + 1,
            size=(batch_size,),
            dtype=torch.int32,
            device=torch.device("cuda"),
        )
    seq_offsets_t = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_t[1:] = torch.cumsum(num_targets, dim=0)

    # Lengths for new history + target qkv
    seq_offsets_q_wt = seq_offsets_q + seq_offsets_t
    L_q = int(seq_offsets_q_wt[-1].item())

    # Generate q, k, v for new history + target
    M_uvqk = torch.empty(
        (L_q, 4, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(-1, 1)
    q = M_uvqk[:, 2, :, :]
    k = M_uvqk[:, 3, :, :]
    v = M_uvqk[:, 1, :, :]

    # Generate user feature + previous history
    if full_batch:
        lengths_k = (
            torch.ones((batch_size,), device=torch.device("cuda"), dtype=torch.int32)
            * max_seq_len_k
        )
    else:
        lengths_k = torch.randint(
            1, max_seq_len_k + 1, size=(batch_size,), device=torch.device("cuda")
        )
    seq_offsets_k = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    seq_offsets_k[1:] = torch.cumsum(lengths_k, dim=0)

    # Lengths for user feature + previous history + new history
    lengths_k_cache = lengths_k + lengths_q
    lengths_page = (lengths_k_cache + page_size - 1) // page_size
    page_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int32, device=torch.device("cuda")
    )
    page_offsets[1:] = torch.cumsum(lengths_page, dim=0)

    total_page = int(page_offsets[-1].item())
    page_ids = torch.randperm(
        total_page, device=torch.device("cuda"), dtype=torch.int32
    )
    last_page_lens = ((lengths_k_cache - 1) % page_size + 1).to(torch.int32)
    kv_cache = torch.empty(
        (total_page, 2, page_size, heads, attn_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-1, 1)

    mask = torch.zeros(
        (
            batch_size,
            1,
            max_seq_len_q + max_seq_len_k + max_target_len,
            max_seq_len_q + max_seq_len_k + max_target_len,
        ),
        device=torch.device("cuda"),
        dtype=torch.float32,
    )
    for i in range(batch_size):
        # new history part
        for j in range(lengths_k_cache[i]):
            mask[i, 0, j, : j + 1] = 1.0

        # target part
        mask[
            i,
            0,
            lengths_k_cache[i] : lengths_k_cache[i] + num_targets[i],
            : lengths_k_cache[i],
        ] = 1.0
        for j in range(num_targets[i]):
            mask[i, 0, lengths_k_cache[i] + j, lengths_k_cache[i] + j] = 1.0

    return (
        L_q,
        seq_offsets_q_wt,
        seq_offsets_k + seq_offsets_q_wt,
        seq_offsets_t,
        num_targets,
        page_offsets,
        page_ids,
        last_page_lens,
        q,
        k,
        v,
        kv_cache,
        mask,
    )


# @torch.compile
def _hstu_paged_kv_attention(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    num_targets: torch.Tensor,
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    upcast: bool = True,
    kv_cache: torch.Tensor = None,
    page_offsets: torch.Tensor = None,
    page_ids: torch.Tensor = None,
    last_page_lens: torch.Tensor = None,
):
    k_con = torch.empty((0, num_heads, attention_dim), device=k.device, dtype=k.dtype)
    v_con = torch.empty((0, num_heads, attention_dim), device=v.device, dtype=v.dtype)

    for i in range(len(last_page_lens)):
        page_num = page_offsets[i + 1] - page_offsets[i]
        new_history_len = q_offsets[i + 1] - q_offsets[i] - num_targets[i]
        for j in range(page_num - 1):
            k_con = torch.cat(
                (k_con, kv_cache[page_ids[page_offsets[i] + j], 0, :, :, :]), dim=0
            )
            v_con = torch.cat(
                (v_con, kv_cache[page_ids[page_offsets[i] + j], 1, :, :, :]), dim=0
            )
        k_con = torch.cat(
            (
                k_con,
                kv_cache[
                    page_ids[page_offsets[i + 1] - 1], 0, : last_page_lens[i], :, :
                ],
            ),
            dim=0,
        )
        k_con = torch.cat(
            (k_con, k[(q_offsets[i] + new_history_len) : q_offsets[i + 1], :, :]), dim=0
        )
        v_con = torch.cat(
            (
                v_con,
                kv_cache[
                    page_ids[page_offsets[i + 1] - 1], 1, : last_page_lens[i], :, :
                ],
            ),
            dim=0,
        )
        v_con = torch.cat(
            (v_con, v[(q_offsets[i] + new_history_len) : q_offsets[i + 1], :, :]), dim=0
        )

    return _hstu_attention_maybe_from_cache(
        num_heads=num_heads,
        attention_dim=attention_dim,
        linear_dim=linear_dim,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        q=q,
        k=k_con,
        v=v_con,
        q_offsets=q_offsets,
        k_offsets=k_offsets,
        rab=None,
        invalid_attn_mask=invalid_attn_mask,
        alpha=alpha,
        upcast=upcast,
        is_delta_q=True,
    )


@pytest.mark.skipif(
    sm_major_version == 9, reason="paged_kv_attn is not supported on sm90"
)
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("heads", [2])
@pytest.mark.parametrize("max_seq_len_q", [21, 53, 201, 302])
@pytest.mark.parametrize("max_seq_len_k", [10, 99, 111, 256, 717])
@pytest.mark.parametrize("max_target_len", [0, 32, 111, 501])
@pytest.mark.parametrize(
    "attn_dim, hidden_dim",
    [
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)
@pytest.mark.parametrize("page_size", [32])
@pytest.mark.parametrize("run_benchmark", [None])
@pytest.mark.parametrize("dtype", [torch.bfloat16])  # torch.float16
@pytest.mark.parametrize("full_batch", [True, False])
@pytest.mark.parametrize("alpha", [1.0, 1.0 / (100**0.5)])
def test_paged_kv_attn(
    batch_size: int,
    heads: int,
    max_seq_len_q: int,  # new history length
    max_seq_len_k: int,  # user feature + previous history
    max_target_len: int,
    attn_dim: int,
    hidden_dim: int,
    page_size: int,
    alpha: float,
    run_benchmark: Optional[int],
    dtype: torch.dtype,
    full_batch: bool,
) -> None:
    if dtype == torch.float8_e4m3fn:
        raise ValueError("float8_e4m3fn for paged_kv_attn is not supported")
    if page_size != 32 and page_size != 64:
        raise ValueError("page_size must be 32 or 64")

    torch.cuda.synchronize()
    if run_benchmark is not None:
        assert run_benchmark in [
            0,
            1,
        ]  # 0 is run hstu benchmark and 1 is run torch benchmark
        iterations = 20
        profiler_step_start = 10

        input_datas = []
        for i in range(2):
            input_data = generate_paged_kv_input(
                batch_size=batch_size,
                heads=heads,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                max_target_len=max_target_len,
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                page_size=page_size,
                dtype=dtype,
                full_batch=full_batch,
            )
            input_datas.append(input_data)

        fwd_event_start = torch.cuda.Event(enable_timing=True)
        fwd_event_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(iterations):
            if i == profiler_step_start:
                fwd_event_start.record()
            (
                L_q,
                seq_offsets_q,
                seq_offsets_k,
                seq_offsets_t,
                num_targets,
                page_offsets,
                page_ids,
                last_page_lens,
                q,
                k,
                v,
                kv_cache,
                mask,
            ) = input_datas[i % 2]
            if run_benchmark == 0:
                fwd_out = hstu_attn_varlen_func(
                    q=q,
                    k=k,
                    v=v,
                    seq_offsets_q=seq_offsets_q,
                    seq_offsets_k=seq_offsets_q,
                    max_seqlen_q=max_seq_len_q + max_target_len,
                    max_seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
                    num_contexts=None,
                    num_targets=num_targets,
                    target_group_size=1,
                    window_size=(-1, 0),
                    alpha=alpha,
                    rab=None,
                    has_drab=False,
                    is_delta_q=True,
                    kv_cache=kv_cache,
                    page_offsets=page_offsets,
                    page_ids=page_ids,
                    last_page_lens=last_page_lens,
                )
            else:
                assert run_benchmark == 1
                fwd_out = _hstu_paged_kv_attention(
                    num_heads=heads,
                    attention_dim=attn_dim,
                    linear_dim=hidden_dim,
                    seqlen_q=max_seq_len_q + max_target_len,
                    seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
                    q=q,
                    k=k,
                    v=v,
                    q_offsets=seq_offsets_q,
                    k_offsets=seq_offsets_k,
                    num_targets=num_targets,
                    invalid_attn_mask=mask,
                    alpha=alpha,
                    upcast=False,
                    kv_cache=kv_cache,
                    page_offsets=page_offsets,
                    page_ids=page_ids,
                    last_page_lens=last_page_lens,
                )
        fwd_event_stop.record()
        torch.cuda.synchronize()
        fwd_time = fwd_event_start.elapsed_time(fwd_event_stop) / (
            iterations - profiler_step_start
        )
        return fwd_time, 0

    (
        L_q,
        seq_offsets_q,
        seq_offsets_k,
        seq_offsets_t,
        num_targets,
        page_offsets,
        page_ids,
        last_page_lens,
        q,
        k,
        v,
        kv_cache,
        mask,
    ) = generate_paged_kv_input(
        batch_size=batch_size,
        heads=heads,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        max_target_len=max_target_len,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        page_size=page_size,
        dtype=dtype,
        full_batch=full_batch,
    )
    out_ref = _hstu_paged_kv_attention(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        seqlen_q=max_seq_len_q + max_target_len,
        seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
        q=q,
        k=k,
        v=v,
        q_offsets=seq_offsets_q,
        k_offsets=seq_offsets_k,
        num_targets=num_targets,
        invalid_attn_mask=mask,
        alpha=alpha,
        upcast=True,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
    )

    torch_out = _hstu_paged_kv_attention(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        seqlen_q=max_seq_len_q + max_target_len,
        seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
        q=q,
        k=k,
        v=v,
        q_offsets=seq_offsets_q,
        k_offsets=seq_offsets_k,
        num_targets=num_targets,
        invalid_attn_mask=mask,
        alpha=alpha,
        upcast=False,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
    )

    hstu_out = hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        seq_offsets_q=seq_offsets_q,
        seq_offsets_k=seq_offsets_k,
        max_seqlen_q=max_seq_len_q + max_target_len,
        max_seqlen_k=max_seq_len_q + max_seq_len_k + max_target_len,
        num_contexts=None,
        num_targets=num_targets,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
        rab=None,
        has_drab=False,
        is_delta_q=True,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
        seq_offsets_t=seq_offsets_t,
    )

    print(f"Output max diff: {(hstu_out.view(L_q, -1) - out_ref).abs().max().item()}")
    print(f"Pytorch max diff: {(torch_out - out_ref).abs().max().item()}")

    print(f"Output mean diff: {(hstu_out.view(L_q, -1) - out_ref).abs().mean().item()}")
    print(f"Pytorch mean diff: {(torch_out - out_ref).abs().mean().item()}")

    assert (hstu_out.view(L_q, -1) - out_ref).abs().max().item() <= 2 * (
        torch_out - out_ref
    ).abs().max().item()
    torch.cuda.synchronize()


# @torch.compile
def _hstu_attention_maybe_from_cache_fp8(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: torch.Tensor,
    alpha: float,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    n_q: int = seqlen_q  # max_seq_len
    n_k: int = seqlen_k  # max_seq_len
    ori_n_q: int = n_q
    ori_n_k: int = n_k
    n_q = 16 * math.ceil(seqlen_q / 16)
    n_k = 16 * math.ceil(seqlen_k / 16)
    dtype_out = torch.float16
    padded_q = pad_input(q, q_offsets, B, n_q)
    padded_k = pad_input(k, k_offsets, B, n_k)
    padded_v = pad_input(v, k_offsets, B, n_k)

    padded_q = padded_q.view(B, n_q, num_heads, attention_dim)
    padded_k = padded_k.view(B, n_k, num_heads, attention_dim)
    padded_v = padded_v.view(B, n_k, num_heads, linear_dim)

    padded_q = padded_q.permute(0, 2, 1, 3).reshape(-1, n_q, attention_dim)
    padded_k = (
        padded_k.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1, n_k, attention_dim)
        .permute(0, 2, 1)
    )
    # only support MK @ KN
    qk_attn = torch._scaled_mm(
        padded_q[0],
        padded_k[0],
        out_dtype=torch.float,
        scale_a=descale_q,
        scale_b=descale_k,
    )
    for i in range(1, padded_q.size(0)):
        qk_attn = torch.cat(
            (
                qk_attn,
                torch._scaled_mm(
                    padded_q[i],
                    padded_k[i],
                    out_dtype=torch.float,
                    scale_a=descale_q,
                    scale_b=descale_k,
                ),
            ),
            dim=0,
        )
    qk_attn = qk_attn.view(B, num_heads, n_q, n_k)

    if rab is not None:
        padding = (
            0,
            qk_attn.shape[-1] - rab.shape[-1],
            0,
            qk_attn.shape[-2] - rab.shape[-2],
        )
        rab = F.pad(rab, padding, value=0)
        masked_qk_attn = qk_attn + rab
    else:
        masked_qk_attn = qk_attn

    masked_qk_attn = masked_qk_attn * alpha
    masked_qk_attn = F.silu(masked_qk_attn)
    masked_qk_attn = masked_qk_attn / ori_n_q
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            if invalid_attn_mask.shape[0] != n_q or invalid_attn_mask.shape[1] != n_k:
                invalid_attn_mask = F.pad(
                    invalid_attn_mask, (0, n_q - ori_n_q, 0, n_k - ori_n_k), value=1
                )
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        elif invalid_attn_mask.shape[2] != n_q or invalid_attn_mask.shape[3] != n_k:
            # pad 3rd and 4th dim
            invalid_attn_mask = F.pad(
                invalid_attn_mask,
                (0, 0, 0, 0, 0, n_q - ori_n_q, 0, n_k - ori_n_k),
                value=1,
            )
        masked_qk_attn = masked_qk_attn * invalid_attn_mask.type(masked_qk_attn.dtype)

    descale_a = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    masked_qk_attn = masked_qk_attn.to(torch.float8_e4m3fn).reshape(-1, n_q, n_k)
    padded_v = (
        padded_v.permute(0, 2, 3, 1)
        .contiguous()
        .permute(0, 1, 3, 2)
        .reshape(-1, n_k, linear_dim)
    )
    attn_output = torch._scaled_mm(
        masked_qk_attn[0],
        padded_v[0],
        out_dtype=torch.float,
        scale_a=descale_a,
        scale_b=descale_v,
    )
    for i in range(1, masked_qk_attn.size(0)):
        attn_output = torch.cat(
            (
                attn_output,
                torch._scaled_mm(
                    masked_qk_attn[i],
                    padded_v[i],
                    out_dtype=torch.float,
                    scale_a=descale_a,
                    scale_b=descale_v,
                ),
            ),
            dim=0,
        )
    attn_output = attn_output.view(B, num_heads, n_q, linear_dim).permute(0, 2, 1, 3)

    attn_output = attn_output.reshape(B, n_q, num_heads * linear_dim)[:, :ori_n_q, :]
    attn_output = unpad_input(attn_output, q_offsets)
    attn_output = attn_output.reshape(-1, num_heads * linear_dim)

    return attn_output.to(dtype_out)


def _bwd_reference_fp8(
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    seqlen_q: int,
    seqlen_k: int,
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    rab: Optional[torch.Tensor],
    invalid_attn_mask: Optional[torch.Tensor],
    alpha: float,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    descale_do: Optional[torch.Tensor] = None,
):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    B: int = q_offsets.size(0) - 1
    n_q: int = seqlen_q
    n_k: int = seqlen_k
    ori_n_q: int = n_q
    ori_n_k: int = n_k
    n_q = 16 * math.ceil(seqlen_q / 16)
    n_k = 16 * math.ceil(seqlen_k / 16)
    L: int = q.size(0)
    do.dtype

    padded_q = pad_input(q, q_offsets, B, n_q)
    padded_k = pad_input(k, k_offsets, B, n_k)
    padded_v = pad_input(v, k_offsets, B, n_k)
    padded_do = pad_input(do, q_offsets, B, n_q)

    padded_q = padded_q.view(B, n_q, num_heads, attention_dim)
    padded_k = padded_k.view(B, n_k, num_heads, attention_dim)
    padded_v = padded_v.view(B, n_k, num_heads, linear_dim)
    padded_do = padded_do.view(B, n_q, num_heads, linear_dim)

    def dsilu(dy, x):
        dy = dy.to(torch.float32)
        x = x.to(torch.float32)
        sigmoid = F.sigmoid(x)
        return dy * sigmoid * (1 + x * (1 - sigmoid))

    padded_q = padded_q.permute(0, 2, 1, 3).reshape(-1, n_q, attention_dim)
    padded_k = (
        padded_k.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1, n_k, attention_dim)
        .permute(0, 2, 1)
    )
    # only support MK @ KN
    qk_attn = torch._scaled_mm(
        padded_q[0],
        padded_k[0],
        out_dtype=torch.float,
        scale_a=descale_q,
        scale_b=descale_k,
    )
    for i in range(1, padded_q.size(0)):
        qk_attn = torch.cat(
            (
                qk_attn,
                torch._scaled_mm(
                    padded_q[i],
                    padded_k[i],
                    out_dtype=torch.float,
                    scale_a=descale_q,
                    scale_b=descale_k,
                ),
            ),
            dim=0,
        )
    qk_attn = qk_attn.view(B, num_heads, n_q, n_k)

    if rab is not None:
        padding = (
            0,
            qk_attn.shape[-1] - rab.shape[-1],
            0,
            qk_attn.shape[-2] - rab.shape[-2],
        )
        rab = F.pad(rab, padding, value=0)
        qk_attn = qk_attn + rab
    qk_attn = qk_attn * alpha
    qk_attn_silu = F.silu(qk_attn) / ori_n_k
    if invalid_attn_mask is not None:
        if invalid_attn_mask.ndim == 2:
            if invalid_attn_mask.shape[0] != n_q or invalid_attn_mask.shape[1] != n_k:
                invalid_attn_mask = F.pad(
                    invalid_attn_mask, (0, n_q - ori_n_q, 0, n_k - ori_n_k), value=1
                )
            invalid_attn_mask = invalid_attn_mask.unsqueeze(0).unsqueeze(0)
        elif invalid_attn_mask.shape[2] != n_q or invalid_attn_mask.shape[3] != n_k:
            # pad 3rd and 4th dim
            invalid_attn_mask = F.pad(
                invalid_attn_mask,
                (0, 0, 0, 0, 0, n_q - ori_n_q, 0, n_k - ori_n_k),
                value=1,
            )
        masked_qk_attn = qk_attn_silu * invalid_attn_mask.type(qk_attn_silu.dtype)
    else:
        masked_qk_attn = qk_attn_silu
    masked_qk_attn = (
        masked_qk_attn.to(torch.float8_e4m3fn)
        .reshape(-1, n_q, n_k)
        .permute(0, 2, 1)
        .contiguous()
    )

    padded_do = (
        padded_do.permute(0, 2, 3, 1)
        .reshape(-1, linear_dim, n_q)
        .contiguous()
        .permute(0, 2, 1)
    )
    dv = torch._scaled_mm(
        masked_qk_attn[0],
        padded_do[0],
        out_dtype=torch.float,
        scale_a=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        scale_b=descale_v,
    )
    for i in range(1, masked_qk_attn.size(0)):
        dv = torch.cat(
            (
                dv,
                torch._scaled_mm(
                    masked_qk_attn[i],
                    padded_do[i],
                    out_dtype=torch.float,
                    scale_a=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    scale_b=descale_v,
                ),
            ),
            dim=0,
        )
    dv = dv.view(B, num_heads, n_q, linear_dim).permute(0, 2, 1, 3)
    dv = unpad_input(dv, q_offsets)

    padded_do = padded_do.contiguous().permute(0, 2, 1)
    padded_v = padded_v.permute(0, 2, 1, 3).reshape(-1, n_k, linear_dim)
    dp = torch._scaled_mm(
        padded_v[0],
        padded_do[0],
        out_dtype=torch.float,
        scale_a=descale_v,
        scale_b=descale_do,
    )
    for i in range(1, padded_v.size(0)):
        dp = torch.cat(
            (
                dp,
                torch._scaled_mm(
                    padded_v[i],
                    padded_do[i],
                    out_dtype=torch.float,
                    scale_a=descale_v,
                    scale_b=descale_do,
                ),
            ),
            dim=0,
        )
    dp = dp.view(B, num_heads, n_k, n_q).permute(0, 1, 3, 2)

    if invalid_attn_mask is not None:
        dp = dp * invalid_attn_mask.type(dp.dtype)
    dp = dsilu(dp / ori_n_k, qk_attn)
    dp = dp * alpha

    dp = dp.to(torch.float8_e4m3fn).contiguous().reshape(-1, n_q, n_k)
    padded_k = padded_k.contiguous().permute(0, 2, 1)
    dq = torch._scaled_mm(
        dp[0],
        padded_k[0],
        out_dtype=torch.float,
        scale_a=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        scale_b=descale_k,
    )
    for i in range(1, dp.size(0)):
        dq = torch.cat(
            (
                dq,
                torch._scaled_mm(
                    dp[i],
                    padded_k[i],
                    out_dtype=torch.float,
                    scale_a=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    scale_b=descale_k,
                ),
            ),
            dim=0,
        )
    dq = dq.view(B, num_heads, n_q, attention_dim).permute(0, 2, 1, 3)
    dq = unpad_input(dq, q_offsets)

    dp = dp.permute(0, 2, 1).contiguous()
    padded_q = padded_q.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    dk = torch._scaled_mm(
        dp[0], padded_q[0], out_dtype=torch.float, scale_a=descale_do, scale_b=descale_q
    )
    for i in range(1, dp.size(0)):
        dk = torch.cat(
            (
                dk,
                torch._scaled_mm(
                    dp[i],
                    padded_q[i],
                    out_dtype=torch.float,
                    scale_a=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    scale_b=descale_q,
                ),
            ),
            dim=0,
        )
    dk = dk.view(B, num_heads, n_k, attention_dim).permute(0, 2, 1, 3)
    dk = unpad_input(dk, k_offsets)

    return (
        dq.view(-1, num_heads, attention_dim),
        dk.view(-1, num_heads, attention_dim),
        dv.view(-1, num_heads, linear_dim),
    )


@pytest.mark.skipif(
    sm_major_version == 8, reason="float8_e4m3fn is not supported on sm8x"
)
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("heads", [2])
@pytest.mark.parametrize(
    "max_seq_len_q, max_seq_len_k, is_delta_q",
    [
        (32, 32, False),
        (99, 99, False),
        (256, 256, False),
        (1024, 1024, False),
        (1111, 1111, False),
    ],
)
@pytest.mark.parametrize(
    "max_target_len, window_size",
    [
        (0, (-1, -1)),
        (0, (11, 111)),
        (0, (111, 222)),
        (0, (-1, 0)),
    ],
)
@pytest.mark.parametrize(
    "attn_dim, hidden_dim",
    [
        # (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
    ],
)  # attn_dim & hidden_dim cannot exceed 256
@pytest.mark.parametrize(
    "has_rab, has_drab, heads_rab",
    [
        (False, False, None),
        (True, False, None),  # None is heads_rab=heads
        (True, False, 1),
    ],
)
@pytest.mark.parametrize("run_benchmark", [None])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
@pytest.mark.parametrize("full_batch", [True, False])
@pytest.mark.parametrize("alpha", [1.0, 1.0 / (100**0.5)])
def test_fused_attn_fp8(
    batch_size: int,
    heads: int,
    heads_rab: Optional[int],
    max_seq_len_q: int,
    max_seq_len_k: int,
    max_target_len: int,
    attn_dim: int,
    hidden_dim: int,
    alpha: float,
    has_rab: bool,
    has_drab: bool,
    window_size: Tuple[int, int],
    run_benchmark: Optional[int],
    dtype: torch.dtype,
    full_batch: bool,
    is_delta_q: bool,
) -> None:
    if sm_major_version == 8:
        print("skipping fp8 test on sm8x")
        return
    if dtype != torch.float8_e4m3fn:
        raise ValueError("dtype is not float8_e4m3fn")
    if has_drab:
        raise ValueError("fp8 does not support drab")
    if max_target_len != 0:
        raise ValueError("for fp8, max_target_len must be 0")
    if max_seq_len_q != max_seq_len_k:
        raise ValueError("max_seq_len_q != max_seq_len_k")
    if is_delta_q:
        raise ValueError("fp8 does not support delta_q")

    torch.cuda.synchronize()
    if run_benchmark is not None:
        assert run_benchmark in [
            0,
            1,
        ]  # 0 is run hstu benchmark and 1 is run torch benchmark
        iterations = 20
        profiler_step_start = 10

        input_datas = []
        for i in range(2):
            input_data = generate_input(
                batch_size=batch_size,
                heads=heads,
                heads_rab=heads_rab,
                max_context_len=0,
                max_seq_len_q=max_seq_len_q,
                max_seq_len_k=max_seq_len_k,
                max_target_len=max_target_len,
                target_group_size=1,
                attn_dim=attn_dim,
                hidden_dim=hidden_dim,
                window_size=window_size,
                dtype=dtype,
                full_batch=full_batch,
                has_drab=has_drab,
                is_delta_q=is_delta_q,
            )
            input_datas.append(input_data)

        output_datas = []
        fwd_event_start = torch.cuda.Event(enable_timing=True)
        fwd_event_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(iterations):
            if i == profiler_step_start:
                fwd_event_start.record()
            (
                L_q,
                L_k,
                _,
                seq_offsets_q,
                seq_offsets_k,
                _,
                qkv,
                q,
                k,
                v,
                rab,
                attn_mask,
            ) = input_datas[i % 2]
            if run_benchmark == 0:
                fwd_out = hstu_attn_varlen_func(
                    q=q.to(dtype),
                    k=k.to(dtype),
                    v=v.to(dtype),
                    seq_offsets_q=seq_offsets_q,
                    seq_offsets_k=seq_offsets_k,
                    max_seqlen_q=max_seq_len_q,
                    max_seqlen_k=max_seq_len_k,
                    num_contexts=None,
                    num_targets=None,
                    target_group_size=1,
                    window_size=window_size,
                    alpha=alpha,
                    rab=rab if has_rab else None,
                    has_drab=has_drab,
                    is_delta_q=is_delta_q,
                    descale_q=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    descale_k=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    descale_v=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    descale_do=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                )
            else:
                assert run_benchmark == 1
                fwd_out = _hstu_attention_maybe_from_cache_fp8(
                    num_heads=heads,
                    attention_dim=attn_dim,
                    linear_dim=hidden_dim,
                    seqlen_q=max_seq_len_q + max_target_len,
                    seqlen_k=max_seq_len_k + max_target_len,
                    q=q.view(L_q, -1),
                    k=k.view(L_k, -1),
                    v=v.view(L_k, -1),
                    q_offsets=seq_offsets_q,
                    k_offsets=seq_offsets_k,
                    rab=rab if has_rab else None,
                    invalid_attn_mask=(
                        1.0 - attn_mask.to(torch.float32)
                        if attn_mask is not None
                        else None
                    ),
                    alpha=alpha,
                    descale_q=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    descale_k=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    descale_v=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                    descale_do=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
                )
            output_datas.append(fwd_out)
        fwd_event_stop.record()
        torch.cuda.synchronize()
        fwd_time = fwd_event_start.elapsed_time(fwd_event_stop) / (
            iterations - profiler_step_start
        )

        grads = []
        for i in range(iterations):
            grad = torch.rand_like(output_datas[i])
            grads.append(grad)

        bwd_event_start = torch.cuda.Event(enable_timing=True)
        bwd_event_stop = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        for i in range(iterations):
            if i == profiler_step_start:
                bwd_event_start.record()
            (
                L_q,
                L_k,
                _,
                seq_offsets_q,
                seq_offsets_k,
                _,
                qkv,
                q,
                k,
                v,
                rab,
                attn_mask,
            ) = input_datas[i % 2]
            g = grads[i]
            fwd_out = output_datas[i]
            if run_benchmark == 0:
                (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
                    fwd_out,
                    (q, k, v),
                    g,
                    retain_graph=True,
                )
            else:
                (dq_torch, dk_torch, dv_torch) = torch.autograd.grad(
                    fwd_out,
                    (q, k, v),
                    g,
                    retain_graph=True,
                )

        bwd_event_stop.record()
        torch.cuda.synchronize()
        bwd_time = bwd_event_start.elapsed_time(bwd_event_stop) / (
            iterations - profiler_step_start
        )
        return fwd_time, bwd_time

    (
        L_q,
        L_k,
        _,
        seq_offsets_q,
        seq_offsets_k,
        _,
        _,
        q,
        k,
        v,
        rab,
        attn_mask,
    ) = generate_input(
        batch_size=batch_size,
        heads=heads,
        heads_rab=heads_rab,
        max_seq_len_q=max_seq_len_q,
        max_seq_len_k=max_seq_len_k,
        max_context_len=0,
        max_target_len=max_target_len,
        target_group_size=1,
        attn_dim=attn_dim,
        hidden_dim=hidden_dim,
        window_size=window_size,
        dtype=dtype,
        full_batch=full_batch,
        has_drab=has_drab,
        is_delta_q=is_delta_q,
    )
    out_ref = _hstu_attention_maybe_from_cache(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        seqlen_q=max_seq_len_q,
        seqlen_k=max_seq_len_k,
        q=q.view(L_q, -1).float(),
        k=k.view(L_k, -1).float(),
        v=v.view(L_k, -1).float(),
        q_offsets=seq_offsets_q,
        k_offsets=seq_offsets_k,
        rab=rab if has_rab else None,
        invalid_attn_mask=attn_mask.to(torch.float32)
        if attn_mask is not None
        else None,
        alpha=alpha,
    )

    torch_out = _hstu_attention_maybe_from_cache_fp8(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        seqlen_q=max_seq_len_q,
        seqlen_k=max_seq_len_k,
        q=q.view(L_q, -1),
        k=k.view(L_k, -1),
        v=v.view(L_k, -1),
        q_offsets=seq_offsets_q,
        k_offsets=seq_offsets_k,
        rab=rab if has_rab else None,
        invalid_attn_mask=attn_mask.to(torch.float32)
        if attn_mask is not None
        else None,
        alpha=alpha,
        descale_q=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_k=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_v=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_do=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    )

    hstu_out = hstu_attn_varlen_func(
        q=q.to(dtype),
        k=k.to(dtype),
        v=v.to(dtype),
        seq_offsets_q=seq_offsets_q,
        seq_offsets_k=seq_offsets_k,
        max_seqlen_q=max_seq_len_q,
        max_seqlen_k=max_seq_len_k,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=window_size,
        alpha=alpha,
        rab=rab if has_rab else None,
        has_drab=has_drab,
        is_delta_q=is_delta_q,
        descale_q=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_k=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_v=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_do=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    )

    print(f"Output max diff: {(hstu_out.view(L_q, -1) - out_ref).abs().max().item()}")
    print(f"Pytorch max diff: {(torch_out - out_ref).abs().max().item()}")

    print(f"Output mean diff: {(hstu_out.view(L_q, -1) - out_ref).abs().mean().item()}")
    print(f"Pytorch mean diff: {(torch_out - out_ref).abs().mean().item()}")

    assert (hstu_out.view(L_q, -1) - out_ref).abs().max().item() <= 2 * (
        torch_out - out_ref
    ).abs().max().item()
    return

    # torch.set_printoptions(profile="full")
    g = torch.rand_like(torch_out)
    (dq_ref, dk_ref, dv_ref) = torch.autograd.grad(
        out_ref, (q, k, v), g, retain_graph=True
    )
    g = g.to(torch.float8_e4m3fn)
    (dq_torch, dk_torch, dv_torch) = _bwd_reference_fp8(
        num_heads=heads,
        attention_dim=attn_dim,
        linear_dim=hidden_dim,
        seqlen_q=max_seq_len_q + max_target_len,
        seqlen_k=max_seq_len_k + max_target_len,
        do=g,
        q=q,
        k=k,
        v=v,
        q_offsets=seq_offsets_q,
        k_offsets=seq_offsets_k,
        rab=rab if has_rab else None,
        invalid_attn_mask=attn_mask,
        alpha=alpha,
        descale_q=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_k=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_v=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
        descale_do=torch.tensor([1.0], dtype=torch.float32, device="cuda"),
    )
    (dq_hstu, dk_hstu, dv_hstu) = torch.autograd.grad(
        hstu_out,
        (q, k, v),
        g.view(-1, heads, hidden_dim),
        retain_graph=True,
    )

    print(f"dV max diff: {(dv_hstu - dv_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_torch - dv_ref).abs().max().item()}")

    print(f"dK max diff: {(dk_hstu - dk_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_torch - dk_ref).abs().max().item()}")

    print(f"dQ max diff: {(dq_hstu - dq_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_torch - dq_ref).abs().max().item()}")

    assert (dv_hstu - dv_ref).abs().max().item() <= 5 * (
        dv_torch - dv_ref
    ).abs().max().item()
    assert (dk_hstu - dk_ref).abs().max().item() <= 5 * (
        dk_torch - dk_ref
    ).abs().max().item()
    assert (dq_hstu - dq_ref).abs().max().item() <= 5 * (
        dq_torch - dq_ref
    ).abs().max().item()

    torch.cuda.synchronize()


if __name__ == "__main__":
    # test_paged_kv_attn(
    #     batch_size=1,
    #     heads=1,
    #     max_seq_len_q=32, # q is new history
    #     max_seq_len_k=32, # k is user feature + previous history  # kv cache = user feature + previous history + new history
    #     max_target_len=32, # target is target length
    #     attn_dim=32,
    #     hidden_dim=32,
    #     page_size=32,
    #     alpha=1.0,
    #     run_benchmark=None,
    #     dtype=torch.bfloat16,
    #     full_batch=False,
    # )

    # test_fused_attn(
    #     batch_size=128,
    #     heads=1,
    #     heads_rab=1,
    #     max_seq_len_q=1024,
    #     max_seq_len_k=1024,
    #     max_context_len=0,
    #     max_target_len=0,
    #     target_group_size=1,
    #     attn_dim=256,
    #     hidden_dim=256,
    #     alpha=1.0,
    #     has_rab=True,
    #     has_drab=True,
    #     window_size=(-1, 0),
    #     dtype=torch.bfloat16,
    #     run_benchmark=None,
    #     full_batch=False,
    #     is_delta_q=False,
    # )

    # test_fused_attn_fp8(
    #     batch_size=1,
    #     heads=1,
    #     heads_rab=None,
    #     max_seq_len_q=200,
    #     max_seq_len_k=200,
    #     max_target_len=0,
    #     attn_dim=64,
    #     hidden_dim=64,
    #     alpha=1.0,
    #     has_rab=False,
    #     has_drab=False,
    #     window_size=(-1, -1),
    #     dtype=torch.float8_e4m3fn,
    #     run_benchmark=None,
    #     full_batch=True,
    #     is_delta_q=False,
    # )

    bs = [8]
    dim = [32, 64, 128, 256]
    num_heads = 4

    has_rabs = [False]
    seq_lens = [8192, 4096]
    seq_lens_t = [0, 4096]
    has_drabs = [False]
    dytpes = [torch.bfloat16]
    for run_benchmark in [0]:
        for h in [num_heads]:
            for b in bs:
                for d in dim:
                    for seq_len, seq_len_t in zip(seq_lens, seq_lens_t):
                        for dtype in dytpes:
                            for has_rab in has_rabs:
                                for has_drab in has_drabs:
                                    fwd_time, bwd_time = test_paged_kv_attn(
                                        batch_size=b,
                                        heads=h,
                                        max_seq_len_q=seq_len // 2,  # q is new history
                                        max_seq_len_k=seq_len
                                        // 2,  # k is user feature + previous history  # kv cache = user feature + previous history + new history
                                        max_target_len=seq_len_t,  # target is target length
                                        attn_dim=d,
                                        hidden_dim=d,
                                        page_size=32,
                                        alpha=1.0,
                                        run_benchmark=run_benchmark,
                                        dtype=torch.bfloat16,
                                        full_batch=True,
                                    )
                                    # fwd_time, bwd_time = test_fused_attn(
                                    #     batch_size=b,
                                    #     heads=h,
                                    #     heads_rab=h,
                                    #     max_seq_len_q=seq_len,
                                    #     max_seq_len_k=seq_len,
                                    #     max_context_len=0,
                                    #     max_target_len=seq_len_t,
                                    #     target_group_size=1,
                                    #     attn_dim=d,
                                    #     hidden_dim=d,
                                    #     alpha=1.0,
                                    #     has_rab=has_rab,
                                    #     has_drab=has_drab,
                                    #     window_size=(-1, 0),
                                    #     run_benchmark=run_benchmark,
                                    #     dtype=dtype,
                                    #     full_batch=True,
                                    #     is_delta_q=False,
                                    # )
                                    info = [
                                        "hstu" if run_benchmark == 0 else "torch",
                                        b,
                                        seq_len,
                                        seq_len_t,
                                        d,
                                        h,
                                        has_rab,
                                        has_drab,
                                        (-1, 0),
                                        dtype,
                                        fwd_time,
                                        bwd_time,
                                    ]
                                    print(",".join([str(v) for v in info]))
