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


import fbgemm_gpu  # pylint: disable-unused-import
import pytest
import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

import distributed_recommender.utils.initialize as init
from distributed_recommender.configs import get_hstu_config
from distributed_recommender.configs.hstu_config import KernelBackend
from distributed_recommender.data.utils import Batch, FeatureConfig
from distributed_recommender.modules.hstu import HSTUBlock, create_hstu_attention
from distributed_recommender.ops.length_to_offsets import length_to_complete_offsets
from distributed_recommender.utils.tensor_initializer import UniformInitializer


@pytest.mark.parametrize(
    "contextual_feature_names", [[], ["user_feature0", "user_feature1"]]
)
@pytest.mark.parametrize("action_feature_name", ["action", None])
@pytest.mark.parametrize("max_num_candidates", [10, 0])
def test_hstu_preprocess(
    contextual_feature_names,
    action_feature_name,
    max_num_candidates,
    dim_size=8,
    batch_size=32,
    max_seqlen=20,
):
    init.initialize_distributed()
    init.set_random_seed(1234)
    device = torch.cuda.current_device()

    item_feature_name = "item"
    item_and_action_feature_names = (
        [item_feature_name]
        if action_feature_name is None
        else [item_feature_name, action_feature_name]
    )
    feature_configs = [
        FeatureConfig(
            feature_names=item_and_action_feature_names,
            max_item_ids=[1000 for _ in item_and_action_feature_names],
            max_sequence_length=max_seqlen,
            is_jagged=True,
        )
    ]
    for n in contextual_feature_names:
        feature_configs.append(
            FeatureConfig(
                feature_names=[n],
                max_item_ids=[1000],
                max_sequence_length=max_seqlen,
                is_jagged=True,
            )
        )

    batch = Batch.random(
        batch_size=batch_size,
        feature_configs=feature_configs,
        item_feature_name=item_feature_name,
        contextual_feature_names=contextual_feature_names,
        action_feature_name=action_feature_name,
        max_num_candidates=max_num_candidates,
        device=device,
    )

    hstu_config = get_hstu_config(
        hidden_size=dim_size,
        kv_channels=128,
        num_attention_heads=4,
        num_layers=1,
        init_method=UniformInitializer,
        position_encoding_config=None,
        dtype=torch.float,
    )
    hstu_block = HSTUBlock(hstu_config)

    seqlen_sum = torch.sum(batch.features.lengths()).cpu().item()
    embeddings = KeyedJaggedTensor.from_lengths_sync(
        keys=batch.features.keys(),
        values=torch.rand((seqlen_sum, dim_size), device=device),
        lengths=batch.features.lengths(),
    )
    embedding_dict = embeddings.to_dict()
    item_embedding = embedding_dict[item_feature_name].values()
    item_embedding_offests_cpu = embedding_dict[item_feature_name].offsets().cpu()
    if action_feature_name is not None:
        action_embedding = embedding_dict[action_feature_name].values()

    jd = hstu_block.hstu_preprocess(embeddings=embeddings, batch=batch)
    for sample_id in range(batch_size):
        start, end = jd.seqlen_offsets[sample_id], jd.seqlen_offsets[sample_id + 1]
        cur_sequence_embedding = jd.values[start:end, :]
        idx = 0
        for contextual_feature_name in contextual_feature_names:
            contextual_embedding = embedding_dict[contextual_feature_name].values()
            contextual_embedding_offsets_cpu = (
                embedding_dict[contextual_feature_name].offsets().cpu()
            )
            cur_start, cur_end = (
                contextual_embedding_offsets_cpu[sample_id],
                contextual_embedding_offsets_cpu[sample_id + 1],
            )
            for i in range(cur_start, cur_end):
                assert torch.allclose(
                    cur_sequence_embedding[idx, :], contextual_embedding[i, :]
                ), "contextual embedding not match"
                idx += 1
        cur_start, cur_end = (
            item_embedding_offests_cpu[sample_id],
            item_embedding_offests_cpu[sample_id + 1],
        )
        for i in range(cur_start, cur_end):
            assert torch.allclose(
                cur_sequence_embedding[idx, :], item_embedding[i, :]
            ), "item embedding not match"
            idx += 1
            if action_feature_name is not None:
                assert torch.allclose(
                    cur_sequence_embedding[idx, :], action_embedding[i, :]
                ), "action embedding not match"
                idx += 1

    result_jd = hstu_block.hstu_postprocess(jd)
    for sample_id in range(batch_size):
        start, end = (
            result_jd.seqlen_offsets[sample_id],
            result_jd.seqlen_offsets[sample_id + 1],
        )
        result_embedding = result_jd.values[start:end, :]
        cur_start, cur_end = (
            item_embedding_offests_cpu[sample_id],
            item_embedding_offests_cpu[sample_id + 1],
        )
        if max_num_candidates > 0:
            num_candidates_cpu = batch.num_candidates.cpu()
            cur_num_candidates_cpu = num_candidates_cpu[sample_id].item()
            candidate_embedding = item_embedding[
                cur_end - cur_num_candidates_cpu : cur_end, :
            ]
            assert torch.allclose(
                result_embedding, candidate_embedding
            ), "candidate embedding not match"
        else:
            all_item_embedding = item_embedding[cur_start:cur_end, :]
            assert torch.allclose(
                result_embedding, all_item_embedding
            ), "all item embedding not match"


@pytest.mark.parametrize("heads", [8])
@pytest.mark.parametrize("hidden_dim", [128])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        #    torch.float16
    ],
)
@pytest.mark.parametrize(
    "max_seqlen,max_num_candidates,max_num_contextuals",
    [
        (1024, 128, 6),
        (32, 0, 0),
        (1024, 128, 0),
    ],
)
@pytest.mark.parametrize("target_group_size", [2, 16, 256, 1])
@pytest.mark.parametrize("batchsize", [32])
@pytest.mark.parametrize(
    "is_causal,kernel_backend,fwd_rtol,fwd_atol,bwd_rtol,bwd_atol",
    [
        (True, KernelBackend.TRITON, 1e-7, 1e-5, 1e-7, 1e-5),
        (True, KernelBackend.CUTLASS, 1e-7, 1e-5, 1e-7, 1e-5),
        (False, KernelBackend.CUTLASS, 1e-7, 1e-5, 1e-7, 1e-5),
    ],
)
def test_hstu_attn(
    heads,
    hidden_dim,
    is_causal,
    dtype,
    max_seqlen,
    max_num_candidates,
    max_num_contextuals,
    target_group_size,
    batchsize,
    kernel_backend,
    fwd_rtol,
    fwd_atol,
    bwd_rtol,
    bwd_atol,
):
    if kernel_backend == KernelBackend.TRITON and target_group_size > 1:
        pytest.skip("Triton is not supported when target_group_size > 1")
    if kernel_backend == KernelBackend.TRITON and max_num_contextuals > 0:
        pytest.skip("Triton is not supported when max_num_contextuals > 0")
    # TODO: uncomment this once cutlass supports causal attention
    if not is_causal and max_num_contextuals > 0:
        pytest.skip("Only causal attention is supported when max_num_contextuals > 0")
    # TODO: remove this once Hopper supports contextual mask
    sm_major_version = torch.cuda.get_device_properties(0).major
    if sm_major_version > 8 and max_num_contextuals > 0:
        pytest.skip("Hopper does not support contextual mask")

    init.initialize_distributed()
    init.set_random_seed(1234)
    device = torch.cuda.current_device()

    if not is_causal:
        max_num_candidates = 0

    ref_hstu_attn = create_hstu_attention(
        KernelBackend.PYTORCH,
        num_heads=heads,
        attention_dim=hidden_dim,
        linear_dim=hidden_dim,
        is_causal=is_causal,
    )
    hstu_attn = create_hstu_attention(
        kernel_backend,
        num_heads=heads,
        attention_dim=hidden_dim,
        linear_dim=hidden_dim,
        is_causal=is_causal,
    )

    for _ in range(100):
        lengths = torch.randint(
            1, max_seqlen + 1, (batchsize,), device=device, dtype=torch.int
        )
        seq_offsets = length_to_complete_offsets(lengths)
        L = int(seq_offsets[-1].item())
        if max_num_candidates == 0:
            num_candidates = None
        else:
            num_candidates = torch.randint(
                0, max_num_candidates + 1, (batchsize,), device=device
            )
            num_candidates = torch.clamp(
                num_candidates, max=lengths - 1, min=torch.zeros_like(num_candidates)
            )  # at least 1 history
        if max_num_contextuals == 0:
            num_contextuals = None
        else:
            num_contextuals = torch.randint(
                0, max_num_contextuals + 1, (batchsize,), device=device, dtype=torch.int
            )
            num_contextuals = torch.clamp(
                num_contextuals,
                max=lengths - 1 - num_candidates
                if num_candidates is not None
                else lengths - 1,
                min=torch.zeros_like(num_contextuals),
            )  # at least 1 history!!

        x = torch.empty(
            (L, heads, hidden_dim * 3),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-0.1, 0.1)
        q, k, v = torch.split(x, [hidden_dim, hidden_dim, hidden_dim], dim=-1)
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)

        ref_out = ref_hstu_attn(
            q,
            k,
            v,
            seq_offsets,
            num_candidates=num_candidates,
            num_contextuals=num_contextuals,
            max_seqlen=max_seqlen,
        )
        dout = torch.randn_like(ref_out) * 0.01
        ref_out.backward(dout)
        ref_dq = q.grad.clone()
        ref_dk = k.grad.clone()
        ref_dv = v.grad.clone()

        q = q.detach().clone().requires_grad_()
        k = k.detach().clone().requires_grad_()
        v = v.detach().clone().requires_grad_()
        dout = dout.detach().clone()
        out = hstu_attn(
            q,
            k,
            v,
            seq_offsets,
            num_candidates=num_candidates,
            num_contextuals=num_contextuals,
            max_seqlen=max_seqlen,
        )
        out.backward(dout)

        torch.testing.assert_close(ref_out, out, atol=fwd_atol, rtol=fwd_rtol)
        torch.testing.assert_close(ref_dq, q.grad, atol=bwd_atol, rtol=bwd_rtol)
        torch.testing.assert_close(ref_dk, k.grad, atol=bwd_atol, rtol=bwd_rtol)
        torch.testing.assert_close(ref_dv, v.grad, atol=bwd_atol, rtol=bwd_rtol)
    init.destroy_global_state()
