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
import abc
from functools import partial
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F
from torchrec.sparse.jagged_tensor import JaggedTensor

from distributed_recommender.configs import HSTUConfig, KernelBackend
from distributed_recommender.data.utils import RankingBatch
from distributed_recommender.modules.jagged_module import JaggedData, JaggedModule
from distributed_recommender.modules.position_encoder import HSTUPositionalEncoder
from distributed_recommender.modules.utils import init_mlp_weights_optional_bias
from distributed_recommender.ops.jagged_tensor_op import concat_2D_jagged_tensors
from distributed_recommender.ops.length_to_offsets import length_to_complete_offsets
from distributed_recommender.ops.triton_ops.triton_jagged import (  # type: ignore[attr-defined]
    triton_concat_2D_jagged,
    triton_split_2D_jagged,
)
from distributed_recommender.utils.nvtx_op import output_nvtx_hook


class HSTUAttention(torch.nn.Module):
    """
    Base module interface for different HSTUAttention backends.

    """

    @abc.abstractmethod
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target <=> candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Abstract method for the forward pass of HSTUAttention.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """


class TorchHSTUAttention(HSTUAttention):
    """
    Native HUST implementation. All jagged inputs are padded to the maximum length before computation.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal

    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Forward pass of the TorchHSTUAttention module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """
        from distributed_recommender.ops.pt_ops.pt_hstu_attention import (
            pytorch_hstu_mha,
        )

        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )

        return pytorch_hstu_mha(
            max_seq_len=max_seqlen,
            alpha=1.0,
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_contextuals=num_contextuals,
            num_targets=num_candidates,
            causal=self.is_causal,
            dropout_pr=0.0,
            training=self.training,
            target_group_size=target_group_size,
        ).view(-1, self.num_heads * self.linear_dim)


class TritonHSTUAttention(HSTUAttention):
    """
    Triton-based HUST implementation.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal

    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:  # T, d
        """
        Forward pass of the TritonHSTUAttention module.

        Args:
             tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.
        Returns:
            torch.Tensor: Output tensor of shape (T, d).
        """
        from distributed_recommender.ops.triton_ops.triton_hstu_attention import (  # type: ignore[attr-defined]
            triton_hstu_mha,
        )

        assert (
            target_group_size is 1
        ), "target_group_size is not supported in TritonHSTUAttention"
        if num_contextuals is None:
            num_contextuals = 0
        assert isinstance(
            num_contextuals, int
        ), "num_contextuals must be an integer in TritonHSTUAttention"
        return triton_hstu_mha(
            N=max_seqlen,
            alpha=1.0,
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets=offsets,
            num_targets=num_candidates,
            causal=self.is_causal,
            contextual_seq_len=num_contextuals,
        ).view(-1, self.num_heads * self.linear_dim)


class FusedHSTUAttention(HSTUAttention):
    """
    Cutlass-based HUST implementations. This is the default implementation on pre hopper GPU.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        from hstu_attn import hstu_attn_varlen_func

        self._hstu_attn_varlen_func = hstu_attn_varlen_func
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal
        assert (
            self.linear_dim == self.attention_dim
        ), "only support linear_dim and attention_dim"

    @output_nvtx_hook(nvtx_tag="FusedHSTUAttn")
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the FusedHSTUAttention module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.

        Returns:
            torch.Tensor: Output tensor.
        """
        # TODO: remove once cutlass backend is ready
        assert (
            self.is_causal or num_contextuals is None
        ), "Only causal attention is supported when max_num_contextuals > 0 in cutlass backend"
        # (b * ~s, nh * hh)
        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )
        return self._hstu_attn_varlen_func(
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets_q=offsets.to(torch.int32),
            seq_offsets_k=offsets.to(torch.int32),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            num_contexts=num_contextuals,
            num_targets=num_candidates.to(torch.int32)
            if isinstance(num_candidates, torch.Tensor)
            else None,
            target_group_size=target_group_size,
            window_size=(-1, 0) if self.is_causal else (-1, -1),
            rab=None,
        ).view(-1, self.num_heads * self.linear_dim)


class FusedHSTUAttentionHopper(HSTUAttention):
    """
    Cutlass-based HUST implementation. This is the specialized implementation on hopper.

    Args:
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.
    """

    def __init__(
        self,
        num_heads: int,
        attention_dim: int,
        linear_dim: int,
        is_causal: bool,
    ):
        super().__init__()
        from hopper.flash_attn_interface import hstu_attn_varlen_func

        self._hstu_attn_varlen_func = hstu_attn_varlen_func
        self.num_heads = num_heads
        self.attention_dim = attention_dim
        self.linear_dim = linear_dim
        self.is_causal = is_causal
        assert (
            self.linear_dim == self.attention_dim
        ), "only support linear_dim and attention_dim"

    @output_nvtx_hook(nvtx_tag="FusedHSTUAttnHopper")
    def forward(
        self,
        tq: torch.Tensor,  # (T, d)
        tk: torch.Tensor,  # (T, d)
        tv: torch.Tensor,  # (T, d)
        offsets: torch.Tensor,  # (batch_size, 1)
        max_seqlen: int,
        target_group_size: int = 1,  # target == candidates
        num_candidates: Optional[torch.Tensor] = None,
        num_contextuals: Optional[Union[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the FusedHSTUAttentionHopper module.

        Args:
            tq (torch.Tensor): Query tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the query.
            tk (torch.Tensor): Key tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the key.
            tv (torch.Tensor): Value tensor of shape (T, d), where T is the total sequence length across all batches and d is the dimensionality of the value.
            offsets (torch.Tensor): Offsets tensor of shape (batch_size, 1), indicating the start position of each sequence in the batch.
            max_seqlen (int): The maximum sequence length across all batches.
            target_group_size (int): The size of the sub-candidate group where causal attention is applied only within a sub-group (usually in the case of ranking). Defaults to 1.
            num_candidates (torch.Tensor): Tensor containing the number of candidates for each sequence.
            num_contextuals (int | torch.Tensor | None): The number of contextuals for each sequence, could be a single integer or a tensor of shape (batch_size,) when different sequences have different number of contextuals.

        Returns:
            torch.Tensor: Output tensor.
        """
        # TODO: remove once cutlass backend is ready
        assert (
            self.is_causal or num_contextuals is None
        ), "Only causal attention is supported when max_num_contextuals > 0 in cutlass backend"
        # (b * ~s, nh * hh)
        if isinstance(num_contextuals, torch.Tensor):
            num_contextuals = num_contextuals.to(torch.int32)
        elif isinstance(num_contextuals, int):
            num_contextuals = (
                torch.tensor([num_contextuals], dtype=torch.int32, device=tq.device)
                .view(1)
                .expand(offsets.size(0) - 1)
                .contiguous()
            )
        # TODO: remove this once Hopper supports contextual mask bwd
        num_contextuals = None
        return self._hstu_attn_varlen_func(
            q=tq.view(-1, self.num_heads, self.attention_dim),
            k=tk.view(-1, self.num_heads, self.attention_dim),
            v=tv.view(-1, self.num_heads, self.linear_dim),
            seq_offsets_q=offsets.to(torch.int32),
            seq_offsets_k=offsets.to(torch.int32),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            num_contexts=num_contextuals,
            num_targets=num_candidates.to(torch.int32)
            if isinstance(num_candidates, torch.Tensor)
            else None,
            target_group_size=target_group_size,
            window_size=(-1, 0) if self.is_causal else (-1, -1),
            rab=None,
        ).view(-1, self.num_heads * self.linear_dim)


def create_hstu_attention(
    kernel_backend: KernelBackend,
    num_heads: int,
    attention_dim: int,
    linear_dim: int,
    is_causal: bool,
) -> HSTUAttention:
    """
    Factory function to create an HSTUAttention module based on the kernel backend.

    Args:
        kernel_backend (KernelBackend): The kernel backend to use.
        num_heads (int): Number of attention heads.
        attention_dim (int): Dimension of the attention.
        linear_dim (int): Dimension of the linear layer.
        is_causal (bool): Whether the attention is causal.

    Returns:
        HSTUAttention: The created HSTUAttention module.

    Raises:
        ValueError: If the kernel backend is not supported.
    """
    if kernel_backend == KernelBackend.CUTLASS:
        sm_major_version = torch.cuda.get_device_properties(0).major
        sm_minor_version = torch.cuda.get_device_properties(0).minor
        if sm_major_version == 9 and sm_minor_version == 0:
            return FusedHSTUAttentionHopper(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
        elif sm_major_version == 8 and sm_minor_version == 0:
            return FusedHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
        print(
            "CUTLASS backend only support H100, H20 and A100, fallback to PyTorch backend"
        )
    elif kernel_backend == KernelBackend.TRITON:
        if is_causal:
            return TritonHSTUAttention(
                num_heads,
                attention_dim,
                linear_dim,
                is_causal,
            )
        else:
            print(
                "Triton backend does not support is_causal=False, fallback to PyTorch backend"
            )
    return TorchHSTUAttention(
        num_heads,
        attention_dim,
        linear_dim,
        is_causal,
    )


class HSTULayer(JaggedModule):
    """
    One basic unit of HSTUBlock. Input and output are all JaggedData.

    Args:
        config (HSTUConfig): Configuration for the HSTU layer.
    """

    def __init__(self, config: HSTUConfig):
        assert (
            config.tensor_model_parallel_size == 1
        ), "HSTULayer does not support tensor model parallel"
        super().__init__(config=config)
        self._embedding_dim: int = config.hidden_size
        # per head dim;
        self._linear_dim_per_head: int = config.kv_channels
        self._attention_dim_per_head: int = config.kv_channels
        # dropout on proj_linear
        self._dropout_ratio: float = config.hidden_dropout
        # dropout on QK; not used now
        self._num_heads: int = config.num_attention_heads

        self._split_arg_list = [
            self._linear_dim_per_head * self._num_heads,
            self._linear_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
            self._attention_dim_per_head * self._num_heads,
        ]
        # [embedding_dim, 4 * num_head * head_dim]
        self._linear_uvqk = torch.nn.Linear(
            self._embedding_dim,
            sum(self._split_arg_list),
            bias=False,
        ).apply(
            partial(
                init_mlp_weights_optional_bias,
                inplace_initializer=self.config.init_method,
            )
        )

        self._linear_proj = torch.nn.Linear(
            self._linear_dim_per_head * self._num_heads,
            self._embedding_dim,
            bias=True,
        ).apply(
            partial(
                init_mlp_weights_optional_bias,
                inplace_initializer=self.config.init_method,
            )
        )

        self._eps = config.layernorm_epsilon
        self._target_group_size = config.target_group_size

        self._attn_func = create_hstu_attention(
            kernel_backend=config.kernel_backend,
            num_heads=self._num_heads,
            attention_dim=self._attention_dim_per_head,
            linear_dim=self._linear_dim_per_head,
            is_causal=config.is_causal,
        )

    def get_user_value_query_key_tensors(self, hidden_states: torch.Tensor):
        """
        Splits the hidden states into user, value, query, and key tensors.

        Args:
            hidden_states (torch.Tensor): The hidden states tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The user, value, query, and key tensors.
        """
        mixed_uvqk = self._linear_uvqk(hidden_states)
        mixed_uvqk = F.silu(mixed_uvqk)
        (user, value, query, key) = torch.split(
            mixed_uvqk,
            self._split_arg_list,
            dim=-1,
        )
        value = value.view(-1, self._num_heads * self._linear_dim_per_head)
        query = query.view(-1, self._num_heads * self._attention_dim_per_head)
        key = key.view(-1, self._num_heads * self._attention_dim_per_head)
        return user, value, query, key

    @output_nvtx_hook(nvtx_tag="HSTULayer", hook_tensor_attr_name="values")
    def forward(self, jd: JaggedData) -> JaggedData:
        """
        Forward pass of the HSTULayer

        Args:
            jd (JaggedData): The input jagged data

        Returns:
            Tensor: The output embeddings [\*, D]
        """
        # input is [*, h]
        x = jd.values
        normed_x = F.layer_norm(
            x, normalized_shape=[self._embedding_dim], eps=self._eps
        )
        tu, tv, tq, tk = self.get_user_value_query_key_tensors(normed_x)
        # TODO: remove contiguous once cutlass backend is ready
        jagged_attn_output = self._attn_func(
            tq.contiguous(),
            tk.contiguous(),
            tv.contiguous(),
            jd.seqlen_offsets,
            num_contextuals=jd.contextual_seqlen,
            num_candidates=jd.num_candidates,
            max_seqlen=jd.max_seqlen,
            target_group_size=self._target_group_size,
        )
        normed_attn_output = F.layer_norm(
            jagged_attn_output,
            normalized_shape=[self._num_heads * self._linear_dim_per_head],
            eps=self._eps,
        )
        o_input = tu * normed_attn_output
        parallel_input = F.dropout(
            o_input,
            p=self._dropout_ratio,
            training=self.training,
        )
        reduced_act = self._linear_proj(parallel_input)
        output = reduced_act + x
        return JaggedData(
            values=output,
            seqlen=jd.seqlen,
            seqlen_offsets=jd.seqlen_offsets,
            max_seqlen=jd.max_seqlen,
            max_num_candidates=jd.max_num_candidates,
            num_candidates=jd.num_candidates,
            num_candidates_offsets=jd.num_candidates_offsets,
            contextual_max_seqlen=jd.contextual_max_seqlen,
            contextual_seqlen=jd.contextual_seqlen,
            contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
            has_interleaved_action=jd.has_interleaved_action,
        )


class HSTUBlock(JaggedModule):
    """
    HSTUBlock module. A stack of HSTULayers.

    Args:
        config (HSTUConfig): Configuration for the HSTU block.
    """

    def __init__(
        self,
        config: HSTUConfig,
    ):
        super().__init__(config=config)
        self._training_dtype = torch.float32
        if self.config.bf16:
            self._training_dtype = torch.bfloat16
        if self.config.fp16:
            self._training_dtype = torch.float16

        self._positional_encoder: Optional[HSTUPositionalEncoder] = None
        if config.position_encoding_config is not None:
            self._positional_encoder = HSTUPositionalEncoder(
                num_position_buckets=config.position_encoding_config.num_position_buckets,
                num_time_buckets=config.position_encoding_config.num_time_buckets,
                embedding_dim=config.hidden_size,
                is_inference=False,
                use_time_encoding=config.position_encoding_config.use_time_encoding,
                training_dtype=self._training_dtype,
            )

        self._attention_layers = torch.nn.ModuleList(
            [HSTULayer(config) for l in range(self.config.num_layers)]
        )

    @output_nvtx_hook(nvtx_tag="hstu_preprocess")
    def hstu_preprocess(
        self, embeddings: Dict[str, JaggedTensor], batch: RankingBatch
    ) -> JaggedData:
        """
        Preprocesses the embeddings for use in the HSTU architecture.

        This method performs the following steps:
        1. **Interleaving**: If action embeddings are present, interleaves them with item embeddings.
        2. **Concatenation**: Concatenates contextual, item, and action embeddings for each sample, following the order specified in the batch.
        3. **Position Encoding**: Applies position encoding to the concatenated embeddings.

        Args:
            embeddings (Dict[str, JaggedTensor]): A dictionary of embeddings where each key corresponds to a feature name and the value is a jagged tensor.
            batch (RankingBatch): The batch of ranking data.

        Returns:
            JaggedData: The preprocessed jagged data, ready for further processing in the HSTU architecture.
        """
        item_jt = embeddings[batch.item_feature_name]  # history + candidate
        sequence_embeddings = item_jt.values()
        sequence_embeddings_lengths = item_jt.lengths()
        sequence_embeddings_lengths_offsets = item_jt.offsets()
        sequence_max_seqlen = batch.feature_to_max_seqlen[batch.item_feature_name]

        if batch.action_feature_name is not None:
            action_jt = embeddings[batch.action_feature_name]
            jagged_size = sequence_embeddings.size(0)
            embedding_dim = sequence_embeddings.size(1)
            sequence_embeddings = torch.cat(
                [sequence_embeddings, action_jt.values()], dim=1
            ).view(2 * jagged_size, embedding_dim)
            sequence_embeddings_lengths = sequence_embeddings_lengths * 2
            sequence_embeddings_lengths_offsets = (
                sequence_embeddings_lengths_offsets * 2
            )
            sequence_max_seqlen = sequence_max_seqlen * 2

        if batch.num_candidates is not None and batch.action_feature_name is not None:
            num_candidates = batch.num_candidates * 2
            max_num_candidates = batch.max_num_candidates * 2
        else:
            num_candidates = batch.num_candidates
            max_num_candidates = batch.max_num_candidates

        contextual_max_seqlen = 0
        contextual_seqlen = None
        contextual_seqlen_offsets = None
        if len(batch.contextual_feature_names) > 0:
            contextual_max_seqlens = [
                batch.feature_to_max_seqlen[name]
                for name in batch.contextual_feature_names
            ]
            contextual_embedding, contextual_seqlen = concat_2D_jagged_tensors(
                jagged_tensors=[
                    embeddings[name] for name in batch.contextual_feature_names
                ],
                max_seqlens=contextual_max_seqlens,
            )

            contextual_max_seqlen = sum(contextual_max_seqlens)
            contextual_seqlen_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                contextual_seqlen
            )

            sequence_embeddings = triton_concat_2D_jagged(
                max_seq_len=contextual_max_seqlen + sequence_max_seqlen,
                values_a=contextual_embedding,
                values_b=sequence_embeddings,
                offsets_a=contextual_seqlen_offsets,
                offsets_b=sequence_embeddings_lengths_offsets,
            )

            sequence_embeddings_lengths = (
                contextual_seqlen + sequence_embeddings_lengths
            )
            sequence_embeddings_lengths_offsets = (
                contextual_seqlen_offsets + sequence_embeddings_lengths_offsets
            )
            sequence_max_seqlen = sequence_max_seqlen + contextual_max_seqlen

        if self._positional_encoder is not None:
            sequence_embeddings = self._positional_encoder(
                max_seq_len=sequence_max_seqlen,
                seq_lengths=sequence_embeddings_lengths,
                seq_offsets=sequence_embeddings_lengths_offsets,
                seq_timestamps=None,
                seq_embeddings=sequence_embeddings,
                num_targets=num_candidates,
            )
        return JaggedData(
            values=sequence_embeddings.to(self._training_dtype),
            seqlen=sequence_embeddings_lengths,  # contextual + history + candidate
            seqlen_offsets=sequence_embeddings_lengths_offsets,
            max_seqlen=sequence_max_seqlen,
            max_num_candidates=max_num_candidates,
            num_candidates=num_candidates,
            num_candidates_offsets=length_to_complete_offsets(num_candidates)
            if num_candidates is not None
            else None,
            contextual_max_seqlen=contextual_max_seqlen,
            contextual_seqlen=contextual_seqlen,
            contextual_seqlen_offsets=contextual_seqlen_offsets,
            has_interleaved_action=batch.action_feature_name is not None,
        )

    @output_nvtx_hook(nvtx_tag="hstu_preprocess")
    def hstu_postprocess(self, jd: JaggedData) -> JaggedData:
        """
        Postprocess the output from the HSTU architecture.
        1. If max_num_candidates > 0, split and only keep last ``num_candidates`` embeddings as candidates embedding for further processing.
        2. Remove action embeddings if present. Only use item embedding for further processing.

        Args:
            jd (JaggedData): The jagged data output from the HSTU architecture that needs further processing.

        Returns:
            JaggedData: The postprocessed jagged data.
        """

        sequence_embeddings: torch.Tensor
        seqlen_offsets: torch.Tensor
        max_seqlen: int
        if jd.max_num_candidates > 0:
            seqlen_offsets = jd.num_candidates_offsets
            max_seqlen = jd.max_num_candidates
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.seqlen_offsets - jd.num_candidates_offsets,
                offsets_b=seqlen_offsets,
            )
        elif jd.contextual_max_seqlen > 0:
            seqlen_offsets = jd.seqlen_offsets - jd.contextual_seqlen_offsets
            max_seqlen = jd.max_seqlen - jd.contextual_max_seqlen
            _, sequence_embeddings = triton_split_2D_jagged(
                jd.values,
                jd.max_seqlen,
                offsets_a=jd.contextual_seqlen_offsets,
                offsets_b=seqlen_offsets,
            )
        else:
            sequence_embeddings = jd.values
            seqlen_offsets = jd.seqlen_offsets
            max_seqlen = jd.max_seqlen

        if jd.has_interleaved_action:
            sequence_embeddings = sequence_embeddings[0::2, ...]
            seqlen_offsets = seqlen_offsets // 2
            max_seqlen = max_seqlen // 2
        return JaggedData(
            values=sequence_embeddings,
            seqlen=(seqlen_offsets[1:] - seqlen_offsets[:-1]).to(jd.seqlen.dtype),
            seqlen_offsets=seqlen_offsets.to(jd.seqlen_offsets.dtype),
            max_seqlen=max_seqlen,
            has_interleaved_action=False,
        )

    @output_nvtx_hook(nvtx_tag="HSTUBlock", hook_tensor_attr_name="values")
    def forward(self, jd: JaggedData) -> JaggedData:
        """
        Forward pass of the HSTUBlock.

        Args:
            jd (JaggedData): The input jagged data.

        Returns:
            JaggedData: The output jagged data.
        """
        for hstu_layer in self._attention_layers:
            jd = hstu_layer(jd)
        return self.hstu_postprocess(jd)
