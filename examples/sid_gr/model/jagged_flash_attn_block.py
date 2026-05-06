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
"""
JaggedFlashAttnBlock: a self-contained GPT Transformer block that uses
jiayus's Flash Attention with arbitrary_func mask encoding.

This replaces Megatron-Core's TransformerBlock for inference with
Method A (Incremental Append) beam search.

Architecture per layer (standard pre-norm GPT):
  Input → LayerNorm → QKV Projection → Flash Attention (arbitrary mask)
        → Output Projection → Residual
        → LayerNorm → FFN → Residual → Output

Reference: examples/hstu/modules/native_hstu_layer.py
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# flash_attn imports are deferred to runtime (inside functions / __init__)
# so that the module can be imported even without flash_attn installed.

# beam_decode_attn kernel import — deferred so module loads without the kernel.
# Falls back to a pure-PyTorch reference implementation when the CuTe kernel
# is not installed (requires ``quack`` / flash_attn CuTe DSL environment).
_beam_decode_attn = None


def _beam_decode_attn_reference(
    q: torch.Tensor,
    k_context: torch.Tensor,
    v_context: torch.Tensor,
    k_beam: torch.Tensor,
    v_beam: torch.Tensor,
    topk_indices: torch.Tensor,
    decode_nums: int,
    softmax_scale: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Pure-PyTorch reference for beam_decode_attn (single-pass).

    Shapes follow the CuTe kernel convention:
        q:            [B, Sq, W, Hq, D]
        k_context:    [B, Sk, Hkv, D]
        v_context:    [B, Sk, Hkv, D]
        k_beam:       [B, dn*W, Hkv, D]
        v_beam:       same
        topk_indices: [B, Sq, Hq, max_dn, W] int32
    Returns:
        out: [B, Sq, W, Hq, D]  (same dtype as q)
        lse: None
    """
    import math

    B, Sq, W, Hq, D = q.shape
    Hkv = k_context.shape[2]
    ngroups = Hq // Hkv
    Sk = k_context.shape[1]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(D)

    q_f = q.float()
    k_ctx_f = k_context.float()
    v_ctx_f = v_context.float()
    k_beam_f = k_beam.float()
    v_beam_f = v_beam.float()

    if ngroups > 1:
        k_ctx_f = k_ctx_f.repeat_interleave(ngroups, dim=2)
        v_ctx_f = v_ctx_f.repeat_interleave(ngroups, dim=2)
        k_beam_f = k_beam_f.repeat_interleave(ngroups, dim=2)
        v_beam_f = v_beam_f.repeat_interleave(ngroups, dim=2)

    # Context KV → [B, 1, 1, Hq, Sk, D]
    k_ctx_exp = k_ctx_f.permute(0, 2, 1, 3).unsqueeze(1).unsqueeze(2)
    k_ctx_exp = k_ctx_exp.expand(B, Sq, W, Hq, Sk, D)
    v_ctx_exp = v_ctx_f.permute(0, 2, 1, 3).unsqueeze(1).unsqueeze(2)
    v_ctx_exp = v_ctx_exp.expand(B, Sq, W, Hq, Sk, D)

    if decode_nums > 0:
        idx = topk_indices[:, :, :, :decode_nums, :]  # [B, Sq, Hq, dn, W]
        idx = idx.permute(0, 1, 4, 2, 3).contiguous()  # [B, Sq, W, Hq, dn]
        b_idx = torch.arange(B, device=q.device)[:, None, None, None, None]
        h_idx = torch.arange(Hq, device=q.device)[None, None, None, :, None]
        k_beam_g = k_beam_f[b_idx, idx, h_idx]  # [B, Sq, W, Hq, dn, D]
        v_beam_g = v_beam_f[b_idx, idx, h_idx]
        k_all = torch.cat([k_ctx_exp, k_beam_g], dim=4)
        v_all = torch.cat([v_ctx_exp, v_beam_g], dim=4)
    else:
        k_all = k_ctx_exp
        v_all = v_ctx_exp

    scores = torch.einsum("bqwhd,bqwhsd->bqwhs", q_f * softmax_scale, k_all)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bqwhs,bqwhsd->bqwhd", attn, v_all)
    return out.to(q.dtype), None


def _get_beam_decode_attn():
    global _beam_decode_attn
    if _beam_decode_attn is None:
        try:
            from interface import beam_decode_attn

            _beam_decode_attn = beam_decode_attn
        except ImportError:
            _beam_decode_attn = _beam_decode_attn_reference
    return _beam_decode_attn


def _build_padded_context_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    seqused: torch.Tensor,
    max_seqlen: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Identity pass-through for padded context K/V.

    Padding-aware masking is handled by the kernel via the ``seqused_k``
    argument (added in our local interface.py extension). This helper
    exists for symmetry with the test-side construction and may grow
    additional logic (e.g. reshape) in the future.
    """
    return k, v


def build_block_sparsity(
    arbitrary_func: torch.Tensor,
    seqlen_q: int,
    seqlen_k: int,
    headdim: int,
) -> Tuple[Optional[object], Optional[object]]:
    """
    Build forward (Q2K) and backward (K2Q) block sparsity indices from an
    arbitrary_func tensor.

    Returns (linear_k, linear_q) — either can be None if the CUDA extension
    is not available (falls back to dense attention).
    """
    try:
        import create_block_mask_cuda
        from flash_attn.cute.block_sparsity import LinearBlockSparseTensorsTorch
        from flash_attn.utils.tile_size import get_arch, get_tile_sizes_by_backend
    except ImportError:
        return None, None

    arch = get_arch()
    fwd_q_block, fwd_kv_block = get_tile_sizes_by_backend(
        backend="dsl",
        pass_type="forward",
        arch=arch,
        headdim=headdim,
        is_causal=False,
        is_local=False,
        is_arbitrary=True,
    )
    bwd_q_block, bwd_kv_block = get_tile_sizes_by_backend(
        backend="dsl",
        pass_type="backward",
        arch=arch,
        headdim=headdim,
        is_causal=False,
        is_local=False,
        is_arbitrary=True,
    )

    (
        k_cnt,
        k_off,
        k_idx,
        k_fcnt,
        k_foff,
        k_fidx,
    ) = create_block_mask_cuda.create_q2k_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=fwd_q_block,
        KV_BLOCK_SIZE=fwd_kv_block,
        check_q_boundary=True,
    )
    linear_k = LinearBlockSparseTensorsTorch(
        mask_block_cnt=k_cnt,
        mask_block_offset=k_off,
        mask_block_idx=k_idx,
        full_block_cnt=k_fcnt,
        full_block_offset=k_foff,
        full_block_idx=k_fidx,
    )

    (
        q_cnt,
        q_off,
        q_idx,
        q_fcnt,
        q_foff,
        q_fidx,
    ) = create_block_mask_cuda.create_k2q_csr_sparse_from_func(
        arbitrary_func,
        seqlen_q,
        seqlen_k,
        Q_BLOCK_SIZE=bwd_q_block,
        KV_BLOCK_SIZE=bwd_kv_block,
    )
    linear_q = LinearBlockSparseTensorsTorch(
        mask_block_cnt=q_cnt,
        mask_block_offset=q_off,
        mask_block_idx=q_idx,
        full_block_cnt=q_fcnt,
        full_block_offset=q_foff,
        full_block_idx=q_fidx,
    )

    return linear_k, linear_q


class JaggedGPTLayer(nn.Module):
    """
    One Transformer layer with jagged Flash Attention.

    Pre-norm GPT structure:
      x = x + Attn(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

    Q/K/V are produced by a single fused linear (same pattern as HSTU's
    ``linear_uvqk``). Flash Attention is called with arbitrary_func for
    tree-shaped beam search masks.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.ffn_hidden_size = ffn_hidden_size

        # --- Attention sub-layers ---
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        # Fused QKV projection: hidden_size → 3 * hidden_size
        self.linear_qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        # Output projection after attention
        self.linear_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(hidden_dropout)

        # --- FFN sub-layers ---
        self.pre_mlp_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)
        self.mlp_fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.mlp_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)
        self.mlp_dropout = nn.Dropout(hidden_dropout)

        if activation == "gelu":
            self.activation_fn = F.gelu
        elif activation == "silu":
            self.activation_fn = F.silu
        else:
            self.activation_fn = F.gelu

    def forward(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: Optional[torch.Tensor] = None,
        linear_k: Optional[object] = None,
        linear_q: Optional[object] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seqlen, hidden_size]
            arbitrary_func: [batch, 1, n_func, seqlen+256] int32 mask encoding.
            linear_k: forward block sparsity (Q2K).
            linear_q: backward block sparsity (K2Q).

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
        """
        # ---- Attention block ----
        residual = hidden_states
        x = self.input_layernorm(hidden_states)

        # QKV projection: [B, S, H] → [B, S, 3*H]
        qkv = self.linear_qkv(x)
        # Reshape to [B, S, 3, num_heads, head_dim] and unbind
        B, S, _ = qkv.shape
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, S, num_heads, head_dim]

        # Flash Attention requires fp16/bf16 inputs
        from flash_attn.cute.interface import flash_attn_func

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        if arbitrary_func is not None:
            attn_out, _ = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=self.head_dim ** (-0.5),
                causal=False,
                arbitrary=True,
                linear_k_block_sparse_tensors=linear_k,
                linear_q_block_sparse_tensors=linear_q,
                aux_tensors=[arbitrary_func],
            )
        else:
            attn_out, _ = flash_attn_func(
                q,
                k,
                v,
                softmax_scale=self.head_dim ** (-0.5),
                causal=True,
            )

        if attn_out.dtype != input_dtype:
            attn_out = attn_out.to(input_dtype)

        # attn_out: [B, S, num_heads, head_dim] → [B, S, hidden_size]
        attn_out = attn_out.reshape(B, S, self.hidden_size)
        attn_out = self.linear_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        hidden_states = residual + attn_out

        # ---- FFN block ----
        residual = hidden_states
        x = self.pre_mlp_layernorm(hidden_states)
        x = self.mlp_fc1(x)
        x = self.activation_fn(x)
        x = self.mlp_fc2(x)
        x = self.mlp_dropout(x)
        hidden_states = residual + x

        return hidden_states

    def _qkv_projection(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared pre-attention: LayerNorm → QKV projection.

        Returns:
            residual, q, k, v — each of q/k/v is [..., num_heads, head_dim].
        """
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        qkv = self.linear_qkv(x)
        leading = qkv.shape[:-1]
        qkv = qkv.view(*leading, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=-3)
        return residual, q, k, v

    def _post_attention(
        self, residual: torch.Tensor, attn_out: torch.Tensor
    ) -> torch.Tensor:
        """Shared post-attention: output proj → residual → FFN."""
        leading = attn_out.shape[:-2]
        attn_out = attn_out.reshape(*leading, self.hidden_size)
        attn_out = self.linear_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)
        hidden_states = residual + attn_out

        residual = hidden_states
        x = self.pre_mlp_layernorm(hidden_states)
        x = self.mlp_fc1(x)
        x = self.activation_fn(x)
        x = self.mlp_fc2(x)
        x = self.mlp_dropout(x)
        return residual + x

    def prefill(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: Optional[torch.Tensor] = None,
        linear_k: Optional[object] = None,
        linear_q: Optional[object] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass that also returns the K/V cache for this layer.

        Uses jiayus's flash_attn for attention, consistent with the
        JaggedTransformerBlock training path.

        Args:
            hidden_states: [batch, seqlen, hidden_size]

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
            (k_cache, v_cache): each [batch, seqlen, num_heads, head_dim]
        """
        residual, q, k, v = self._qkv_projection(hidden_states)

        from flash_attn.cute.interface import flash_attn_func

        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()

        k_cache = k.clone()
        v_cache = v.clone()

        if arbitrary_func is not None:
            attn_out, _ = flash_attn_func(
                q, k, v,
                softmax_scale=self.head_dim ** (-0.5),
                causal=False,
                arbitrary=True,
                linear_k_block_sparse_tensors=linear_k,
                linear_q_block_sparse_tensors=linear_q,
                aux_tensors=[arbitrary_func],
            )
        else:
            attn_out, _ = flash_attn_func(
                q, k, v,
                softmax_scale=self.head_dim ** (-0.5),
                causal=True,
            )

        if attn_out.dtype != input_dtype:
            attn_out = attn_out.to(input_dtype)

        hidden_states = self._post_attention(residual, attn_out)
        return hidden_states, (k_cache, v_cache)

    def decode_beam(
        self,
        hidden_states: torch.Tensor,
        k_context: torch.Tensor,
        v_context: torch.Tensor,
        k_beam: Optional[torch.Tensor],
        v_beam: Optional[torch.Tensor],
        topk_indices: torch.Tensor,
        decode_nums: int,
        softmax_scale: Optional[float] = None,
        seqused_k: Optional[torch.Tensor] = None,
        backend: str = "3kernel",
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Decode step using beam_decode_attn kernel.

        Args:
            hidden_states: [batch, beam_width, hidden_size]
            k_context: [batch, seqlen_context, num_heads, head_dim]
            v_context: [batch, seqlen_context, num_heads, head_dim]
            k_beam: [batch, prev_decode_nums * beam_width, num_heads, head_dim]
                or None if no previous decode steps.
            v_beam: same shape as k_beam, or None.
            topk_indices: [batch, 1, num_heads, decode_nums, beam_width] int32
            decode_nums: number of decode steps in beam KV (including self).
            seqused_k: [batch] int32 valid context length per sample, or None.
            backend: "3kernel" (default) or "dsl" (fused). The fused path is
                currently unsafe across decode steps with varying decode_nums
                due to a kernel cache key bug; "3kernel" is the reliable
                default.

        Returns:
            hidden_states: [batch, beam_width, hidden_size]
            (k_new, v_new): each [batch, beam_width, num_heads, head_dim]
        """
        residual, q, k, v = self._qkv_projection(hidden_states)
        # q, k, v: [B, W, num_heads, head_dim]

        if softmax_scale is None:
            softmax_scale = self.head_dim ** (-0.5)

        B, W = q.shape[0], q.shape[1]
        k_new = k  # [B, W, num_heads, D]
        v_new = v

        # The kernel requires fp16/bf16 for q, k, v. We assume the caller
        # has already converted context_kv and beam_kv to a supported dtype
        # (generate_beam_decode does this once after prefill). We only
        # need to convert q/k_new/v_new if the layer was run in fp32
        # (e.g. unit tests with fp32 weights).
        input_dtype = q.dtype
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.bfloat16()
            k_new = k_new.bfloat16()
            v_new = v_new.bfloat16()
        # Sanity: cached tensors must already be fp16/bf16.
        assert k_context.dtype in (torch.float16, torch.bfloat16), (
            f"k_context must be fp16/bf16, got {k_context.dtype}"
        )

        if k_beam is not None:
            assert k_beam.dtype in (torch.float16, torch.bfloat16), (
                f"k_beam must be fp16/bf16, got {k_beam.dtype}"
            )
            k_beam_full = torch.cat([k_beam, k_new], dim=1)
            v_beam_full = torch.cat([v_beam, v_new], dim=1)
        else:
            k_beam_full = k_new
            v_beam_full = v_new

        # Reshape Q for beam_decode_attn: [B, 1, W, H, D]
        q_5d = q.unsqueeze(1)

        beam_decode_attn = _get_beam_decode_attn()
        # FIXME(kernel): default backend="3kernel" because the fused/dsl
        # path in gr-decode_atten/interface.py has a compile-cache key bug
        # that deadlocks when decode_nums varies across calls. Once that's
        # fixed upstream, "dsl" should be preferred on SM8x/SM90/SM120 for
        # better performance.
        kernel_kwargs = {}
        if seqused_k is not None:
            # seqused_k is part of our local interface.py extension and
            # only valid with backend="3kernel". The PyTorch reference
            # silently ignores it.
            kernel_kwargs["seqused_k"] = seqused_k
        attn_out, _ = beam_decode_attn(
            q_5d,
            k_context,
            v_context,
            k_beam_full,
            v_beam_full,
            topk_indices,
            decode_nums,
            softmax_scale=softmax_scale,
            backend=backend,
            **kernel_kwargs,
        )
        # attn_out: [B, 1, W, H, D] → [B, W, H, D]
        attn_out = attn_out.squeeze(1)

        if attn_out.dtype != input_dtype:
            attn_out = attn_out.to(input_dtype)

        hidden_states = self._post_attention(residual, attn_out)
        return hidden_states, (k_new, v_new)


class JaggedFlashAttnBlock(nn.Module):
    """
    A stack of JaggedGPTLayers — the GPT decoder block using jiayus's
    Flash Attention with arbitrary_func mask encoding.

    This module owns its own weights (not shared with Megatron-Core).
    It is used in place of Megatron's TransformerBlock for inference
    with Method A beam search.

    Usage::

        block = JaggedFlashAttnBlock(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            ffn_hidden_size=1024,
        )
        # padded input: [B, S, D]
        output = block(hidden_states, arbitrary_func=af, seqlen=S)
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_attention_heads
        self.layers = nn.ModuleList(
            [
                JaggedGPTLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    ffn_hidden_size=ffn_hidden_size,
                    layernorm_epsilon=layernorm_epsilon,
                    hidden_dropout=hidden_dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: Optional[torch.Tensor] = None,
        seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seqlen, hidden_size] padded input.
            arbitrary_func: [batch, 1, n_func, seqlen+256] int32 mask tensor.
                If None, uses standard causal attention.
            seqlen: sequence length (used for block sparsity construction).
                If None, inferred from hidden_states.shape[1].

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
        """
        if seqlen is None:
            seqlen = hidden_states.shape[1]

        # Build block sparsity from arbitrary_func (once per forward, shared by all layers)
        linear_k, linear_q = None, None
        if arbitrary_func is not None:
            linear_k, linear_q = build_block_sparsity(
                arbitrary_func, seqlen, seqlen, self.head_dim
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                arbitrary_func=arbitrary_func,
                linear_k=linear_k,
                linear_q=linear_q,
            )

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    def prefill(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: Optional[torch.Tensor] = None,
        seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward through all layers, returning per-layer KV caches.

        Args:
            hidden_states: [batch, seqlen, hidden_size]

        Returns:
            hidden_states: [batch, seqlen, hidden_size]
            kv_caches: list of (k, v) per layer, each [batch, seqlen, H, D]
        """
        if seqlen is None:
            seqlen = hidden_states.shape[1]

        linear_k, linear_q = None, None
        if arbitrary_func is not None:
            linear_k, linear_q = build_block_sparsity(
                arbitrary_func, seqlen, seqlen, self.head_dim
            )

        kv_caches = []
        for layer in self.layers:
            hidden_states, kv = layer.prefill(
                hidden_states,
                arbitrary_func=arbitrary_func,
                linear_k=linear_k,
                linear_q=linear_q,
            )
            kv_caches.append(kv)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, kv_caches

    def decode_beam(
        self,
        hidden_states: torch.Tensor,
        context_kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        beam_kv_caches: List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
        topk_indices: torch.Tensor,
        decode_nums: int,
        seqused_k: Optional[torch.Tensor] = None,
        backend: str = "3kernel",
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Decode one beam search step through all layers.

        Args:
            hidden_states: [batch, beam_width, hidden_size]
            context_kv_caches: per-layer (k, v) from prefill.
            beam_kv_caches: per-layer (k_beam, v_beam) accumulated from
                previous decode steps, or None for each layer if no
                previous steps.
            topk_indices: [B, 1, H, decode_nums, W] int32
            decode_nums: total decode steps including self.
            seqused_k: [B] int32 valid context length per sample, or None.
            backend: forwarded to the kernel; see JaggedGPTLayer.decode_beam.

        Returns:
            hidden_states: [batch, beam_width, hidden_size]
            new_beam_kvs: per-layer (k_new, v_new), each [B, W, H, D]
        """
        new_beam_kvs = []
        for i, layer in enumerate(self.layers):
            k_context, v_context = context_kv_caches[i]
            k_beam = beam_kv_caches[i][0] if beam_kv_caches[i] is not None else None
            v_beam = beam_kv_caches[i][1] if beam_kv_caches[i] is not None else None
            hidden_states, kv_new = layer.decode_beam(
                hidden_states,
                k_context, v_context,
                k_beam, v_beam,
                topk_indices, decode_nums,
                seqused_k=seqused_k,
                backend=backend,
            )
            new_beam_kvs.append(kv_new)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, new_beam_kvs


class JaggedTransformerBlock(nn.Module):
    """
    Wrapper that accepts jagged (variable-length) hidden states and a
    pre-built arbitrary_func tensor in the flattened (B=1) coordinate space.

    All batch sequences are concatenated into a single sequence of length
    *total_tokens* (no padding).  The arbitrary_func encodes both the
    block-diagonal batch isolation and the desired attention pattern
    (causal, target-grouped, etc.).

    Internally:
      1. Reshape jagged [total_tokens, D] → [1, total_tokens, D]
      2. Forward through JaggedFlashAttnBlock (FA with arbitrary mask)
      3. Reshape [1, total_tokens, D] → [total_tokens, D]

    This is intended to replace Megatron-Core's TransformerBlock in
    SIDGRDecoder.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        ffn_hidden_size: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.block = JaggedFlashAttnBlock(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            ffn_hidden_size=ffn_hidden_size,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            activation=activation,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        arbitrary_func: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: jagged [total_tokens, hidden_size].
            arbitrary_func: [1, 1, n_func, total_tokens + pad] int32 tensor
                in flattened (B=1) coordinate space, encoding both batch
                isolation and the attention pattern.

        Returns:
            jagged output [total_tokens, hidden_size].
        """
        total_tokens = hidden_states.shape[0]

        # [total_tokens, D] → [1, total_tokens, D]
        flat_input = hidden_states.unsqueeze(0)

        output = self.block(
            flat_input, arbitrary_func=arbitrary_func, seqlen=total_tokens
        )

        # [1, total_tokens, D] → [total_tokens, D]
        return output.squeeze(0)
