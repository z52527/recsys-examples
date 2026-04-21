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
Tests for beam_decode_attn kernel integration:
  1. BeamSearch.build_beam_topk_indices correctness.
  2. JaggedGPTLayer prefill/decode_beam smoke test.
  3. JaggedFlashAttnBlock prefill + decode_beam pipeline.
"""

import os
import sys

import pytest
import torch

from beam_search.beam_search import BeamSearch

# Import jagged_flash_attn_block directly to avoid model/__init__.py
# which pulls in heavy dependencies (dynamicemb, megatron, torchrec).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from jagged_flash_attn_block import JaggedFlashAttnBlock, JaggedGPTLayer

sys.path.pop(0)


# ---------------------------------------------------------------------------
# Test: BeamSearch.build_beam_topk_indices
# ---------------------------------------------------------------------------
class TestBuildBeamTopkIndices:
    def test_step0_self_only(self):
        """At decode step 0, topk should be self-pointers: [0..W-1]."""
        W = 4
        bs = BeamSearch(W, 3, [10, 10, 10], record_history=True)
        log_probs = torch.randn(2, 1, 10, device="cuda")
        bs.propagate(log_probs)

        topk = bs.build_beam_topk_indices(decode_step=0, num_heads=4)
        assert topk.shape == (2, 1, 4, 1, W)
        # At step 0, each beam w should point to 0*W + w = w
        for w in range(W):
            assert (topk[:, 0, :, 0, w] == w).all()

    def test_step1_parent_traced(self):
        """At decode step 1, ancestors at step 0 should match parent_indices[1]."""
        W = 3
        B = 2
        bs = BeamSearch(W, 3, [10, 10, 10], record_history=True)

        # Step 0
        log_probs0 = torch.randn(B, 1, 10, device="cuda")
        bs.propagate(log_probs0)

        # Step 1
        log_probs1 = torch.randn(B, W, 10, device="cuda")
        bs.propagate(log_probs1)

        topk = bs.build_beam_topk_indices(decode_step=1, num_heads=4)
        assert topk.shape == (B, 1, 4, 2, W)

        for b in range(B):
            for w in range(W):
                # Self at step 1
                assert topk[b, 0, 0, 1, w].item() == 1 * W + w
                # Ancestor at step 0
                parent = bs.parent_indices[1][b, w].item()
                assert topk[b, 0, 0, 0, w].item() == 0 * W + parent

    @pytest.mark.parametrize("num_hierarchies", [2, 3, 4])
    def test_shape_and_range(self, num_hierarchies):
        """Verify topk_indices shape and value range."""
        W = 3
        B = 2
        num_heads = 4
        bs = BeamSearch(W, num_hierarchies, [10] * num_hierarchies, record_history=True)

        topk_prev = 1
        for s in range(num_hierarchies):
            log_probs = torch.randn(B, topk_prev, 10, device="cuda")
            bs.propagate(log_probs)
            topk_prev = W

        for d in range(num_hierarchies):
            topk = bs.build_beam_topk_indices(decode_step=d, num_heads=num_heads)
            decode_nums = d + 1
            assert topk.shape == (B, 1, num_heads, decode_nums, W)
            # All indices should be in [0, decode_nums * W)
            assert (topk >= 0).all()
            assert (topk < decode_nums * W).all()
            # Self at last position should be d*W + w
            for w in range(W):
                assert (topk[:, 0, :, d, w] == d * W + w).all()


# ---------------------------------------------------------------------------
# Test: BeamSearch parent_indices bug fix
# ---------------------------------------------------------------------------
class TestBeamSearchParentIndices:
    def test_parent_indices_stored(self):
        """parent_indices should be recorded at each propagate step."""
        bs = BeamSearch(2, 3, [10, 10, 10])
        topk_prev = 1
        for _ in range(3):
            log_probs = torch.randn(1, topk_prev, 10, device="cuda")
            bs.propagate(log_probs)
            topk_prev = 2
        assert len(bs.parent_indices) == 3

    def test_reset_clears_parent_indices(self):
        bs = BeamSearch(2, 2, [10, 10])
        topk_prev = 1
        for _ in range(2):
            log_probs = torch.randn(1, topk_prev, 10, device="cuda")
            bs.propagate(log_probs)
            topk_prev = 2
        bs.reset()
        assert len(bs.parent_indices) == 0

    def test_beam_width_list_bug_fixed(self):
        """beam_width as list should not raise."""
        bs = BeamSearch([2, 3, 4], 3, [10, 10, 10])
        assert bs.beam_widths == [2, 3, 4]


# ---------------------------------------------------------------------------
# Test: JaggedGPTLayer prefill & decode_beam (smoke)
# ---------------------------------------------------------------------------
class TestJaggedGPTLayerPrefillDecode:
    @pytest.fixture
    def layer(self):
        return JaggedGPTLayer(
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
        ).cuda().bfloat16()

    def test_prefill_returns_kv_cache(self, layer):
        B, S, D = 2, 16, 64
        x = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        out, (k_cache, v_cache) = layer.prefill(x)
        assert out.shape == (B, S, D)
        assert k_cache.shape == (B, S, 4, 16)  # [B, S, H, D_head]
        assert v_cache.shape == k_cache.shape

    def test_decode_beam_smoke(self, layer):
        """decode_beam should run without errors (requires beam_decode_attn)."""
        # Uses reference fallback if CuTe kernel not available

        B, W, S_ctx, H, D_head = 2, 3, 16, 4, 16
        D = H * D_head

        x = torch.randn(B, W, D, device="cuda", dtype=torch.bfloat16)
        k_ctx = torch.randn(B, S_ctx, H, D_head, device="cuda", dtype=torch.bfloat16)
        v_ctx = torch.randn(B, S_ctx, H, D_head, device="cuda", dtype=torch.bfloat16)

        # First decode step: no previous beam KV
        topk = torch.arange(W, device="cuda").view(1, 1, 1, 1, W).expand(B, 1, H, 1, W).to(torch.int32)

        out, (k_new, v_new) = layer.decode_beam(
            x, k_ctx, v_ctx,
            k_beam=None, v_beam=None,
            topk_indices=topk, decode_nums=1,
        )
        assert out.shape == (B, W, D)
        assert k_new.shape == (B, W, H, D_head)


# ---------------------------------------------------------------------------
# Test: Full JaggedFlashAttnBlock prefill + decode pipeline
# ---------------------------------------------------------------------------
class TestJaggedFlashAttnBlockPipeline:
    @pytest.fixture
    def block(self):
        return JaggedFlashAttnBlock(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
        ).cuda().bfloat16()

    def test_prefill_and_decode(self, block):
        """Prefill + one decode step should run end-to-end."""
        # Uses reference fallback if CuTe kernel not available

        B, S_ctx, W, D = 2, 16, 3, 64
        H, D_head = 4, 16
        num_layers = 2

        # Prefill
        x_prefill = torch.randn(B, S_ctx, D, device="cuda", dtype=torch.bfloat16)
        prefill_out, ctx_kv = block.prefill(x_prefill)
        assert prefill_out.shape == (B, S_ctx, D)
        assert len(ctx_kv) == num_layers

        # Decode step 0
        x_decode = torch.randn(B, W, D, device="cuda", dtype=torch.bfloat16)
        topk = torch.arange(W, device="cuda").view(1, 1, 1, 1, W).expand(B, 1, H, 1, W).to(torch.int32)
        beam_kv = [None] * num_layers

        out, new_kvs = block.decode_beam(
            x_decode, ctx_kv, beam_kv, topk, decode_nums=1,
        )
        assert out.shape == (B, W, D)
        assert len(new_kvs) == num_layers

        # Accumulate beam KV
        for l in range(num_layers):
            beam_kv[l] = new_kvs[l]

        # Decode step 1
        x_decode2 = torch.randn(B, W, D, device="cuda", dtype=torch.bfloat16)
        # topk for step 1: self + ancestor at step 0
        topk2 = torch.zeros(B, 1, H, 2, W, device="cuda", dtype=torch.int32)
        for w in range(W):
            topk2[:, 0, :, 0, w] = w  # ancestor at step 0 (self, for simplicity)
            topk2[:, 0, :, 1, w] = W + w  # self at step 1

        out2, new_kvs2 = block.decode_beam(
            x_decode2, ctx_kv, beam_kv, topk2, decode_nums=2,
        )
        assert out2.shape == (B, W, D)
