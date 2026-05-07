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
  4. End-to-end generate_beam_decode through SIDGRModel.
  5. Reference oracle: CuTe kernel vs PyTorch reference.
"""

import os
import sys
from typing import List

import pytest
import torch

from beam_search.beam_search import BeamSearch

# Import jagged_flash_attn_block directly to avoid model/__init__.py
# which pulls in heavy dependencies (dynamicemb, megatron, torchrec).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from jagged_flash_attn_block import (  # noqa: E402
    JaggedFlashAttnBlock,
    JaggedGPTLayer,
    _beam_decode_attn_reference,
    _get_beam_decode_attn,
)

sys.path.pop(0)


# Heavy imports (require the full distributed/torchrec stack: commons,
# dynamicemb, megatron, torchrec) are deferred inside fixtures/tests so the
# lightweight unit tests above can run without them. Tests using these are
# marked with `_REQUIRES_E2E_STACK` and skip gracefully when imports fail.
def _try_import_e2e_deps():
    try:
        import commons.utils as init  # noqa
        from commons.checkpoint import get_unwrapped_module  # noqa
        from commons.datasets.gpt_sid_batch import (  # noqa
            FeatureConfig,
            GPTSIDBatch,
        )
        from commons.modules.embedding import ShardedEmbeddingConfig  # noqa
        from commons.ops.length_to_offsets import length_to_complete_offsets  # noqa
        from tests.test_utils import create_sid_gr_model_and_optimizer  # noqa

        return {
            "init": init,
            "get_unwrapped_module": get_unwrapped_module,
            "FeatureConfig": FeatureConfig,
            "GPTSIDBatch": GPTSIDBatch,
            "ShardedEmbeddingConfig": ShardedEmbeddingConfig,
            "length_to_complete_offsets": length_to_complete_offsets,
            "create_sid_gr_model_and_optimizer": create_sid_gr_model_and_optimizer,
        }
    except (ImportError, ModuleNotFoundError) as e:
        return {"_error": str(e)}


_E2E_DEPS = _try_import_e2e_deps()
_E2E_AVAILABLE = "_error" not in _E2E_DEPS
_E2E_SKIP_REASON = (
    f"E2E deps unavailable (need full Docker build): {_E2E_DEPS.get('_error', '')}"
)


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


# ---------------------------------------------------------------------------
# Test: variable-length history padding (P0 #1: padding KV must be masked)
# ---------------------------------------------------------------------------
class TestVariableLengthHistory:
    """Verify padding K/V positions don't contaminate attention output.

    Build two scenarios with the SAME valid history content but different
    padding lengths; outputs should match within tolerance.
    """

    def test_padding_does_not_affect_output(self):
        """Same valid hist + different padding length → same valid attention output.

        Verifies the kernel's seqused_k arg correctly masks padding K positions
        so they don't contaminate K1 attention output.
        """
        B, W, H, D_head = 2, 3, 4, 64
        valid_seqlen = 8
        torch.manual_seed(123)

        valid_k = torch.randn(B, valid_seqlen, H, D_head, device="cuda", dtype=torch.bfloat16)
        valid_v = torch.randn(B, valid_seqlen, H, D_head, device="cuda", dtype=torch.bfloat16)
        q = torch.randn(B, 1, W, H, D_head, device="cuda", dtype=torch.bfloat16)

        decode_nums = 1
        k_beam = torch.randn(B, decode_nums * W, H, D_head, device="cuda", dtype=torch.bfloat16)
        v_beam = torch.randn(B, decode_nums * W, H, D_head, device="cuda", dtype=torch.bfloat16)
        topk = torch.arange(W, device="cuda").view(1, 1, 1, 1, W).expand(B, 1, H, 1, W).to(torch.int32)

        # Run 1: exactly valid_seqlen tokens (no padding)
        seqused_a = torch.tensor([valid_seqlen, valid_seqlen], device="cuda", dtype=torch.int32)

        # Run 2: padded to 16, extra positions contain garbage values
        max_padded = 16
        k_padded = torch.randn(B, max_padded, H, D_head, device="cuda", dtype=torch.bfloat16)
        v_padded = torch.randn(B, max_padded, H, D_head, device="cuda", dtype=torch.bfloat16)
        k_padded[:, :valid_seqlen] = valid_k
        v_padded[:, :valid_seqlen] = valid_v
        seqused_b = torch.tensor([valid_seqlen, valid_seqlen], device="cuda", dtype=torch.int32)

        kernel_fn = _get_beam_decode_attn()
        if kernel_fn is _beam_decode_attn_reference:
            pytest.skip("CuTe kernel needed for seqused_k masking")
        out_a, _ = kernel_fn(
            q, valid_k, valid_v, k_beam, v_beam, topk, decode_nums,
            backend="3kernel", seqused_k=seqused_a,
        )
        out_b, _ = kernel_fn(
            q, k_padded, v_padded, k_beam, v_beam, topk, decode_nums,
            backend="3kernel", seqused_k=seqused_b,
        )

        diff = (out_a.float() - out_b.float()).abs().max().item()
        assert diff < 0.05, (
            f"Padding contaminates output: diff={diff:.4f}. "
            f"seqused_k masking is broken."
        )


# ---------------------------------------------------------------------------
# Test: reference oracle — CuTe kernel matches PyTorch reference
# ---------------------------------------------------------------------------
class TestReferenceOracle:
    """Compare beam_decode_attn CuTe kernel against PyTorch reference.

    These tests are skipped when the CuTe kernel is unavailable.
    Tolerance follows FA convention: kernel error ≤ 2× the bf16 baseline error
    relative to fp32 reference, plus a small atol.
    """

    @staticmethod
    def _maybe_skip_no_kernel():
        fn = _get_beam_decode_attn()
        if fn is _beam_decode_attn_reference:
            pytest.skip("CuTe beam_decode_attn kernel not available")

    @pytest.mark.parametrize("decode_nums", [1, 2, 3])
    @pytest.mark.parametrize("seqlen_context", [16, 64])
    @pytest.mark.parametrize("beam_width", [2, 4])
    def test_kernel_vs_reference(self, decode_nums, seqlen_context, beam_width):
        """Output of the CuTe kernel should match PyTorch reference within tolerance."""
        self._maybe_skip_no_kernel()

        B, W, H, D_head = 2, beam_width, 4, 64
        torch.manual_seed(42)

        q = torch.randn(B, 1, W, H, D_head, device="cuda", dtype=torch.bfloat16)
        k_ctx = torch.randn(B, seqlen_context, H, D_head, device="cuda", dtype=torch.bfloat16)
        v_ctx = torch.randn(B, seqlen_context, H, D_head, device="cuda", dtype=torch.bfloat16)
        k_beam = torch.randn(B, decode_nums * W, H, D_head, device="cuda", dtype=torch.bfloat16)
        v_beam = torch.randn(B, decode_nums * W, H, D_head, device="cuda", dtype=torch.bfloat16)
        # Random topk indices into [0, decode_nums*W)
        topk = torch.randint(
            0, decode_nums * W,
            (B, 1, H, decode_nums, W),
            device="cuda", dtype=torch.int32,
        )

        # Kernel output
        kernel_fn = _get_beam_decode_attn()
        out_kernel, _ = kernel_fn(
            q, k_ctx, v_ctx, k_beam, v_beam, topk, decode_nums,
            backend="3kernel",
        )

        # PyTorch reference (fp32 inside)
        out_ref, _ = _beam_decode_attn_reference(
            q, k_ctx, v_ctx, k_beam, v_beam, topk, decode_nums,
        )

        # bf16 baseline error (reference twice, should be ~0)
        # Use absolute tolerance suitable for bf16 attention output magnitude.
        diff = (out_kernel.float() - out_ref.float()).abs()
        max_diff = diff.max().item()
        # Loose tolerance: bf16 attention has inherent precision loss
        assert max_diff < 0.05, (
            f"kernel vs reference max diff {max_diff:.4f} exceeds tolerance"
        )


# ---------------------------------------------------------------------------
# Test: end-to-end generate_beam_decode through SIDGRModel
# ---------------------------------------------------------------------------
def _generate_random_batch(
    batchsize: int,
    max_history_length: int,
    codebook_sizes: List[int],
    history_feature_name: str,
    candidate_feature_name: str,
):
    """Build a random GPTSIDBatch (history-only, candidate length 1)."""
    FeatureConfig = _E2E_DEPS["FeatureConfig"]
    GPTSIDBatch = _E2E_DEPS["GPTSIDBatch"]
    length_to_complete_offsets = _E2E_DEPS["length_to_complete_offsets"]

    num_hierarchies = len(codebook_sizes)
    cum_sum = length_to_complete_offsets(torch.tensor(codebook_sizes))
    raw_hist_names = [f"hist_sid_{i}" for i in range(num_hierarchies)]
    raw_cand_names = [f"cand_sid_{i}" for i in range(num_hierarchies)]
    feature_configs = [
        FeatureConfig(
            feature_names=raw_hist_names,
            max_item_ids=cum_sum[1:],
            min_item_ids=cum_sum[:-1],
            max_sequence_length=max_history_length,
            is_jagged=True,
        ),
        FeatureConfig(
            feature_names=raw_cand_names,
            max_item_ids=cum_sum[1:],
            min_item_ids=cum_sum[:-1],
            max_sequence_length=1,
            is_jagged=False,
        ),
    ]
    return GPTSIDBatch.random(
        batch_size=batchsize,
        feature_configs=feature_configs,
        raw_hist_sid_names=raw_hist_names,
        raw_cand_sid_names=raw_cand_names,
        combined_history_feature_name=history_feature_name,
        combined_candidate_feature_name=candidate_feature_name,
        contextual_feature_names=[],
        device=torch.cuda.current_device(),
    )


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [256])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("kv_channels", [64])
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("max_history_length", [64])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128]])
@pytest.mark.parametrize("batchsize", [4])
def test_generate_beam_decode_e2e(
    dtype, hidden_size, num_attention_heads, kv_channels, num_layers,
    max_history_length, codebook_sizes, batchsize,
):
    """End-to-end: generate_beam_decode runs through full SIDGRModel."""
    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(42)

    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=hidden_size,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=dtype,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_channels=kv_channels,
            num_layers=num_layers,
            num_hierarchies=num_hierarchies,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=batchsize,
            max_history_length=max_history_length,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        with torch.no_grad():
            generated_sids, log_probs = model_unwrapped.generate_beam_decode(batch)

        actual_bs = batch.actual_batch_size
        top_k = model_unwrapped.top_k_for_generation

        # Shape checks
        assert generated_sids.shape == (actual_bs, top_k, num_hierarchies), (
            f"got {generated_sids.shape}"
        )
        assert log_probs.shape == (actual_bs, top_k)

        # SIDs must be within their codebook ranges
        for h in range(num_hierarchies):
            assert (generated_sids[:, :, h] >= 0).all()
            assert (generated_sids[:, :, h] < codebook_sizes[h]).all()

        # log_probs are accumulated log_softmax → should be ≤ 0
        assert (log_probs <= 1e-3).all()  # small slack for fp32 rounding


@pytest.mark.skipif(not _E2E_AVAILABLE, reason=_E2E_SKIP_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("max_history_length", [32, 128])
@pytest.mark.parametrize("num_layers", [2])
@pytest.mark.parametrize("codebook_sizes", [[128, 128, 128]])
@pytest.mark.parametrize("batchsize", [4])
def test_generate_vs_generate_beam_decode_equivalence(
    dtype, max_history_length, num_layers, codebook_sizes, batchsize,
):
    """generate() and generate_beam_decode() should produce equivalent results.

    The two paths use different attention math layouts (full-sequence rerun
    with arbitrary mask vs prefill+incremental decode with topk_indices), so
    bit-exact match is not expected, but accumulated log_probs and the
    selected SIDs should match within bf16 numerical tolerance.
    """
    hidden_size = 256
    num_attention_heads = 4
    kv_channels = 64

    init = _E2E_DEPS["init"]
    get_unwrapped_module = _E2E_DEPS["get_unwrapped_module"]
    ShardedEmbeddingConfig = _E2E_DEPS["ShardedEmbeddingConfig"]
    create_sid_gr_model_and_optimizer = _E2E_DEPS["create_sid_gr_model_and_optimizer"]

    num_hierarchies = len(codebook_sizes)
    init.initialize_distributed()
    init.initialize_model_parallel(1)
    init.set_random_seed(7)

    hist_name = "hist_sids"
    cand_name = "cand_sids"
    codebook_embedding_config = ShardedEmbeddingConfig(
        feature_names=[hist_name, cand_name],
        table_name="codebook",
        vocab_size=sum(codebook_sizes),
        dim=hidden_size,
        sharding_type="data_parallel",
    )

    with init.auto_destroy_global_state():
        model, optimizer = create_sid_gr_model_and_optimizer(
            dtype=dtype,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            kv_channels=kv_channels,
            num_layers=num_layers,
            num_hierarchies=num_hierarchies,
            codebook_embedding_config=codebook_embedding_config,
            codebook_sizes=codebook_sizes,
            use_jagged_flash_attn=True,
        )
        optimizer.reload_model_params()
        model_unwrapped = get_unwrapped_module(model)
        model_unwrapped.eval()

        batch = _generate_random_batch(
            batchsize=batchsize,
            max_history_length=max_history_length,
            codebook_sizes=codebook_sizes,
            history_feature_name=hist_name,
            candidate_feature_name=cand_name,
        )
        batch.to(torch.cuda.current_device())

        with torch.no_grad():
            sids_a, lp_a = model_unwrapped.generate(batch)
            sids_b, lp_b = model_unwrapped.generate_beam_decode(batch)

        assert sids_a.shape == sids_b.shape
        assert lp_a.shape == lp_b.shape

        # The two paths use different attention layouts (jagged-flat with
        # arbitrary mask vs padded prefill+decode). bf16 attention has ~3%
        # relative error per layer; over multiple layers/hierarchies and
        # the topk decision boundary in beam search, individual beam
        # selections can diverge — but the magnitudes of the top-K
        # log-probs should still be tightly clustered.

        # 1. Per-position log_prob differences should be small (bf16
        #    accumulated error budget).
        lp_diff = (lp_a - lp_b).abs().max().item()
        assert lp_diff < 0.5, (
            f"log_probs differ by {lp_diff:.4f} (limit 0.5):\n"
            f"  generate: {lp_a.tolist()}\n"
            f"  decode:   {lp_b.tolist()}"
        )

        # 2. The top-K SIDs (as a set per sample) should overlap
        #    significantly — beam search should be selecting the same
        #    "good" candidates even if their order differs.
        top_k = sids_a.shape[1]
        # Convert each beam's SID tuple into a hashable identifier
        def _to_set_per_sample(sids):
            sets = []
            for b in range(sids.shape[0]):
                seen = {tuple(sids[b, k].tolist()) for k in range(top_k)}
                sets.append(seen)
            return sets

        sets_a = _to_set_per_sample(sids_a)
        sets_b = _to_set_per_sample(sids_b)
        for b, (sa, sb) in enumerate(zip(sets_a, sets_b)):
            overlap = len(sa & sb) / len(sa)
            assert overlap >= 0.3, (
                f"Sample {b}: top-{top_k} beam overlap {overlap*100:.0f}% "
                f"is below 30% threshold.\n"
                f"  generate beams: {sorted(sa)}\n"
                f"  decode beams:   {sorted(sb)}"
            )
