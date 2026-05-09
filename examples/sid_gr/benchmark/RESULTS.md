# Benchmark results: `generate()` vs `generate_beam_decode()`

Hardware: **NVIDIA H100 PCIe** (114 SMs).
Container: recsys-examples Docker (torch 2.11.0a0+nv26, CUDA 13).
Branch: `fea-mask-beam-search` (SID-GR + Jerry's `beam_decode_attn` kernel).
Date: 2026-05-09 (re-run with jagged-native option and phase breakdown).

## ⚠️ Correctness preconditions

These speedup numbers compare the **corrected** `generate()` baseline against
`generate_beam_decode()`. Both paths now implement beam-isolated attention:
- `generate()` builds `padded_target_aware_causal_mask` per step and feeds
  it through the jagged FA path (the `528cf77` regression has been fixed).
- `generate_beam_decode()` uses Jerry's `beam_decode_attn` with
  `topk_indices` for the same isolation.

The speedup is therefore an apples-to-apples comparison of two
mathematically equivalent implementations — not a comparison of "correct
new path vs broken baseline".

**Required local kernel patches** (in our `gr-decode_atten/interface.py`
clone, not yet upstream):
1. K1 accepts a `seqused_k` kwarg for variable-length history padding.
2. `BeamDecodeAttn.forward` forces `num_splits=1` when `seqused_k` is set
   (workaround for split-KV + seqused_k hang).

If you re-install `quack-kernels` from PyPI without re-cloning
`gr-decode_atten`, the patches stay (they live in the local clone). If
you re-clone `gr-decode_atten` from upstream, the patches must be
re-applied.

## Setup

Both paths share the same model weights (`use_jagged_flash_attn=True`).
- **Original (`generate()`)** — re-runs the full transformer over
  `[history + all_generated]` per hierarchy step, with a beam-isolating
  arbitrary mask (built directly via `build_jagged_target_aware_arbitrary_func`).
- **New (`generate_beam_decode()`)** — prefill once → context KV cache;
  per-step decode reuses cached context KV and accumulates beam KV via
  `topk_indices` (Jerry's CuTe kernel, 3-kernel backend).

Median latency (ms) over 10 iterations after 3 warmup runs;
`cuda.synchronize()` before/after each iteration.

## Summary

| Metric | Value |
|---|---|
| Configs tested | 24 (4 hist_len × 3 beam_w × 2 dtypes) |
| Speedup range | 1.31× – 1.43× |
| Speedup median | **1.38×** |
| Phase split | prefill ≈ 3.0 ms, decode_loop ≈ 5.7 ms |

`generate_beam_decode` is consistently faster across all configurations
and both dtypes. Speedup sits tightly around 1.35–1.40× — the workload
is dominated by the FFN/MLP and embedding-table lookup that both paths
share; the attention savings show up but don't dominate.

## Sweep results

Fixed: `batch=4, hierarchies=3, hidden=256, heads=4, kv_channels=64, layers=2, codebook=256`.

| dtype | hist_len | beam_w | generate (ms) | decode total (ms) | prefill (ms) | decode_loop (ms) | speedup |
|-------|---------:|-------:|--------------:|------------------:|-------------:|-----------------:|--------:|
|  bf16 |       32 |      4 |         11.71 |              8.93 |         3.12 |             5.76 |   1.31× |
|  bf16 |       32 |     10 |         12.11 |              8.76 |         3.05 |             5.68 |   1.38× |
|  bf16 |       32 |     20 |         12.13 |              8.84 |         3.08 |             5.70 |   1.37× |
|  bf16 |       64 |      4 |         12.25 |              8.76 |         3.07 |             5.75 |   1.40× |
|  bf16 |       64 |     10 |         12.16 |              8.77 |         3.02 |             5.68 |   1.39× |
|  bf16 |       64 |     20 |         12.08 |              8.89 |         3.06 |             5.75 |   1.36× |
|  bf16 |      128 |      4 |         11.97 |              8.64 |         2.99 |             5.61 |   1.39× |
|  bf16 |      128 |     10 |         12.02 |              8.62 |         3.02 |             5.66 |   1.39× |
|  bf16 |      128 |     20 |         12.01 |              8.67 |         2.99 |             5.60 |   1.39× |
|  bf16 |      256 |      4 |         12.07 |              8.52 |         2.98 |             5.58 |   1.42× |
|  bf16 |      256 |     10 |         11.97 |              8.65 |         3.03 |             5.71 |   1.38× |
|  bf16 |      256 |     20 |         11.95 |              8.85 |         3.03 |             5.69 |   1.35× |
|  fp16 |       32 |      4 |         12.05 |              8.75 |         3.00 |             5.68 |   1.38× |
|  fp16 |       32 |     10 |         12.02 |              8.80 |         3.02 |             5.70 |   1.37× |
|  fp16 |       32 |     20 |         11.62 |              8.53 |         2.97 |             5.65 |   1.36× |
|  fp16 |       64 |      4 |         12.10 |              8.75 |         2.98 |             5.74 |   1.38× |
|  fp16 |       64 |     10 |         11.96 |              8.52 |         2.93 |             5.57 |   1.40× |
|  fp16 |       64 |     20 |         11.79 |              8.24 |         2.92 |             5.59 |   1.43× |
|  fp16 |      128 |      4 |         11.78 |              8.25 |         2.99 |             5.70 |   1.43× |
|  fp16 |      128 |     10 |         11.82 |              8.75 |         2.97 |             5.60 |   1.35× |
|  fp16 |      128 |     20 |         11.88 |              8.64 |         2.97 |             5.65 |   1.37× |
|  fp16 |      256 |      4 |         11.84 |              8.82 |         2.94 |             5.61 |   1.34× |
|  fp16 |      256 |     10 |         11.85 |              8.55 |         2.99 |             5.68 |   1.38× |
|  fp16 |      256 |     20 |         11.78 |              8.59 |         2.91 |             5.60 |   1.37× |

## Observations

### Speedup is workload-stable

The speedup ratio sits tightly in 1.31–1.43× across:
- 4 history lengths (32, 64, 128, 256 — note this counts items; the
  actual transformer seqlen is `3 × hist_len + 1` because each item is
  3 SIDs and we add a BOS).
- 3 beam widths (4, 10, 20).
- 2 dtypes (bf16, fp16).

Earlier expectations (longer hist → more savings; bigger beam → more
savings) don't show up strongly — the dominant cost in this small model
is **FFN/MLP and embedding lookup**, not attention.

### Phase breakdown

- **Prefill**: ~3 ms regardless of config. This is the cost of running
  history+BOS through the transformer once, including jiayus FA + MLP/FFN
  + KV-cache materialisation.
- **Decode loop**: ~5.7 ms regardless of config. This is the cost of
  `(num_hierarchies − 1)` decode iterations, each containing: KJT lookup,
  layer-stack of `beam_decode_attn` calls, MLP, log_softmax,
  `beam_search.propagate`.

Both phases scale poorly with hist_len/beam_w because per-iteration
overhead (KJT roundtrip, MLP) dominates over the kernel time.

### Where the speedup comes from

The original `generate()` re-runs the full transformer over
`[history + all_generated]` at every hierarchy step, with a dense
arbitrary mask to isolate beams. For `num_hierarchies=3`, that's 3 forward
passes over a sequence growing each step.

`generate_beam_decode()` does:
1. **One** prefill over `[history + BOS]`.
2. **`num_hierarchies − 1`** lightweight decode steps, each processing
   only `beam_width` tokens (per layer) via the sparse `beam_decode_attn`
   kernel.

The savings come from not re-attending to the full history each step.

### dtype: bf16 vs fp16

No meaningful difference in either path — the H100 hardware is equally
happy with both for tensor-core ops. We test fp16 mainly to verify
correctness; numeric range is comparable for these SID-GR sizes.

### Backend choice: 3kernel vs dsl (kernel-level)

The model path defaults to `backend="3kernel"` because it's the only
backend that supports `seqused_k` (variable-length history). For
reference, a kernel-level micro-benchmark on H100 NVL (SM90, 2026-05)
at SID-GR shapes (B=4, Hq=Hkv=4, D=64, decode_nums=3) shows the fused
("dsl") path is ~1.46–1.49× faster per call across all measured
configs (hist∈{32,64,128,256}, W∈{4,10,20}, bf16/fp16): ~0.111 ms vs
~0.164 ms median.

End-to-end this only saves ~0.15 ms on a ~6.7 ms decode loop (<3 %);
prefill + dense projections dominate. `generate_beam_decode` accepts
`backend="dsl"` but enforces uniform history at runtime (the fused
path silently ignores `seqused_k`). On SM100 (B200) the kernel
internally auto-routes to 3kernel regardless of the `backend` arg.

### Jagged-native vs dense+seqused_k path (`use_jagged_kv` flag)

`generate_beam_decode` supports two ways of feeding the variable-length
context K/V to the K1 launch:

- **Dense + seqused_k** (default, `use_jagged_kv=False`): pad history to
  `[B, Sk_max, D]`, run prefill with FA's `causal=True` fast path,
  pass dense K/V plus `seqused_k` to K1 so pad positions are masked out.
- **Jagged-native** (`use_jagged_kv=True`): concatenate history into a
  flat `[total_tokens, D]` stream, run prefill with FA's `arbitrary=True`
  + a jagged-causal mask, feed jagged `[total_k, H, D]` K/V caches to K1
  with `cu_seqlens_k`. No padding is computed anywhere.

Phase-broken-down measurement on H100 PCIe (bf16, 12 configs, warmup=10
iter=50):

| Phase | Dense (B) | Jagged (C) | Δ (C - B) |
|---|---:|---:|---:|
| Prefill | **2.18 ms** | **2.50 ms** | **+0.33 ms** |
| Decode loop | 4.14 ms | 4.10 ms | -0.04 ms |
| **Total** | **6.28 ms** | **6.60 ms** | **+0.32 ms** |

C/B median = **0.951×** (dense wins by ~5%). Top-1 SIDs match exactly
across both paths; log-prob delta is ~0.012 (bf16 reduction-order noise,
well below the 0.15 regression threshold).

**Why dense wins on these shapes:**

- The K1 attention is only ~5% of decode loop time; saving its pad
  positions (~50% of K1 compute, i.e. ~2.5% of decode) gives ~0.04 ms.
- FA's `causal=True` is a heavily-optimized fast path; the jagged path
  uses `arbitrary=True` which adds `build_block_sparsity` CSR kernels
  (~0.15 ms) and slightly slower per-block indexing (~0.13 ms) plus
  arbitrary_func construction (~0.05 ms).
- Net: prefill overhead +0.33 ms eats up the +0.04 ms K1 saving.

**When jagged would win** (extrapolation, not measured):

| Shape | Δprefill (est) | Δdecode (est) | Net |
|---|---:|---:|---:|
| current bench (hist≤256) | +0.33 ms | -0.04 ms | dense wins ~5% |
| medium (hist=512) | +0.4 ms | -0.2 ms | dense still wins |
| long (hist=2048, active users) | +0.6 ms | -1.0 ms | jagged wins ~5% |
| very long (hist=8192) | +1.0 ms | -4 ms | jagged wins ~15% |

Crossover point is roughly `hist >= 1000` items based on K1 scaling as
O(B·W·Sk·D) while prefill setup overhead is roughly Sk-independent.
This matches active-user history lengths in real SID-GR datasets
(KuaiRand, ml-1m 99-percentile), but typical training/eval truncates
history to under 512, so the default stays `use_jagged_kv=False`.

## Correctness verification

Three tiers of correctness signal, in order of strength:

1. **Reference oracle** (`TestReferenceOracle::test_kernel_vs_reference`,
   12 cases): the CuTe kernel output is compared against a pure-PyTorch
   reference implementation in fp32 using FA-style tolerance
   (`max_diff < 0.05`). This is the strongest mathematical check.

2. **Mask isolation unit tests** (`TestBeamIsolationMask`): direct check
   on `padded_target_aware_causal_mask` geometry — different beam
   regions are mutually invisible, all targets see all history.

3. **End-to-end regression guard**
   (`test_generate_vs_generate_beam_decode_regression_guard`): runs both
   paths with identical model weights and asserts:
   - top-1 SID per sample matches exactly,
   - per-position `|log_prob delta| < 0.15`,
   - top-K beam SID set overlap ≥ 70%.

   This is a regression GUARD, not a mathematical equivalence proof.
   bf16 attention's per-layer rounding plus beam-search's topk decision
   boundary make bit-exact equivalence impossible; the thresholds catch
   significant divergence.

## How to reproduce

```bash
cd examples/sid_gr
PYTHONNOUSERSITE=1 LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
  PYTHONPATH=$REPO/examples:$KERNEL_PATH \
  torchrun --nproc_per_node 1 --master_port 29504 \
  benchmark/benchmark_beam_decode.py \
  --sweep --sweep_hist 32,64,128,256 --sweep_beam 4,10,20 --sweep_dtype bf16,fp16
```

## Known issues

- **Local kernel patches are not upstream** in `quack-kernels`: see the
  preconditions section. Documented in
  `corelib/.../docker_env_setup.md`-style memory.
- **Split-KV + `seqused_k`**: hangs in the K1 kernel; worked around by
  forcing `num_splits=1` when `seqused_k` is set.
- **Non-uniform `beam_widths`**: the kernel asserts uniform widths via
  `k_beam.shape[1] == decode_nums * beam_width`.
  `SIDGRModel.generate_beam_decode` validates uniformity at entry and
  rejects non-uniform lists; `BeamSearch` itself accepts non-uniform
  widths so other consumers can use them. The math in
  `build_beam_topk_indices` is general (cumulative offsets), so the
  BeamSearch side is ready if/when the kernel grows non-uniform support.
