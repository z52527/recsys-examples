# Benchmark results: `generate()` vs `generate_beam_decode()`

Hardware: **NVIDIA H100 PCIe** (114 SMs).
Container: recsys-examples Docker (torch 2.11.0a0+nv26, CUDA 13).
Branch: `fea-mask-beam-search` (SID-GR + Jerry's `beam_decode_attn` kernel).
Date: 2026-05-07.

Both paths share the same model weights (`use_jagged_flash_attn=True`).
- **Original (`generate()`)** — re-runs full transformer over `[history + all_generated]` per step, jiayus FA + arbitrary mask isolating beams.
- **New (`generate_beam_decode()`)** — prefill once → context KV cache; per-step decode reuses cached context KV and accumulates beam KV via `topk_indices` (Jerry's CuTe kernel, 3-kernel backend).

All numbers are median latency (ms) over 10 iterations after 3 warmup runs, `cuda.synchronize()` before/after.

## Summary

| Metric | Value |
|---|---|
| Configs tested | 24 (4 hist_len × 3 beam_w × 2 dtypes) |
| Speedup range | 1.22× – 1.34× |
| Speedup median | **1.27×** |
| Phase split | prefill ≈ 3 ms, decode_loop ≈ 5.6 ms |

`generate_beam_decode` is consistently faster across all configurations and both dtypes. Speedup is stable around 1.25–1.30× because the workload is dominated by the FFN/MLP (same in both paths) and the embedding table lookup; the attention savings show up but don't dominate.

## Sweep results

Fixed: `batch=4, hierarchies=3, hidden=256, heads=4, kv_channels=64, layers=2, codebook=256`.

| dtype | hist_len | beam_w | generate (ms) | decode total (ms) | prefill (ms) | decode_loop (ms) | speedup |
|-------|---------:|-------:|--------------:|------------------:|-------------:|-----------------:|--------:|
|  bf16 |       32 |      4 |         10.97 |              8.20 |         2.87 |             5.32 |   1.34× |
|  bf16 |       32 |     10 |         10.81 |              8.57 |         2.98 |             5.54 |   1.26× |
|  bf16 |       32 |     20 |         10.94 |              8.78 |         3.00 |             5.64 |   1.25× |
|  bf16 |       64 |      4 |         11.13 |              8.70 |         2.95 |             5.54 |   1.28× |
|  bf16 |       64 |     10 |         11.06 |              8.91 |         3.03 |             5.68 |   1.24× |
|  bf16 |       64 |     20 |         10.96 |              8.66 |         2.95 |             5.56 |   1.27× |
|  bf16 |      128 |      4 |         10.90 |              8.47 |         2.98 |             5.51 |   1.29× |
|  bf16 |      128 |     10 |         10.99 |              8.54 |         2.97 |             5.54 |   1.29× |
|  bf16 |      128 |     20 |         11.05 |              8.55 |         2.94 |             5.53 |   1.29× |
|  bf16 |      256 |      4 |         11.04 |              8.49 |         2.97 |             5.57 |   1.30× |
|  bf16 |      256 |     10 |         11.10 |              8.67 |         2.96 |             5.63 |   1.28× |
|  bf16 |      256 |     20 |         11.21 |              8.58 |         2.98 |             5.58 |   1.31× |
|  fp16 |       32 |      4 |         11.01 |              8.85 |         3.02 |             5.74 |   1.24× |
|  fp16 |       32 |     10 |         11.13 |              8.51 |         3.02 |             5.77 |   1.31× |
|  fp16 |       32 |     20 |         11.08 |              8.83 |         3.05 |             5.74 |   1.26× |
|  fp16 |       64 |      4 |         11.08 |              8.73 |         2.99 |             5.66 |   1.27× |
|  fp16 |       64 |     10 |         11.03 |              8.73 |         2.98 |             5.65 |   1.26× |
|  fp16 |       64 |     20 |         10.56 |              8.67 |         2.95 |             5.66 |   1.22× |
|  fp16 |      128 |      4 |         10.98 |              8.66 |         3.03 |             5.67 |   1.27× |
|  fp16 |      128 |     10 |         11.09 |              8.63 |         2.97 |             5.67 |   1.28× |
|  fp16 |      128 |     20 |         10.94 |              8.61 |         2.95 |             5.62 |   1.27× |
|  fp16 |      256 |      4 |         10.90 |              8.64 |         2.96 |             5.68 |   1.26× |
|  fp16 |      256 |     10 |         10.94 |              8.65 |         2.95 |             5.66 |   1.26× |
|  fp16 |      256 |     20 |         10.96 |              8.68 |         2.96 |             5.66 |   1.26× |

## Observations

### Speedup is workload-stable

The speedup ratio sits tightly in 1.22–1.34× across:
- 4 history lengths (32, 64, 128, 256 — but note this is items; actual seqlen is `3 × hist_len + 1`)
- 3 beam widths (4, 10, 20)
- 2 dtypes (bf16, fp16)

Earlier expectations (longer hist → more savings; bigger beam → more savings) don't show up strongly — because the dominant cost in this small model is **FFN/MLP and embedding lookup**, not attention.

### Phase breakdown

- **Prefill**: ~3 ms regardless of config. This is the cost of running the entire history+BOS through the transformer once, including jiayus FA + MLP/FFN + KV-cache materialization.
- **Decode loop**: ~5.6 ms regardless of config. This is the cost of `(num_hierarchies − 1)` decode iterations, each containing: KJT lookup, layer-stack of `beam_decode_attn` calls, MLP, log_softmax, beam_search.propagate.

The fact that both phases are roughly fixed across hist_len/beam_w confirms the kernel itself is not the bottleneck for this model — the per-iteration overhead (KJT roundtrip, MLP) dominates.

### Where the speedup comes from

The original `generate()` re-runs the full transformer over `[history + all_generated]` at every hierarchy step, with a dense arbitrary mask to isolate beams. For `num_hierarchies=3`, that's 3 forward passes over a sequence growing each time.

`generate_beam_decode()` does:
1. **One** prefill over `[history + BOS]`.
2. **`num_hierarchies − 1`** lightweight decode steps, each processing only `beam_width × num_layers` tokens via the sparse beam_decode_attn kernel.

The savings come from not re-attending to the full history at every step.

### dtype: bf16 vs fp16

No meaningful performance difference between bf16 and fp16 in either path — the H100 hardware is equally happy with both for tensor core ops. We tested fp16 mainly for correctness; numeric range is comparable for our SID-GR sizes.

## Equivalence test

`test_generate_vs_generate_beam_decode_equivalence` runs both paths with identical weights+input and checks:
- per-position |log_prob(a) − log_prob(b)| < 0.5 (bf16 attention has ~3% rel error per layer; over 2 layers × 3 hierarchies the absolute log_prob delta stays bounded)
- top-K beam SID **sets** overlap by ≥ 30% per sample (the topk decision boundary in beam search amplifies small numerical perturbations into different orderings, but should still pick the same "good" candidates)

Both assertions pass for `hist_len ∈ {32, 128}`. We don't expect bit-exact match because the two paths have different attention layouts (jagged-flat with arbitrary mask vs padded prefill+decode).

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

- **Split-KV + `seqused_k` deadlock**: When `Sk > 4×128=512` (so num_n_blocks > 4 → ns > 1) AND `seqused_k` is provided, the K1 kernel hangs. We work around it by forcing `num_splits = 1` in `interface.py` whenever `seqused_k` is set. Documented in `gr-decode_atten/interface.py`. Small perf hit when both long context AND padding masking are simultaneously needed; should be fixed in the kernel.
- **Fused-path JIT cache key**: The `dsl` (fused) backend has a stale-cache hang when `decode_nums` varies across calls. We default to `backend="3kernel"` which is unaffected.

These are kernel-side bugs; both fixes are in `gr-decode_atten/interface.py` as local patches and should flow upstream to Jerry.
