# Bug report: `beam_decode_attn(backend="dsl")` hangs when `beam_width % 8 != 0` on SM90

## Status: RESOLVED

Fixed upstream in `gr-decode_atten` commit `9f1c2a9` ("fix: w is not divided by 4", 2026-05-07). Verified end-to-end on H100 NVL (SM90) by re-running the W sweep below: every value in `{1, 2, 4, 8, 9, 10, 11, 12, 14, 15, 16, 17, 20, 24, 32, 40, 48, 64, 128}` returns in <2 s. This document is preserved as a historical record of the bug we filed; the workaround it describes (forcing `backend="3kernel"`) is no longer required on the kernel side, though our model path still defaults to `3kernel` because the fused path silently ignores `seqused_k` (a separate, by-design limitation, not the hang).

## TL;DR

The fused / `dsl` path of `beam_decode_attn` hangs (GPU at 100% util, kernel never returns) on H100 PCIe (SM90) when the beam width `W` is not a multiple of 8. The 3-kernel pipeline path (`backend="3kernel"`) works correctly for all `W` values we tested.

## Environment

| | |
|---|---|
| GPU | NVIDIA H100 PCIe (114 SMs), compute capability (9, 0) |
| Container | ` gitlab-master.nvidia.com/devtech-compute/distributed-recommender:devel_latest` based |
| `torch` | 2.11.0a0+eb65b36914.nv26.02 |
| `nvidia-cutlass-dsl` | 4.4.0 |
| `quack-kernels` | 0.4.1 |
| `flash-attn-cute` | 0.1.0 |
| `gr-decode_atten` | clone at `master` |

`make tt` (`tests/test_fwd.py`) passes for both `3kernel` and `dsl` paths because all those tests use `W ∈ {128, 256, 512, 1024}` — all multiples of 8.

## Reproducer

```python
# probe_bw.py
import sys, time, torch
from interface import beam_decode_attn

W = int(sys.argv[1])
B, H, D = 1, 4, 64
Sk = 256
torch.manual_seed(7)
q     = torch.randn(B, 1, W, H, D, device='cuda', dtype=torch.bfloat16)
k_ctx = torch.randn(B, Sk, H, D,    device='cuda', dtype=torch.bfloat16)
v_ctx = torch.randn(B, Sk, H, D,    device='cuda', dtype=torch.bfloat16)
k_bm  = torch.randn(B, W, H, D,     device='cuda', dtype=torch.bfloat16)
v_bm  = torch.randn(B, W, H, D,     device='cuda', dtype=torch.bfloat16)
topk  = torch.zeros(B, 1, H, 1, W, device='cuda', dtype=torch.int32)
for w in range(W):
    topk[..., 0, w] = w  # self-pointers

t0 = time.time()
out, _ = beam_decode_attn(q, k_ctx, v_ctx, k_bm, v_bm, topk, 1, backend='dsl')
torch.cuda.synchronize()
print(f"W={W}: done in {time.time()-t0:.2f}s")
```

Run with various `W` values:

```bash
for W in 1 2 4 8 9 10 11 12 14 15 16 17 20 24 32 40 48 64 128; do
  timeout 30 python probe_bw.py $W 2>&1 | tail -1
done
```

## Observed pattern

```
W=1   Terminated (timeout)
W=2   Terminated (timeout)
W=4   Terminated (timeout)
W=8   done in 2.55s         ← OK
W=9   Terminated (timeout)
W=10  Terminated (timeout)
W=11  Terminated (timeout)
W=12  Terminated (timeout)
W=14  Terminated (timeout)
W=15  Terminated (timeout)
W=16  done in 2.57s         ← OK
W=17  Terminated (timeout)
W=20  Terminated (timeout)
W=24  done in 2.59s         ← OK
W=32  done in 2.58s         ← OK
W=40  done in 2.62s         ← OK
W=48  done in 2.64s         ← OK
W=64  done in 2.58s         ← OK
W=128 done in 2.67s         ← OK
```

**Hang ⇔ `W % 8 != 0`**, exact match.

For hung runs: GPU utilization is 100% and `torch.cuda.synchronize()` never returns. py-spy shows the Python thread parked in `torch.cuda.synchronize`, i.e. the kernel was launched (or partially launched) but its CUDA work never completes.

## Why not just `3kernel`?

Confirmed working — same parameters with `backend="3kernel"` returns in <10 s for all `W` values, including `W=10` which is our SID-GR default.

## Suspected root cause

The fused kernel template likely makes a `W` divisibility assumption (block-tile size = 8?) that isn't validated at runtime. When `W % 8 != 0` the kernel may go into an infinite waiting loop — possibly a barrier mis-count or a `cluster_arrive` style sync that never completes because some warps execute a different code path on the partial tile.

If you can point at where in `_fused_context_beam` (or the SM90 backend it dispatches to) the assumption lives, a clean fix would be either:
1. assert `W % 8 == 0` at the Python entry of the fused path and fall back to `3kernel` otherwise, or
2. make the fused kernel actually handle non-multiple-of-8 `W` (probably what most callers expect).

## Workaround in our code (historical — bug is now fixed)

While this bug was open we hard-coded `backend="3kernel"` in `SIDGRModel.generate_beam_decode`. After the upstream fix (`9f1c2a9`) we relaxed that to a runtime-checked option: `backend="3kernel"` remains the default (it's the only backend that supports our local `seqused_k` extension), but `backend="dsl"` is now selectable when the caller guarantees uniform-length history.
