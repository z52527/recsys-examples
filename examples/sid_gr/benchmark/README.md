# SID-GR Generation Benchmark

Compares the two generation paths in `SIDGRModel`:

| Path | Method | Attention |
|---|---|---|
| Original | `generate()` | Arbitrary-mask FlashAttention over full `[history + all_generated]` per step, with arbitrary mask isolating beams |
| New | `generate_beam_decode()` | Prefill once → KV cache; per-step decode via the CuTe `beam_decode_attn` kernel (pipelined backend) |

## Requirements

Runs inside the recsys-examples Docker container, which provides `commons`,
`dynamicemb`, `megatron`, `torchrec`, and the vendored `beam_decode_attn`
kernel at `corelib/gr_decode_atten/` (auto-added to `PYTHONPATH` by the
Dockerfile — no extra setup needed).

## Usage

### Single config

```bash
cd examples/sid_gr
torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
  --batch_size 4 --max_hist_len 128 --beam_width 10 \
  --num_hierarchies 3 --num_layers 4
```

### Sweep

```bash
cd examples/sid_gr
torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
  --sweep --sweep_hist 32,64,128,256 --sweep_beam 4,10,20 \
  --num_layers 4
```

Output: per-config latency lines plus a markdown table at the end.

### KV-mode comparison (3-way)

To reproduce the dense-vs-jagged numbers in `RESULTS.md` (compares
`generate()` baseline, `generate_beam_decode(use_jagged_kv=False)`, and
`generate_beam_decode(use_jagged_kv=True)` side by side, with phase
breakdown for the latter two):

```bash
cd examples/sid_gr
torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
  --compare_kv_modes --sweep_hist 32,64,128,256 --sweep_beam 4,10,20 \
  --sweep_dtype bf16 --num_warmup 10 --num_iter 50
```

`--use_jagged_kv` and `--compare_kv_modes` rely on the `cu_seqlens_k`
kernel entry point. The vendored kernel at `corelib/gr_decode_atten/`
already includes it; the benchmark also probes for it at runtime and
raises a clear error if the resolved kernel cannot be verified.

## Tunable arguments

| Flag | Default | Description |
|---|---|---|
| `--batch_size` | 4 | |
| `--max_hist_len` | 128 | |
| `--beam_width` | 10 | |
| `--num_hierarchies` | 3 | |
| `--codebook_size` | 256 | per-level codebook size |
| `--hidden_size` | 256 | |
| `--num_heads` | 4 | |
| `--kv_channels` | 64 | per-head dim |
| `--num_layers` | 2 | |
| `--num_warmup` | 5 | warmup iterations (CuTe JIT compile is slow on first call) |
| `--num_iter` | 20 | timed iterations |
| `--backend` | `3kernel` | `beam_decode_attn` backend (`3kernel` or `dsl`) |
| `--use_jagged_kv` | off | jagged-native prefill + `cu_seqlens_k` (requires the cu_seqlens_k kernel patch) |
| `--compare_kv_modes` | off | 3-way sweep (generate / dense / jagged) |
| `--validate_outputs` | off | In `--sweep` mode, add an untimed A-vs-B correctness check per config |
| `--allow_validation_fail` | off | Allow `--compare_kv_modes` / `--validate_outputs` to exit successfully on validation failure (default: raise `RuntimeError`) |
