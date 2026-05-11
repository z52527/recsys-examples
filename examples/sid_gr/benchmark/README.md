# SID-GR Generation Benchmark

Compares the two generation paths in `SIDGRModel`:

| Path | Method | Attention |
|---|---|---|
| Original | `generate()` | jiayus FA over full `[history + all_generated]` per step, with arbitrary mask isolating beams |
| New | `generate_beam_decode()` | Prefill once → KV cache; per-step decode via the CuTe `beam_decode_attn` kernel (pipelined backend) |

## Requirements

Runs inside the recsys-examples Docker container (needs `commons` + `dynamicemb` + `megatron` + `torchrec` to import). The CuTe kernel must be importable:

```bash
pip install quack-kernels>=0.3.3 nvidia-cutlass-dsl==4.4.1 apache-tvm-ffi
```

`gr-decode_atten` must be on `PYTHONPATH`.

## Usage

### Single config

```bash
cd examples/sid_gr
PYTHONPATH=/path/to/gr-decode_atten:$PYTHONPATH \
  torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
  --batch_size 4 --max_hist_len 128 --beam_width 10 \
  --num_hierarchies 3 --num_layers 4
```

### Sweep

```bash
cd examples/sid_gr
PYTHONPATH=/path/to/gr-decode_atten:$PYTHONPATH \
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
PYTHONPATH=/path/to/gr-decode_atten:$PYTHONPATH \
  torchrun --nproc_per_node 1 benchmark/benchmark_beam_decode.py \
  --compare_kv_modes --sweep_hist 32,64,128,256 --sweep_beam 4,10,20 \
  --sweep_dtype bf16 --num_warmup 10 --num_iter 50
```

The `--use_jagged_kv` mode requires the `cu_seqlens_k` patch in
`gr-decode_atten/interface.py`. The benchmark probes for it at
runtime and raises a clear error otherwise.

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
