# HSTU

HSTU is a high-performance attention implementation for different NVIDIA GPU architectures:
- **HSTU-2**: Runs on Ampere and Ada GPUs (A100, RTX 3090, RTX 4090)
- **HSTU-3**: Optimized for Hopper GPUs (H100, H20)

## Installation

### HSTU-2
```sh
make install
```

### HSTU-3
```sh
cd hopper/
make install
```

## Features

### HSTU-2
- **Source**: `csrc/hstu_attn/src`
- **Interface**: `hstu_attn/hstu_attn_interface.py`
- **Supported GPUs**: Ampere, Ada, Hopper (without Hopper-specific features)
- **Data types**: FP16, BF16
- **Head dimensions**: 32, 64, 128, 256
- **Attention masks**: No mask, local mask, group target mask, contextual mask, causal mask

### HSTU-3
- **Source**: `hopper/`
- **Interface**: `hopper/flash_attn_interface.py`
- **Supported GPUs**: Hopper only (H100, H20)
- **Data types**: FP16, BF16 (forward and backward), FP8 (forward only)
- **Head dimensions**: 32, 64, 128, 256 for FP16/BF16; 64, 128, 256 for FP8
- **Attention masks**: No mask, local mask, causal mask, target mask (not supported in FP8)
- **Note**: Only undeterministic backward implementation

## Unified Interface

Both HSTU-2 and HSTU-3 share the same function signature:

```python
def hstu_attn_varlen_func(
    q,                      # (total_q, nheads, headdim)
    k,                      # (total_k, nheads_k, headdim), nheads should be divisible by nhead_k
    v,                      # (total_k, nheads_k, headdim)
    seq_offsets_q,          # (batch_size + 1,), cumulative sequence lengths for q
    seq_offsets_k,          # (batch_size + 1,), cumulative sequence lengths for k/v
    max_seqlen_q,           # Maximum query sequence length
    max_seqlen_k,           # Maximum k/v sequence length
    num_contexts=None,      # (batch_size,), context tokens per batch
    num_targets=None,       # (batch_size,), target tokens per batch
    target_group_size=1,    # Number of target tokens per group
    window_size=(-1, -1),   # (left, right) for sliding window; (-1, 0) for causal
    alpha=1.0,              # Scaling factor between add rab and silu
    rab=None,               # (batch_size, nhead_rab, max_seqlen_k, max_seqlen_k), relative attention bias
                            # nheads should be divisible by nhead_rab
    has_drab=False,         # Whether to apply drab
    is_delta_q=False,       # Whether to apply delta_q
    descale_q=None,         # (1,), descaling factor for query
    descale_k=None,         # (1,), descaling factor for key
    descale_v=None,         # (1,), descaling factor for value
    descale_do=None,        # (1,), descaling factor for do
)
```

See Makefiles for customization compiling options (dimensions, data types, functions).