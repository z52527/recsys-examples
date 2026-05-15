import torch


@torch.library.register_fake("hstu_cuda_ops::split_by_lengths")
def _split_by_lengths_fake(values, lengths_1d, num_splits: int):
    torch._check(values.dim() == 1 or values.dim() == 2)
    torch._check(lengths_1d.dim() == 1)

    # Tensor[] return length is static (num_splits), which is export-friendly.
    # Each output tensor length along dim 0 is data-dependent; represent with symbolic sizes.
    # Keep trailing dimensions the same as `values`.
    ctx = torch.library.get_ctx()
    out = []
    for _ in range(num_splits):
        dyn_len = ctx.new_dynamic_size()
        if values.dim() == 1:
            out.append(values.new_empty((dyn_len,)))
        else:
            out.append(values.new_empty((dyn_len, values.size(1))))
    return out


@torch.library.register_fake("hstu_cuda_ops::lengths_splits")
def _lengths_splits_fake(lengths_1d, num_splits: int):
    torch._check(lengths_1d.dim() == 1)
    torch._check(num_splits > 0)

    # num_splits is static at export time.
    out = []
    for _ in range(num_splits):
        out.append(lengths_1d.new_empty((lengths_1d.size(0) // num_splits,)))
    return out


@torch.library.register_fake("hstu_cuda_ops::lengths_reduce_dim1")
def _lengths_reduce_dim1_fake(lengths_1d, num_splits: int):
    torch._check(lengths_1d.dim() == 1)
    torch._check(num_splits > 0)

    # num_splits is static at export time.
    return lengths_1d.new_empty((num_splits,))


@torch.library.register_fake("hstu_cuda_ops::permute_and_split")
def _permute_and_split_fake(
    jagged_features,
    jagged_lengths,
    jagged_offsets,
    num_static_features: int,
    num_dynamic_features: int,
    features_order: list,
):
    # num_static_features and num_dynamic_features are static at export time.

    torch._check(jagged_features.dim() == 1)
    torch._check(jagged_lengths.dim() == 1)
    torch._check(num_static_features > 0)
    torch._check(num_dynamic_features > 0)

    num_features = num_static_features + num_dynamic_features
    ctx = torch.library.get_ctx()
    static_output_len = ctx.new_dynamic_size()
    dynamic_output_len = ctx.new_dynamic_size()
    out = [
        jagged_features.new_empty((static_output_len,)),
        jagged_features.new_empty((dynamic_output_len,)),
        jagged_lengths.new_empty(
            ((jagged_lengths.size(0) // num_features) * num_static_features,)
        ),
        jagged_lengths.new_empty(
            ((jagged_lengths.size(0) // num_features) * num_dynamic_features,)
        ),
    ]
    return out
