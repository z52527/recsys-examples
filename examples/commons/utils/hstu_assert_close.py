import torch


def assert_hstu_close(actual, out_ref, fp32_ref, fwd: bool = True):
    """
    compare the maximum absolute error between actual and fp32_ref
    with the maximum absolute error between out_ref and fp32_ref
    """
    L = actual.size(0)
    actual = actual.view(L, -1)
    out_ref = out_ref.view(L, -1)
    fp32_ref = fp32_ref.view(L, -1)
    assert fp32_ref.dtype == torch.float32, "fp32_ref should be float32"

    multiplier = 2 if fwd else 5
    left_abs_max = (actual - fp32_ref).abs().max().item()
    right_abs_max = (out_ref - fp32_ref).abs().max().item()

    assert left_abs_max <= multiplier * right_abs_max, (
        f"actual - fp32_ref: {left_abs_max}, " f"out_ref - fp32_ref: {right_abs_max}"
    )
