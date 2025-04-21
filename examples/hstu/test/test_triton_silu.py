import pytest
import torch

from ops.triton_ops.triton_silu import triton_silu


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        torch.float32,
        torch.float16,
    ],
)
@pytest.mark.parametrize("batch_size", [200, 500, 1000])
@pytest.mark.parametrize("hidden_dim", [128, 256, 512])
def test_triton_silu_fwd(dtype, batch_size, hidden_dim):
    input = (
        torch.empty(batch_size, hidden_dim, device="cuda", dtype=dtype)
        .uniform_(-1, 1)
        .requires_grad_(True)
    )
    ref_input = input.detach().clone().requires_grad_(True)

    ref_output = torch.nn.functional.silu(ref_input)
    output = triton_silu(input)

    grad_output = torch.ones_like(ref_output)

    ref_output.backward(grad_output)
    output.backward(grad_output)

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(input.grad, ref_input.grad)
