import numpy as np
import pytest
import torch
import torch.nn as nn

from numpy_sequential import Softmax


@pytest.mark.parametrize(
    "batch_size,seq_len,n_heads",
    [(2, 3, 8), (1, 1, 8), (2, 2, 64)],
)
def test_softmax(
    batch_size: int,
    seq_len: int,
    n_heads: int,
):
    """Test that a forward pass from the Linear module is approximately
    the same with that of a basic torch Linear.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
    rng = np.random.default_rng(42)

    inputs = rng.random((batch_size, n_heads, seq_len, seq_len)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    softmax = Softmax(axis=-1)
    softmax_torch = torch.nn.Softmax(dim=-1)

    # Forward through both models.
    softmax_forward = softmax.forward(inputs)
    softmax_forward_torch = softmax_torch(inputs_torch)

    # Backward through both models.
    y = softmax_forward_torch.sum().backward()
    grads = softmax.backward(np.ones((batch_size, n_heads, seq_len, seq_len)))

    # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        softmax_forward,
        softmax_forward_torch.detach().numpy(),
        atol=1e-5,
    )

    # TODO: can we somehow check the grad_out in pytorch?