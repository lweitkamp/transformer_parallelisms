import numpy as np
import pytest
import torch
import torch.nn as nn

from numpy_sequential import Attention


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads",
    [(2, 3, 8, 2), (1, 1, 8, 2), (2, 2, 64, 4)],
)
def test_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
):
    """Test that a forward pass from the Linear module is approximately
    the same with that of a basic torch Linear.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
    rng = np.random.default_rng(42)

    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    attention = Attention(d_model, n_heads, d_model // n_heads, rng)
    # linear_torch = nn.Linear(d_model, d_model * 4)

    # Transfer weights.
    # linear_torch.weight = nn.Parameter(torch.from_numpy(linear.weight.T))
    # linear_torch.bias = nn.Parameter(torch.from_numpy(linear.bias))

    # Forward through both models.
    attention_forward = attention.forward(inputs)
    # attention_forward_torch = linear_torch(inputs_torch)

    # Backward through both models.
    # attention_forward_torch.sum().backward()
    attention.backward(np.ones((batch_size, seq_len, d_model)))

    # Forward pass should be (approx) equal.
    # np.testing.assert_allclose(
    #     attention_forward.reshape(batch_size * seq_len, d_model * 4),
    #     attention_forward_torch.detach().numpy(),
    #     atol=1e-5,
    # )

    # # Gradients calculated should be (approx) equal.
    # np.testing.assert_allclose(
    #     attention.grads["weight"].T,
    #     linear_torch.weight.grad,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     attention.grads["bias"],
    #     linear_torch.bias.grad,
    #     atol=1e-5,
    # )
