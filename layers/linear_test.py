import numpy as np
import pytest
import torch
import torch.nn as nn

from layers import Linear


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads",
    [(2, 3, 8, 2), (1, 1, 8, 2), (2, 2, 64, 8)],
)
def test_attention_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
):
    """Test to ensure a linear layer for attention with heads would work."""
    rng = np.random.default_rng(42)

    d_head = d_model // n_heads

    # d_model to d_head, n_head.
    inputs = rng.random((batch_size, seq_len, d_model))
    linear = Linear(d_model, (d_head, n_heads), rng)
    assert linear.forward(inputs).shape == (batch_size, seq_len, d_head, n_heads)

    # d_head, n_head to d_model.
    inputs = rng.random((batch_size, seq_len, d_head, n_heads))
    linear = Linear((d_head, n_heads), d_model, rng)
    assert linear.forward(inputs).shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(2, 3, 8), (1, 1, 8), (2, 2, 64)],
)
def test_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
):
    """Test that a forward pass from the Linear module is approximately
    the same with that of a basic torch Linear.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
    rng = np.random.default_rng(42)

    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    linear = Linear(d_model, d_model * 4, rng)
    linear_torch = nn.Linear(d_model, d_model * 4)

    # Transfer weights.
    linear_torch.weight = nn.Parameter(torch.from_numpy(linear.weight.T))
    linear_torch.bias = nn.Parameter(torch.from_numpy(linear.bias))

    # Forward through both models.
    linear_forward = linear.forward(inputs)
    linear_forward_torch = linear_torch(inputs_torch)

    # Backward through both models.
    linear_forward_torch.sum().backward()
    linear.backward(np.ones((batch_size, seq_len, d_model * 4)))

    # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        linear_forward.reshape(batch_size * seq_len, d_model * 4),
        linear_forward_torch.detach().numpy(),
        atol=1e-5,
    )

    # Gradients calculated should be (approx) equal.
    np.testing.assert_allclose(
        linear.grads["weight"].T,
        linear_torch.weight.grad,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        linear.grads["bias"],
        linear_torch.bias.grad,
        atol=1e-5,
    )
