import numpy as np
import pytest
import torch
import torch.nn as nn

from layers import MLP


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(2, 3, 8), (1, 1, 8), (2, 2, 32)],
)
def test_mlp(
    batch_size: int,
    seq_len: int,
    d_model: int,
):
    """Test that a forward pass from the MLP module is approximately
    the same with that of a basic torch sequential MLP.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
    rng = np.random.default_rng(42)

    inputs = (
        rng.random((batch_size, seq_len, d_model)).astype(np.float32) + 1
    ) / d_model
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    mlp = MLP(d_model, d_model * 4, rng)
    mlp_torch = nn.Sequential(
        nn.Linear(d_model, d_model * 4),
        nn.ReLU(),
        nn.Linear(d_model * 4, d_model),
    )

    # Transfer weights.
    mlp_torch[0].weight = nn.Parameter(torch.from_numpy(mlp.layers[0].weight.T))
    mlp_torch[0].bias = nn.Parameter(torch.from_numpy(mlp.layers[0].bias))
    mlp_torch[2].weight = nn.Parameter(torch.from_numpy(mlp.layers[2].weight.T))
    mlp_torch[2].bias = nn.Parameter(torch.from_numpy(mlp.layers[2].bias))

    # Forward through both models.
    mlp_forward = mlp(inputs)
    mlp_forward_torch = mlp_torch(inputs_torch)

    # Backward through both models.
    mlp_forward_torch.sum().backward()
    mlp.backward(np.ones_like(inputs)).reshape(batch_size * seq_len, d_model)

    # # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        mlp_forward.reshape(batch_size * seq_len, d_model),
        mlp_forward_torch.detach().numpy(),
        atol=1e-5,
    )

    # Gradients calculated should be (approx) equal.
    np.testing.assert_allclose(
        mlp.layers[2].grads["weight"].T,
        mlp_torch[2].weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        mlp.layers[2].grads["bias"],
        mlp_torch[2].bias.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        mlp.layers[0].grads["weight"].T,
        mlp_torch[0].weight.grad,
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        mlp.layers[0].grads["bias"],
        mlp_torch[0].bias.grad,
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == "__main__":
    test_mlp(2, 2, 64)
