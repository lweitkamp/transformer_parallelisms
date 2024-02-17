import numpy as np
import pytest
import torch
import torch.nn as nn

from layers import LayerNorm, Linear


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,seed",
    [(2, 3, 8, 42), (1, 1, 8, 42), (2, 2, 64, 42)],
)
def test_layer_norm(
    batch_size: int,
    seq_len: int,
    d_model: int,
    seed: int,
):
    """Test layer norm."""
    rng = np.random.default_rng(seed)

    inputs = rng.random((batch_size, seq_len, d_model), dtype=np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    norm = LayerNorm(d_model, rng)
    norm_torch = nn.LayerNorm(d_model)

    # Copy weights
    norm_torch.weight = nn.Parameter(torch.from_numpy(norm.weight.T))
    norm_torch.bias = nn.Parameter(torch.from_numpy(norm.bias))

    outputs = norm.forward(inputs)
    outputs_torch = norm_torch.forward(inputs_torch)

    norm.backward(np.ones_like(inputs))
    outputs_torch.sum().backward()

    np.testing.assert_allclose(outputs, outputs_torch.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(
        norm.grads["weight"], norm_torch.weight.grad.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        norm.grads["bias"], norm_torch.bias.grad.detach().numpy(), atol=1e-5
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,seed",
    [(2, 3, 8, 42), (1, 1, 8, 42), (2, 2, 64, 42)],
)
def test_layer_norm_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    seed: int,
):
    """Test layer norm."""
    rng = np.random.default_rng(seed)

    inputs = rng.random((batch_size, seq_len, d_model), dtype=np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    norm = [
        Linear(d_model, d_model, rng),
        LayerNorm(d_model, rng),
    ]
    norm_torch = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.LayerNorm(d_model),
    )

    # Copy weights
    norm_torch[1].weight = nn.Parameter(torch.from_numpy(norm[1].weight))
    norm_torch[1].bias = nn.Parameter(torch.from_numpy(norm[1].bias))
    norm_torch[0].weight = nn.Parameter(torch.from_numpy(norm[0].weight.T))
    norm_torch[0].bias = nn.Parameter(torch.from_numpy(norm[0].bias))

    outputs = norm[1].forward(norm[0].forward(inputs))
    outputs_torch = norm_torch.forward(inputs_torch)

    norm[0].backward(norm[1].backward(np.ones_like(inputs)))
    outputs_torch.sum().backward()

    np.testing.assert_allclose(outputs, outputs_torch.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(
        norm[1].grads["weight"], norm_torch[1].weight.grad.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        norm[1].grads["bias"], norm_torch[1].bias.grad.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        norm[0].grads["weight"].T,
        norm_torch[0].weight.grad.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        norm[0].grads["bias"],
        norm_torch[0].bias.grad.detach().numpy(),
        rtol=1e-4,
        atol=1e-4,
    )
