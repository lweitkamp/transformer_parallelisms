import numpy as np
import pytest
import torch
import torch.nn as nn

from layers import LayerNorm


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads",
    [(2, 3, 8, 2), (1, 1, 8, 2), (2, 2, 64, 8)],
)
def test_layer_norm(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
):
    """Test layer norm."""
    rng = np.random.default_rng(42)

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

    np.testing.assert_allclose(outputs, outputs_torch.detach().numpy(), atol=1e-5)

