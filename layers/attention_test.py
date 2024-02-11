import numpy as np
import pytest
import torch
import torch.nn as nn

from layers import Attention


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads",
    [
        (1, 1, 8, 2),
        (3, 1, 8, 2),
        (1, 3, 8, 2),
        (3, 3, 8, 2),
    ],
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
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    attention = Attention(d_model, n_heads, d_model // n_heads, rng)
    attention_torch = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    # Transfer weights.
    attention_torch.in_proj_weight = nn.Parameter(
        torch.from_numpy(attention.in_proj.weight.reshape(d_model, 3 * d_model).T)
    )
    attention_torch.bias = nn.Parameter(
        torch.from_numpy(attention.in_proj.bias.reshape(-1))
    )
    attention_torch.out_proj.weight = nn.Parameter(
        torch.from_numpy(attention.out_proj.weight.reshape(d_model, d_model).T)
    )
    attention_torch.out_proj.bias = nn.Parameter(
        torch.from_numpy(attention.out_proj.bias.reshape(-1))
    )
    # Forward through both models.
    attention_forward = attention.forward(inputs)
    attention_forward_torch, _ = attention_torch(
        inputs_torch,
        inputs_torch,
        inputs_torch,
        is_causal=True,
        attn_mask=torch.from_numpy(~attention.ctx["mask"].squeeze((0, 1))),
        average_attn_weights=False,
        need_weights=False,
    )

    # Backward through both models.
    attention_forward_torch.sum().backward()
    attention.backward(np.ones((batch_size, seq_len, d_model)))

    # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        attention_forward,
        attention_forward_torch.detach().numpy(),
        atol=1e-5,
    )

    # Gradients calculated should be (approx) equal.
    np.testing.assert_allclose(
        attention.out_proj.grads["weight"].reshape((8, 8)).T,
        attention_torch.out_proj.weight.grad,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        attention.out_proj.grads["bias"],
        attention_torch.out_proj.bias.grad,
        atol=1e-5,
    )

    # TODO: ensure gradients of in_proj are also approx equal.
    # hard to figure out if its an issue with gradient calculation
    # or with float differences adding up.
