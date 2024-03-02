import numpy as np
import pytest

import torch
from torch.nn import Parameter
from torch import nn

from numpitron.nn import (
    Softmax,
    Attention,
    InputEmbedding,
    Linear,
    MLP,
    LayerNorm,
    SoftmaxCrossEntropy,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


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
    rng,
):
    """Test that a forward pass from the Linear module is approximately
    the same with that of a basic torch Linear.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    attention = Attention(d_model, n_heads, d_model // n_heads, rng)
    attention_torch = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    # Transfer weights.
    attention_torch.in_proj_weight = Parameter(
        torch.from_numpy(
            np.concatenate(
                [
                    attention.q_proj.weight,
                    attention.k_proj.weight,
                    attention.v_proj.weight,
                ]
            ).reshape(attention_torch.in_proj_weight.shape)
        )
    )

    attention_torch.in_proj_bias = Parameter(
        torch.from_numpy(
            np.concatenate(
                [
                    attention.q_proj.bias,
                    attention.k_proj.bias,
                    attention.v_proj.bias,
                ]
            ).reshape(attention_torch.in_proj_bias.shape)
        )
    )

    attention_torch.out_proj.weight = Parameter(
        torch.from_numpy(attention.out_proj.weight.reshape(d_model, d_model)).T
    )
    attention_torch.out_proj.bias = Parameter(
        torch.from_numpy(attention.out_proj.bias.reshape(-1))
    )
    # Forward through both models.
    attention_forward = attention(inputs)
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
        atol=1e-3,
    )

    # Gradients calculated should be (approx) equal.
    # np.testing.assert_allclose(
    #     attention.out_proj.grads["weight"].reshape((d_model, d_model)).T,
    #     attention_torch.out_proj.weight.grad,
    #     atol=1e-5,
    # )
    # np.testing.assert_allclose(
    #     attention.out_proj.grads["bias"],
    #     attention_torch.out_proj.bias.grad,
    #     atol=1e-5,
    # )


@pytest.mark.parametrize(
    "batch_size,seq_len,n_heads",
    [(2, 3, 8), (1, 1, 8), (2, 2, 64)],
)
def test_softmax(
    batch_size: int,
    seq_len: int,
    n_heads: int,
    rng,
):
    """Test that a forward pass from the Linear module is approximately
    the same with that of a basic torch Linear.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
    inputs = rng.random((batch_size, n_heads, seq_len, seq_len)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    softmax = Softmax(axis=-1)
    softmax_torch = nn.Softmax(dim=-1)

    # Forward through both models.
    softmax_forward = softmax(inputs)
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


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,vocab_size",
    [(2, 3, 8, 20), (1, 1, 8, 20), (2, 2, 64, 20)],
)
def test_input_embedding(
    batch_size: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
    rng,
):
    """Test to ensure a linear layer for attention with heads would work."""
    # d_model to d_head, n_head.
    inputs = rng.integers((batch_size, seq_len, vocab_size))
    inputs_torch = torch.from_numpy(inputs)

    embedding = InputEmbedding(d_model, vocab_size, rng)
    embedding_torch = nn.Embedding(vocab_size, d_model)

    # Transfer weights.
    embedding_torch.weight = Parameter(torch.from_numpy(embedding.weight.T))

    # Forward through both models.
    embedding_forward = embedding(inputs)
    embedding_forward_torch = embedding_torch(inputs_torch)

    # Backward through both models.
    embedding.backward(np.ones_like(embedding_forward))
    embedding_forward_torch.sum().backward()

    # Forward pass should be (approx) equal.
    np.testing.assert_allclose(
        embedding_forward,
        embedding_forward_torch.detach().numpy(),
        atol=1e-5,
    )
    # Gradients calculated should be (approx) equal.
    np.testing.assert_allclose(
        embedding.grads["weight"].T,
        embedding_torch.weight.grad,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads",
    [(2, 3, 8, 2), (1, 1, 8, 2), (2, 2, 64, 8)],
)
def test_attention_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    rng,
):
    """Test to ensure a linear layer for attention with heads would work."""
    d_head = d_model // n_heads

    # d_model to d_head, n_head.
    inputs = rng.random((batch_size, seq_len, d_model))
    linear = Linear(d_model, (d_head, n_heads), rng)
    assert linear(inputs).shape == (batch_size, seq_len, d_head, n_heads)

    # d_head, n_head to d_model.
    inputs = rng.random((batch_size, seq_len, d_head, n_heads))
    linear = Linear((d_head, n_heads), d_model, rng)
    assert linear(inputs).shape == (batch_size, seq_len, d_model)


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(2, 3, 8), (1, 1, 8), (2, 2, 64)],
)
def test_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng,
):
    """Test that a forward pass from the Linear module is approximately
    the same with that of a basic torch Linear.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
    inputs = rng.random((batch_size, seq_len, d_model)).astype(np.float32)
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    linear = Linear(d_model, d_model * 4, rng)
    linear_torch = nn.Linear(d_model, d_model * 4)

    # Transfer weights.
    linear_torch.weight = Parameter(torch.from_numpy(linear.weight.T))
    linear_torch.bias = Parameter(torch.from_numpy(linear.bias))

    # Forward through both models.
    linear_forward = linear(inputs)
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



@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(2, 3, 8), (1, 1, 8), (2, 2, 32)],
)
def test_mlp(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng,
):
    """Test that a forward pass from the MLP module is approximately
    the same with that of a basic torch sequential MLP.

    Here we have to make sure the output is the same, but also
    that the collected gradients for each parameter is the same."""
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
    mlp_torch[0].weight = Parameter(torch.from_numpy(mlp.layers[0].weight.T))
    mlp_torch[0].bias = Parameter(torch.from_numpy(mlp.layers[0].bias))
    mlp_torch[2].weight = Parameter(torch.from_numpy(mlp.layers[2].weight.T))
    mlp_torch[2].bias = Parameter(torch.from_numpy(mlp.layers[2].bias))

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


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(2, 3, 8), (1, 1, 8), (2, 2, 64)],
)
def test_layer_norm(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng,
):
    """Test layer norm."""
    inputs = rng.random((batch_size, seq_len, d_model), dtype=np.float32)
    inputs_torch = torch.from_numpy(inputs)
    inputs_torch.requires_grad = True

    norm = LayerNorm(d_model, rng)
    norm_torch = nn.LayerNorm(d_model)

    # Copy weights
    norm_torch.weight = Parameter(torch.from_numpy(norm.weight))
    norm_torch.bias = Parameter(torch.from_numpy(norm.bias))

    outputs = norm(inputs)
    outputs_torch = norm_torch(inputs_torch)

    bw = norm.backward(np.ones_like(inputs))
    outputs_torch.sum().backward()

    np.testing.assert_allclose(outputs, outputs_torch.detach().numpy(), atol=1e-5)
    np.testing.assert_allclose(
        norm.grads["weight"], norm_torch.weight.grad.detach().numpy(), atol=1e-5
    )
    np.testing.assert_allclose(
        norm.grads["bias"], norm_torch.bias.grad.detach().numpy(), atol=1e-5
    )

    np.testing.assert_allclose(bw, inputs_torch.grad.detach().numpy(), atol=1e-5)

@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(2, 3, 8), (1, 1, 8), (2, 2, 64)],
)
def test_layer_norm_linear(
    batch_size: int,
    seq_len: int,
    d_model: int,
    rng,
):
    """Test layer norm."""
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

    outputs = norm[1](norm[0](inputs))
    outputs_torch = norm_torch(inputs_torch)

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


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size",
    [(2, 3, 20), (1, 1, 20), (2, 2, 256)],
)
def test_softmax_cross_entropy(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    rng,
):
    """Test the cross-entropy loss function with pytorch as ground-truth."""
    inputs = rng.random((batch_size, seq_len, vocab_size))
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    labels = rng.integers(0, vocab_size, (batch_size, seq_len))
    labels_torch = torch.from_numpy(labels).reshape(-1)

    ce_loss = SoftmaxCrossEntropy()
    ce_loss_torch = nn.CrossEntropyLoss(reduction="none")

    loss = ce_loss(inputs, labels)
    loss_torch = ce_loss_torch(inputs_torch, labels_torch)

    loss_torch.sum().backward()

    # Forward & backward pass should be equal.
    np.testing.assert_allclose(loss.reshape(-1), loss_torch.detach().numpy())
    np.testing.assert_allclose(
        ce_loss.backward().reshape(batch_size * seq_len, vocab_size),
        inputs_torch.grad.detach().numpy(),
    )
