import numpy as np
import pytest
import torch
import torch.nn as nn

from layers import InputEmbedding


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,vocab_size",
    [(2, 3, 8, 20), (1, 1, 8, 20), (2, 2, 64, 20)],
)
def test_input_embedding(
    batch_size: int,
    seq_len: int,
    d_model: int,
    vocab_size: int,
):
    """Test to ensure a linear layer for attention with heads would work."""
    rng = np.random.default_rng(42)

    # d_model to d_head, n_head.
    inputs = rng.integers((batch_size, seq_len, vocab_size))
    inputs_torch = torch.from_numpy(inputs)
    # inputs_torch.requires_grad = True

    embedding = InputEmbedding(d_model, vocab_size, rng)
    embedding_torch = nn.Embedding(vocab_size, d_model)

    # Transfer weights.
    embedding_torch.weight = nn.Parameter(torch.from_numpy(embedding.weight.T))

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
