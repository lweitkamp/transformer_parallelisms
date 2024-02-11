import numpy as np
import pytest
import torch

from layers import SoftmaxCrossEntropy


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size",
    [(2, 3, 20), (1, 1, 20), (2, 2, 256)],
)
def test_softmax_cross_entropy(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
):
    """Test the cross-entropy loss function with pytorch as ground-truth."""
    rng = np.random.default_rng(42)

    inputs = rng.random((batch_size, seq_len, vocab_size))
    inputs_torch = torch.from_numpy(inputs).reshape(batch_size * seq_len, -1)
    inputs_torch.requires_grad = True

    labels = rng.integers(0, vocab_size, (batch_size, seq_len))
    labels_torch = torch.from_numpy(labels).reshape(-1)

    ce_loss = SoftmaxCrossEntropy()
    ce_loss_torch = torch.nn.CrossEntropyLoss(reduction="none")

    loss = ce_loss.forward(inputs, labels)
    loss_torch = ce_loss_torch(inputs_torch, labels_torch)

    loss_torch.sum().backward()

    # Forward & backward pass should be equal.
    np.testing.assert_allclose(loss.reshape(-1), loss_torch.detach().numpy())
    np.testing.assert_allclose(
        ce_loss.backward().reshape(batch_size * seq_len, vocab_size),
        inputs_torch.grad.detach().numpy(),
    )
