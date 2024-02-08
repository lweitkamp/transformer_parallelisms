import numpy as np


def softmax(inputs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax cross-entropy loss function."""

    ctx: dict = {"inputs": None, "labels": None}

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate the cross-entropy loss given logits (`inputs`).

        Arguments:
            inputs (B, S, V): A batch (B) of sequences S with vocab size V.
            labels (B, S, 1): A dense set of labels for each batch & sequence.

        Returns:
            Cross-entropy loss.
        """
        self.ctx["logits"] = logits
        self.ctx["labels"] = labels
        batch_size, seq_len, vocab_size = logits.shape

        logits = logits.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        # Subtract max for stability.
        logits = logits - logits.max(axis=-1, keepdims=True)

        predicted_logits = logits[np.arange(batch_size * seq_len), labels]
        logsumexp_logits = np.log(np.sum(np.exp(logits), axis=-1))

        loss = logsumexp_logits - predicted_logits
        loss = loss.reshape((batch_size, seq_len))
        return loss

    def backward(self):
        """Backwards pass of the softmax-ce loss."""
        logits = self.ctx["logits"]
        labels = self.ctx["labels"]
        batch_size, seq_len, vocab_size = logits.shape

        logits = logits.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        probabilities = softmax(logits, axis=-1)
        probabilities[np.arange(batch_size * seq_len), labels] -= 1

        # Clear cache.
        self.ctx["inputs"] = None
        self.ctx["labels"] = None

        return probabilities.reshape(batch_size, seq_len, vocab_size)
