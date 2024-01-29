import numpy as np


def softmax(inputs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax cross-entropy loss function."""

    ctx: dict = {"inputs": None, "labels": None}

    def forward(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate the cross-entropy loss given logits (`inputs`).

        Arguments:
            inputs (B, S, V): A batch (B) of sequences S with vocab size V.
            labels (B, S, 1): A dense set of labels for each batch & sequence.

        Returns:
            Cross-entropy loss.
        """
        self.ctx["inputs"] = inputs
        self.ctx["labels"] = labels

        batch_size, seq_len, vocab_size = inputs.shape
        inputs = inputs.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)
        logits = inputs[np.arange(batch_size * seq_len), labels]

        cross_entropy = -logits + np.log(np.sum(np.exp(inputs), axis=-1))
        cross_entropy = cross_entropy.reshape((batch_size, seq_len))

        return cross_entropy

    def backward(self):
        """Backwards pass of the softmax-ce loss."""
        inputs = self.ctx["inputs"]
        labels = self.ctx["labels"]
        batch_size, seq_len, vocab_size = inputs.shape

        target = np.zeros((batch_size * seq_len, vocab_size))
        target[np.arange(batch_size * seq_len), labels.reshape(-1)] = 1
        target = target.reshape((batch_size, seq_len, -1))
        grads = softmax(inputs) - target

        # Clear cache.
        self.ctx["inputs"] = None
        self.ctx["labels"] = None

        return grads
