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

        max_logit = logits.max(axis=-1, keepdims=True)
        logits = logits - max_logit
        logits = logits.reshape(batch_size * seq_len, vocab_size)
        labels = labels.reshape(batch_size * seq_len)

        predicted_logits = logits[np.arange(batch_size * seq_len), labels]
        sum_exp_logits = np.sum(np.exp(logits), axis=-1)

        loss = (np.log(sum_exp_logits) - predicted_logits) / sum_exp_logits
        loss = loss.reshape((batch_size, seq_len))
        return loss

    def backward(self):
        """Backwards pass of the softmax-ce loss."""
        inputs = self.ctx["logits"]
        labels = self.ctx["labels"]
        batch_size, seq_len, vocab_size = inputs.shape

        target = np.zeros((batch_size * seq_len, vocab_size))
        target[np.arange(batch_size * seq_len), labels.reshape(-1)] = 1
        target = target.reshape((batch_size, seq_len, -1))

        # Clear cache.
        self.ctx["inputs"] = None
        self.ctx["labels"] = None

        grads = softmax(inputs) * (1 - target)

        return grads
