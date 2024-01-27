import numpy as np


def softmax(inputs_: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(inputs_ - np.max(inputs_, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


class SoftmaxCrossEntropy:
    """Softmax cross-entropy loss function."""

    def forward(self, inputs_: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate the cross-entropy loss given logits (`inputs_`).

        Arguments:
            inputs_ (B, S, V): A batch (B) of sequences S with vocab size V.
            labels (B, S, 1): A dense set of labels for each batch & sequence.

        Returns:
            Cross-entropy loss.
        """
        logits = inputs_[np.arange(len(inputs_)), labels]
        cross_entropy = -logits + np.log(np.sum(np.exp(inputs_), axis=-1))
        return cross_entropy

    def backward(self):
        raise NotImplementedError
