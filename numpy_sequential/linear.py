import numpy as np


class Linear:
    """A linear layer."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        self.weight = rng.random((d_model, d_hidden))
        self.bias = rng.random((d_hidden, ))

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        """Compute the matrix product x @ W + b."""
        return inputs_ @ self.weight + self.bias

    def backward(self):
        raise NotImplementedError
