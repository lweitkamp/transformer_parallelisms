import numpy as np


class Linear:
    """A linear layer."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        self.weights = rng.random((d_model, d_hidden))
        self.bias = rng.random((d_hidden,))

        self.ctx = {"inputs": None}
        self.grads = {"weights": None, "bias": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the matrix product x @ W + b."""
        forward = inputs @ self.weights + self.bias
        self.ctx["inputs"] = inputs
        return forward

    def backward(self, grads: np.ndarray):
        """Perform a backward pass, calculating the gradients."""
        self.grads["weights"] = np.einsum("bsm, bsd -> md", self.ctx["inputs"], grads)
        self.grads["bias"] = grads.sum(axis=(0, 1), keepdims=True)
        self.ctx["inputs"] = None
        return grads @ self.weights.T
