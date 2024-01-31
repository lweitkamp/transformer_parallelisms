import numpy as np


class InputEmbedding:
    """The input embedding lookup-table."""

    def __init__(self, d_model: int, vocab_size: int, rng):
        self.weights = rng.random((d_model, vocab_size))

        self.ctx: dict = {"inputs": None}
        self.grads: dict = {"weights": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.

        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        self.ctx["inputs"] = inputs
        return np.take(self.weights.T, inputs, axis=0)

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Perform a backward pass, calculating the gradients."""
        self.grads["weights"] = None
        return grads


class OutputEmbedding:
    """The output embedding producing logits. Weights are tied with that
    of the input embedding layer."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights

        self.ctx: dict = {"inputs": None}
        self.grads: dict = {"weights": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate the logits through a simple matrix product."""
        self.ctx["inputs"] = inputs
        return inputs @ self.weights

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Perform a backward pass, calculating the gradients."""
        divisor = np.prod(self.ctx["inputs"].shape[:2])
        self.grads["weights"] = (
            np.einsum("bsd, bsv -> dv", self.ctx["inputs"], grads) / divisor
        )
        self.ctx["inputs"] = None
        return grads @ self.weights.T
