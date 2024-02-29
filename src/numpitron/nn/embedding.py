import numpy as np

from nn.core import Layer


class InputEmbedding(Layer):
    """The input embedding lookup-table."""

    def __init__(self, d_model: int, vocab_size: int, rng):
        super().__init__()

        self.weight = rng.random((d_model, vocab_size))

        self.ctx: dict = {"inputs": None}
        self.grads["weight"] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.

        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        self.ctx["inputs"] = inputs
        return np.take(self.weight.T, inputs, axis=0)

    def backward(self, grads: np.ndarray) -> np.ndarray:
        self.grads["weight"] = np.zeros_like(self.weight)
        np.add.at(self.grads["weight"].T, self.ctx["inputs"], grads)
        self.ctx["inputs"] = None
        return grads


class OutputEmbedding(Layer):
    """The output embedding producing logits. weight are tied with that
    of the input embedding layer."""

    def __init__(self, weight: np.ndarray):
        super().__init__()

        self.weight = weight

        self.ctx: dict = {"inputs": None}
        self.grads["weight"] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate the logits through a simple matrix product."""
        self.ctx["inputs"] = inputs
        return inputs @ self.weight

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Perform a backward pass, calculating the gradients."""
        divisor = np.prod(self.ctx["inputs"].shape[:2])
        self.grads["weight"] = (
            np.einsum("bsd, bsv -> dv", self.ctx["inputs"], grads) / divisor
        )
        self.ctx["inputs"] = None
        return grads @ self.weight.T


class PositionalEmbedding(Layer):
    """Technically an encoding, just using fourier features."""

    def __init__(self, d_model: int, seq_len: int, dtype=np.float32):
        super().__init__()

        pos = np.expand_dims(np.arange(0, seq_len), -1)
        _2i = np.arange(d_model, step=2) / d_model

        self.encoding = np.zeros((seq_len, d_model), dtype=dtype)
        self.encoding[:, 0::2] = np.sin(pos / (10000**_2i))
        self.encoding[:, 1::2] = np.cos(pos / (10000**_2i))

    def forward(self, inputs: np.ndarray):
        _, seq_len, *_ = inputs.shape
        return self.encoding[:seq_len, :] + inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        return grads
