import numpy as np

from numpitron.nn.core import Layer, Parameter


class InputEmbedding(Layer):
    """The input embedding lookup-table."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        rng,
        dtype=np.float32,
    ):
        super().__init__()
        self.add_parameter("weight", (d_model, vocab_size), dtype, rng=rng)
        self.ctx: dict = {"inputs": None}

        self.weight.data *= 50  # yeah

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.

        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        self.ctx["inputs"] = inputs
        return np.take(self.weight.data.T, inputs, axis=0)

    def backward(self, grads: np.ndarray) -> np.ndarray:
        np.add.at(self.weight.gradient.T, self.ctx["inputs"], grads)
        self.ctx["inputs"] = None
        return grads


class OutputEmbedding(Layer):
    """The output embedding producing logits. weight are tied with that
    of the input embedding layer."""

    def __init__(self, weight: Parameter):
        super().__init__()
        self.weight: Parameter = weight

        self.ctx: dict = {"inputs": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate the logits through a simple matrix product."""
        self.ctx["inputs"] = inputs
        return inputs @ self.weight.data

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Perform a backward pass, calculating the gradients."""
        self.weight.gradient = np.einsum("bsd, bsv -> dv", self.ctx["inputs"], grads)
        self.ctx["inputs"] = None
        return grads @ self.weight.data.T


class PositionalEmbedding(Layer):
    """Technically an encoding, just using fourier features."""

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        dtype=np.float32,
    ):
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
