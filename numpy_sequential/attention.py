import numpy as np
from numpy_sequential.softmax_cross_entropy import softmax


class Attention:
    """A Multi-headed self-Attention (decoder-only) layer."""

    def __init__(self, d_model: int, n_heads: int, d_hidden: int, rng):
        self.q, self.k, self.v = rng.random((3, d_model, n_heads, d_hidden))
        self.b = rng.random((n_heads, d_hidden, d_model))

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        """Forward pass through the self-attention layer."""
        # b: batch, s: seq len, d: d_model, h: num heads, m: d_hidden
        q = np.einsum("bsd, dhm -> bshm", inputs_, self.q)
        k = np.einsum("bsd, dhm -> bshm", inputs_, self.k)
        v = np.einsum("bsd, dhm -> bshm", inputs_, self.v)

        attention = softmax(np.einsum("bshm, bzhm -> bhsz", q, k), axis=-1)
        y = np.einsum("bhss, bshm -> bshm", attention, v)
        z = np.einsum("bshm, hmd -> bsd", y, self.b)
        return z

    def backward(self):
        """Backward pass through the Attention layer."""
        raise NotImplementedError
