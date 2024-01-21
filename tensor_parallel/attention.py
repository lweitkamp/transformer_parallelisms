import numpy as np

import numpy_distributed as ndist


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


class Attention:
    """A Multi-headed self-Attention (decoder-only) layer."""

    def __init__(self, d_model: int, n_heads_chunk: int, d_hidden: int, rng):
        """Initiate weights for the Attention layer. We split across
        /heads/ to leave the attention values valid."""
        self.q, self.k, self.v = rng.random(
            (3, d_model, n_heads_chunk, d_hidden)
        )
        self.b = rng.random(
            (n_heads_chunk, d_hidden, d_model)
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the self-attention layer.

        The operations here are scattered on the head dimension, so an
        all-reduce is required at the end.
        """
        # b: batch, s: seq len, d: d_model, h: num heads, m: d_hidden
        q = np.einsum("bsd, dhm -> bshm", x, self.q)
        k = np.einsum("bsd, dhm -> bshm", x, self.k)
        v = np.einsum("bsd, dhm -> bshm", x, self.v)

        attention = softmax(np.einsum("bshm, bzhm -> bhsz", q, k), axis=-1)
        y = np.einsum("bhss, bshm -> bshm", attention, v)
        z = np.einsum("bshm, hmd -> bsd", y, self.b)
        ndist.all_reduce(z)

        return z

    def backward(self):
        """Backward pass through the Attention layer."""
        raise NotImplementedError
