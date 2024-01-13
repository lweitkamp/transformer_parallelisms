import numpy as np
from mpi4py import MPI

from world_utils.tensor import scatter_init, all_reduce, broadcast
from world_utils.world_info import get_rank


def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


class Attention:
    """A Multi-headed Attention layer. We ignore causal masking
    for simplicity."""
    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.d_hidden = d_model // n_heads
        self.n_heads = n_heads

    @staticmethod
    def forward(weights: dict, x: np.ndarray) -> np.ndarray:
        """Broadcast x to all devices, multiply x by scattered weights and
        sum the results."""
        x = broadcast(x)

        # b: batch, s: seq len, d: d_model, h: num heads, m: d_hidden
        q = np.einsum("bsd, dhm -> bshm", x, weights["Q"])
        k = np.einsum("bsd, dhm -> bshm", x, weights["K"])
        v = np.einsum("bsd, dhm -> bshm", x, weights["V"])

        attention = softmax(np.einsum("bshm, bzhm -> bhsz", q, k), axis=-1)
        y = np.einsum("bhss, bshm -> bshm", attention, v)
        z = np.einsum("bshm, hmd -> bsd", y, weights["B"])

        # All-reduce.
        out = all_reduce(z, reduction=MPI.SUM)
        return out

    def backward(self):
        """Backward pass through the Attention layer."""
        raise NotImplementedError

    def init_weights(self, rng):
        """Initiate weights for the Attention layer.
        We split across /heads/ to leave the attention values valid."""
        qkv_shape = (self.d_model, self.n_heads, self.d_hidden)
        b_shape = (self.d_hidden, self.n_heads, self.d_model)
        return {
            "Q": scatter_init(qkv_shape, rng, axis=1),
            "K": scatter_init(qkv_shape, rng, axis=1),
            "V": scatter_init(qkv_shape, rng, axis=1),
            "B": scatter_init(b_shape, rng, axis=0),
        }
