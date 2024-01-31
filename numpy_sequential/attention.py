import numpy as np

import numpy_sequential as npseq


class Attention:
    """A Multi-headed self-Attention (decoder-only) layer."""

    def __init__(self, d_model: int, n_heads: int, d_hidden: int, rng):
        self.q, self.k, self.v = rng.random((3, d_model, n_heads, d_hidden))
        self.b = rng.random((n_heads, d_hidden, d_model))

        self.ctx: dict = {"inputs": None}
        self.grads: dict = {"q": None, "k": None, "v": None, "b": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the self-attention layer."""
        self.ctx["inputs"] = inputs

        # b: batch, s: seq len, d: d_model, h: num heads, m: d_hidden
        q = np.einsum("bsd, dhm -> bshm", inputs, self.q)  # 1
        k = np.einsum("bsd, dhm -> bshm", inputs, self.k)  # 2
        v = np.einsum("bsd, dhm -> bshm", inputs, self.v)  # 3

        attention_weights = np.einsum("bshm, bzhm -> bhsz", q, k)
        attention = npseq.softmax(attention_weights, axis=-1)  # 4

        y = np.einsum("bhss, bshm -> bshm", attention, v)  # 5
        z = np.einsum("bshm, hmd -> bsd", y, self.b)  # 6

        self.ctx["q"] = q
        self.ctx["k"] = k
        self.ctx["v"] = k
        self.ctx["attention"] = attention
        self.ctx["y"] = y

        return z

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass through the Attention layer."""
        divisor = np.prod(self.ctx["inputs"].shape[:2])

        self.grads["b"] = np.einsum("bsd, bshm -> hmd", grads, self.ctx["y"]) / divisor
        grads = np.einsum("bsd, hmd -> bshm", grads, self.b)

        grads_v = np.einsum("bshm, bhsz -> bshm", grads, self.ctx["attention"])
        grads = np.einsum("bshm, bzhm -> bhsz", grads, self.ctx["v"])

        # TODO: softmax derivative.
        grads = grads

        grads_q = np.einsum("bhsz, bshm -> bshm", grads, self.ctx["k"])
        grads_k = np.einsum("bhsz, bshm -> bzhm", grads, self.ctx["q"])

        # Update QKV
        self.grads["q"] = (
            np.einsum("bshm, bsd -> dhm", grads_q, self.ctx["inputs"]) / divisor
        )
        self.grads["k"] = (
            np.einsum("bshm, bsd -> dhm", grads_k, self.ctx["inputs"]) / divisor
        )
        self.grads["v"] = (
            np.einsum("bshm, bsd -> dhm", grads_v, self.ctx["inputs"]) / divisor
        )

        grads = (
            np.einsum("bshm, dhm -> bsd", grads_q, self.q)
            + np.einsum("bshm, dhm -> bsd", grads_k, self.k)
            + np.einsum("bshm, dhm -> bsd", grads_v, self.v)
        )

        # Clear the cache.
        self.ctx["inputs"] = None
        self.ctx["attention"] = None
        self.ctx["y"] = None

        return grads
