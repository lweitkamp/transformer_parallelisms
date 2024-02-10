import numpy as np

import numpy_sequential as npseq


class Attention:
    """A Multi-headed self-Attention (decoder-only) layer."""

    def __init__(
        self, d_model: int, n_heads: int, d_hidden: int, rng, dtype=np.float32
    ):
        self.q = npseq.Linear(d_model, (n_heads, d_hidden), rng, dtype)
        self.k = npseq.Linear(d_model, (n_heads, d_hidden), rng, dtype)
        self.v = npseq.Linear(d_model, (n_heads, d_hidden), rng, dtype)
        self.b = npseq.Linear((n_heads, d_hidden), d_model, rng, dtype)
        self.softmax = npseq.Softmax(axis=-1)

        self.ctx: dict = {"attention": None, "q": None, "k": None, "v": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the self-attention layer."""
        self.ctx["inputs"] = inputs

        q, k, v = self.q.forward(inputs), self.k.forward(inputs), self.v.forward(inputs)

        attention_weights = np.einsum("bshm, bzhm -> bhsz", q, k)
        attention = self.softmax.forward(attention_weights)

        y = np.einsum("bhss, bshm -> bshm", attention, v)
        z = self.b.forward(y)

        self.ctx["attention"] = np.copy(attention)
        self.ctx["q"] = np.copy(q)
        self.ctx["k"] = np.copy(k)
        self.ctx["v"] = np.copy(v)

        return z

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass through the Attention layer."""

        grads = self.b.backward(grads)

        grads_v = np.einsum("bsmh, bhsz -> bshm", grads, self.ctx["attention"])
        grads = np.einsum("bsmh, bzhm -> bhsz", grads, self.ctx["v"])

        grads = self.softmax.backward(grads)

        grads_q = np.einsum("bhsz, bshm -> bshm", grads, self.ctx["k"])
        grads_k = np.einsum("bhsz, bshm -> bzhm", grads, self.ctx["q"])

        grads = (
            self.q.backward(grads_q)
            + self.k.backward(grads_k)
            + self.v.backward(grads_v)
        )

        self.ctx["attention"] = None
        self.ctx["q"] = None
        self.ctx["k"] = None
        self.ctx["v"] = None

        return grads
