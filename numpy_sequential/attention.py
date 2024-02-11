import numpy as np

import numpy_sequential as npseq


class Attention:
    """A Multi-headed self-Attention (decoder-only) layer."""

    def __init__(
        self, d_model: int, n_heads: int, d_hidden: int, rng, dtype=np.float32
    ):
        self.in_proj = npseq.Linear(d_model, (3 * n_heads, d_hidden), rng, dtype)
        self.out_proj = npseq.Linear((n_heads, d_hidden), d_model, rng, dtype)
        self.softmax = npseq.Softmax(axis=-1)

        self.scale = np.sqrt(d_hidden)

        self.ctx: dict = {"attention": None, "q": None, "k": None, "v": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the self-attention layer."""
        self.ctx["inputs"] = inputs

        q, k, v = np.split(self.in_proj.forward(inputs), 3, axis=2)

        attention_weights = np.einsum("bshm, bzhm -> bhsz", q, k) / self.scale
        _, _, s1, s2 = attention_weights.shape
        mask = np.expand_dims(np.tri(s1, s2, dtype=bool), (0, 1))

        attention_weights = np.where(mask, attention_weights, float("-inf"))
        attention = self.softmax.forward(attention_weights)

        y = np.einsum("bhss, bshm -> bshm", attention, v)
        out = self.out_proj.forward(y)

        self.ctx["attention"] = np.copy(attention)
        self.ctx["mask"] = np.copy(mask)
        self.ctx["q"] = np.copy(q)
        self.ctx["k"] = np.copy(k)
        self.ctx["v"] = np.copy(v)

        return out

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass through the Attention layer."""
        grads = self.out_proj.backward(grads)

        grads_v = np.einsum("bshm, bhsz -> bshm", grads, self.ctx["attention"])
        grads = np.einsum("bshm, bzhm -> bhsz", grads, self.ctx["v"])

        grads = self.softmax.backward(grads)
        grads = np.where(self.ctx["mask"], grads, 0) / self.scale

        grads_q = np.einsum("bhsz, bshm -> bshm", grads, self.ctx["k"])
        grads_k = np.einsum("bhsz, bshm -> bzhm", grads, self.ctx["q"])

        grads = self.in_proj.backward(
            np.concatenate([grads_q, grads_k, grads_v], axis=2)
        )

        self.ctx["attention"] = None
        self.ctx["mask"] = None
        self.ctx["q"] = None
        self.ctx["k"] = None
        self.ctx["v"] = None

        return grads
