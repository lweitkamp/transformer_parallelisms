import numpy as np

import layers as npseq


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
        seq_len = inputs.shape[1]
        mask = np.expand_dims(np.tri(seq_len, seq_len, dtype=bool), (0, 1))

        q, k, v = np.split(self.in_proj.forward(inputs), 3, axis=2)

        attention_weights = np.einsum("bshm, bzhm -> bhsz", q, k) / self.scale
        attention_weights = np.where(mask, attention_weights, float("-inf"))
        attention_weights = self.softmax.forward(attention_weights)

        attention = np.einsum("bhsz, bzhm -> bshm", attention_weights, v)
        out = self.out_proj.forward(attention)

        self.ctx["attention_weights"] = np.copy(attention_weights)
        self.ctx["mask"] = np.copy(mask)
        self.ctx["q"] = np.copy(q)
        self.ctx["k"] = np.copy(k)
        self.ctx["v"] = np.copy(v)

        return out

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """Backward pass through the Attention layer."""
        grads = self.out_proj.backward(grads)

        grads_v = np.einsum("bshm, bhsz -> bshm", grads, self.ctx["attention_weights"])
        grads = np.einsum("bshm, bzhm -> bhsz", grads, self.ctx["v"])

        grads = self.softmax.backward(grads)
        grads = np.where(self.ctx["mask"], grads, 0) / self.scale

        grads_q = np.einsum("bhsz, bshm -> bshm", grads, self.ctx["k"])
        grads_k = np.einsum("bhsz, bshm -> bzhm", grads, self.ctx["q"])

        grads = self.in_proj.backward(
            np.concatenate([grads_q, grads_k, grads_v], axis=2)
        )

        self.ctx["attention_weights"] = None
        self.ctx["mask"] = None
        self.ctx["q"] = None
        self.ctx["k"] = None
        self.ctx["v"] = None

        return grads
