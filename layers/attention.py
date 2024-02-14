import numpy as np

import layers as npseq


class Attention:
    """A Multi-headed self-Attention (decoder-only) layer."""

    def __init__(
        self, d_model: int, n_heads: int, d_hidden: int, rng, dtype=np.float32
    ):
        self.q_proj = npseq.Linear(d_model, (n_heads, d_hidden), rng, dtype)
        self.k_proj = npseq.Linear(d_model, (n_heads, d_hidden), rng, dtype)
        self.v_proj = npseq.Linear(d_model, (n_heads, d_hidden), rng, dtype)
        self.out_proj = npseq.Linear((n_heads, d_hidden), d_model, rng, dtype)
        self.softmax = npseq.Softmax(axis=-1)

        self.scale = np.sqrt(d_hidden)

        self.ctx: dict = {"attention_weights": None, "q": None, "k": None, "v": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the self-attention layer."""
        self.ctx["inputs"] = inputs
        seq_len = inputs.shape[1]
        mask = np.expand_dims(np.tri(seq_len, seq_len, dtype=bool), (0, 1))

        q = self.q_proj.forward(inputs)
        k = self.k_proj.forward(inputs)
        v = self.v_proj.forward(inputs)

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

        grads_v = np.matmul(
            self.ctx["attention_weights"].transpose(0, 1, 3, 2),
            grads.transpose(0, 2, 1, 3),
        ).transpose(0, 2, 1, 3)

        grads = np.matmul(
            grads.transpose(0, 2, 1, 3), self.ctx["v"].transpose(0, 2, 3, 1)
        )

        grads = self.softmax.backward(grads)
        grads = np.where(self.ctx["mask"], grads, 0) / self.scale

        grads_q = np.matmul(grads, self.ctx["k"].transpose(0, 2, 1, 3)).transpose(
            0, 2, 1, 3
        )
        grads_k = np.einsum("bhsz, bshm -> bzhm", grads, self.ctx["q"])

        grads = (
            self.q_proj.backward(grads_q)
            + self.k_proj.backward(grads_k)
            + self.v_proj.backward(grads_v)
        )
        # grads = self.in_proj.backward(
        #     np.concatenate([grads_q, grads_k, grads_v], axis=2)
        # )

        self.ctx["attention_weights"] = None
        self.ctx["mask"] = None
        self.ctx["q"] = None
        self.ctx["k"] = None
        self.ctx["v"] = None

        return grads
