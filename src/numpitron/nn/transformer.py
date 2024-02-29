import nn
import numpy as np

from nn.core import Block


class TransformerBlock(Block):
    """A vanilla transformer block."""

    def __init__(self, d_model, n_heads, rng, dtype):
        super().__init__()

        self.attention = nn.Attention(
            d_model, n_heads, d_model // n_heads, rng, dtype
        )
        self.norm1 = nn.LayerNorm(d_model, rng)
        self.mlp = nn.MLP(d_model, d_model * 4, rng, dtype)
        self.norm2 = nn.LayerNorm(d_model, rng)

        self.layers.extend([self.attention, self.norm1, self.mlp, self.norm2])

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = self.norm1(self.attention(inputs)) + inputs
        inputs = self.norm2(self.mlp(inputs)) + inputs
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        grads = self.norm2.backward(grads)
        grads = self.mlp.backward(grads)
        grads = self.norm1.backward(grads)
        grads = self.attention.backward(grads)
        return grads
