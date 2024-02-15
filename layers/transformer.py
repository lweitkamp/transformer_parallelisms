import layers
import numpy as np


class TransformerBlock:
    """A vanilla transformer block."""
    def __init__(self, d_model, n_heads, rng, dtype):
        self.attention = layers.Attention(
            d_model, n_heads, d_model // n_heads, rng, dtype
        )
        self.norm1 = layers.LayerNorm(d_model, rng)
        self.mlp = layers.MLP(d_model, d_model * 4, rng, dtype)
        self.norm2 = layers.LayerNorm(d_model, rng)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = self.norm1.forward(self.attention.forward(inputs)) + inputs
        inputs = self.norm2.forward(self.mlp.forward(inputs)) + inputs
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        grads = self.norm2.backward(grads)
        grads = self.mlp.backward(grads)
        grads = self.norm1.backward(grads)
        grads = self.attention.backward(grads)
        return grads
