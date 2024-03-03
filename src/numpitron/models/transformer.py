import numpy as np


from numpitron import nn
from numpitron.nn.core import Block


class Transformer(Block):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        rng,
        dtype=np.float32,
    ):
        """..."""
        super().__init__()

        self.layers.extend([
            nn.InputEmbedding(d_model, vocab_size, rng),
            nn.PositionalEmbedding(d_model, seq_len),
        ])
        self.layers.extend([
            nn.TransformerBlock(d_model, n_heads, rng, dtype)
            for _ in range(n_layers)
        ])
        self.layers.append(nn.OutputEmbedding(self.layers[0].weight))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        return grads
