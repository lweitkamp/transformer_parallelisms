import numpy as np

import layers


class Transformer:
    def __init__(
        self,
        n_layers: int,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        rng,
        dtype,
    ):
        """..."""
        self.input_embedding = layers.InputEmbedding(d_model, vocab_size, rng)
        self.layers = [
            layers.TransformerBlock(d_model, n_heads, rng, dtype) for _ in range(n_layers)
        ]
        self.output_embedding = layers.OutputEmbedding(self.input_embedding.weights)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = self.input_embedding.forward(inputs)
        for layer in self.layers:
            inputs = layer.forward(inputs)
        inputs = self.output_embedding.forward(inputs)
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        grads = self.output_embedding.backward(grads)
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        grads = self.input_embedding.backward(grads)
        return grads
