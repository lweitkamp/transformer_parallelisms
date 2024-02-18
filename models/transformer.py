import numpy as np

import layers


class Transformer:
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

        # Input embeddings.
        self.input_embedding = layers.InputEmbedding(d_model, vocab_size, rng)
        self.pos_embed = layers.PositionalEmbedding(d_model, seq_len)

        # Sequence of transformer blocks.
        self.layers = [
            layers.TransformerBlock(d_model, n_heads, rng, dtype) for _ in range(n_layers)
        ]

        # De-embedding
        self.output_embedding = layers.OutputEmbedding(self.input_embedding.weights)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """..."""
        inputs = self.input_embedding.forward(inputs)
        inputs = self.pos_embed.forward(inputs)
        for layer in self.layers:
            inputs = layer.forward(inputs)
        inputs = self.output_embedding.forward(inputs)
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """..."""
        grads = self.output_embedding.backward(grads)
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        grads = self.pos_embed.backward(grads)
        grads = self.input_embedding.backward(grads)
        return grads


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    transformer = Transformer(
        seq_len=128,
        vocab_size=512,
        n_layers=2,
        d_model=256,
        n_heads=8,
        rng=rng,
    )

    out = transformer.forward(
        rng.integers(0, 512, size=(1, 128))
    )
    transformer.backward(np.ones_like(out))

    assert out.shape == (1, 128, 512)
