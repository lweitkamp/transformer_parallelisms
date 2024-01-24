import numpy as np


class InputEmbedding:
    """The input embedding lookup-table."""

    def __init__(self, d_model: int, vocab_size: int, rng):
        self.e = rng.random((d_model, vocab_size))

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.
        
        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        return np.take(self.e.T, inputs_, axis=0)


class OutputEmbedding:
    """The output embedding producing logits. Weights are tied with that
    of the input embedding layer."""

    def __init__(self, weights: np.ndarray):
        self.e = weights

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        """Calculate the logits through a simple matrix product."""
        return inputs_ @ self.e
