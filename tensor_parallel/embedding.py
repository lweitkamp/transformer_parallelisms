import numpy as np

import numpy_distributed as ndist


class InputEmbedding:
    """The input embedding lookup-table."""
    
    def __init__(self, d_model: int, vocab_size: int, rng):
        self.e = rng.random((d_model, vocab_size))

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.
        
        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        return np.take(self.e.T, tokens, axis=0)


class VocabParallelEmbedding(InputEmbedding):
    """The input embedding lookup-table, split over the vocab dim."""

    def __init__(self, d_model: int, vocab_size: int, rng):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size // ndist.world_size(),
            rng=rng,
        )

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.

        Embedding weights are parallelized along the vocab dim (columns).
        To combine the embedded tokens, an an all-reduce is required after
        this layer.

        Arguments:
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            (Masked) token embeddings.
        """
        # Figure out token valid range for this specific embedding chunk.
        chunk_start = ndist.rank() * self.e.shape[1]
        chunk_end = chunk_start + self.e.shape[1]
        mask = np.logical_or(tokens < chunk_start, tokens >= chunk_end)

        # Set tokens to chunk range, mask tokens outside range.
        tokens = tokens - chunk_start
        tokens[mask] = 0

        # Take the correct embeddings and mask outside range.
        embedded_tokens = np.take(self.e.T, tokens, axis=0)
        embedded_tokens[mask, :] = 0.0

        # All-reduce, ensuring that the masked embeddings here
        # will be overwritten by the true embeddings elsewhere.
        return embedded_tokens
