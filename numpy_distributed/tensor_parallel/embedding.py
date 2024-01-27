import numpy as np

import numpy_distributed as npdist
from numpy_sequential import InputEmbedding


class VocabParallelInputEmbedding(InputEmbedding):
    """The input embedding lookup-table, split over the vocab dim."""

    def __init__(self, d_model: int, vocab_size: int, rng):
        super().__init__(
            d_model=d_model,
            vocab_size=vocab_size // npdist.world_size(),
            rng=rng,
        )

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
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
        chunk_start = npdist.rank() * self.e.shape[1]
        chunk_end = chunk_start + self.e.shape[1]
        mask = np.logical_or(inputs_ < chunk_start, inputs_ >= chunk_end)

        # Set tokens to chunk range, mask tokens outside range.
        inputs_ = inputs_ - chunk_start
        inputs_[mask] = 0

        # Take the correct embeddings and mask outside range.
        embedded_tokens = np.take(self.e.T, inputs_, axis=0)
        embedded_tokens[mask, :] = 0.0

        return embedded_tokens
