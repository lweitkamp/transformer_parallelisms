import numpy as np
from mpi4py import MPI

from world_utils.tensor import scatter_init, all_reduce
from world_utils.world_info import get_rank


class InputEmbedding:
    """The input embedding lookup-table."""
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size

    @staticmethod
    def forward(weights: dict, tokens: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.

        Embedding weights are parallelized along the vocab dim (columns).
        To combine the embedded tokens, an an all-reduce is required.

        Arguments:
            weights: The embedding table at key `E`.
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        # Figure out token valid range for this specific embedding chunk.
        chunk_start = get_rank() * weights["E"].shape[1]
        chunk_end = chunk_start + weights["E"].shape[1]
        mask = np.logical_or(tokens < chunk_start, tokens >= chunk_end)

        # Set tokens to chunk range, mask tokens outside range.
        tokens = tokens - chunk_start
        tokens[mask] = 0

        # Take the correct embeddings and mask outside range.
        embedded_tokens = np.take(weights["E"].T, tokens, axis=0)
        embedded_tokens[mask, :] = 0.0

        # All-reduce, ensuring that the masked embeddings here
        # will be overwritten by the true embeddings elsewhere.
        embedded_tokens = all_reduce(embedded_tokens, reduction=MPI.SUM)
        return embedded_tokens

    def init_weights(self, rng):
        """The embeddings are a simple tensor of hidden dim by vocab size.
        We split them along the columns."""
        return {
            "E": scatter_init((self.d_model, self.vocab_size), rng, axis=1),
        }
