import numpy as np

import numpy_distributed as ndist


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
        chunk_start = ndist.get_rank() * weights["E"].shape[1]
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
        ndist.all_reduce(embedded_tokens)
        return embedded_tokens

    def init_weights(self, rng):
        """The embeddings are a simple tensor of hidden dim by vocab size.
        We split them along the columns."""
        return {
            "E": ndist.scatter_init((self.d_model, self.vocab_size), rng, axis=1),
        }


class OutputEmbedding:
    """The output embedding of the model (produces logits)."""

    @staticmethod
    def forward(weights: dict, tokens: np.ndarray) -> np.ndarray:
        """Given the embedding table, un-embed the tokens into logits.

        Embedding weights are parallelized along the vocab dim (columns).
        We keep the output parallelized for the softmax + cross-entropy combo,
        this saves us from all-reducing the entire vocab size.

        Arguments:
            weights: The embedding table at key `E`.
            tokens (B, S, D): A batch B of tokens S of size D.

        Returns:
            Token logits.
        """
        y = tokens @ weights["E"]
        return y
