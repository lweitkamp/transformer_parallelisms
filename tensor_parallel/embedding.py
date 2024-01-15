import numpy as np

from world_utils.tensor import scatter_init, all_gather


class InputEmbedding:
    """The input embedding lookup-table."""
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size

    @staticmethod
    def forward(weights: dict, tokens: np.ndarray) -> np.ndarray:
        """Given an embedding table and input tokens, embed the tokens.

        Embedding weights are parallelized along the vocab dim (columns).
        To combine the embedded tokens, an an all-gather is required.

        NOTE: the paper mentions that an all-reduce is required, but that is
        probably a typo.

        Arguments:
            weights: The embedding table at key `E`.
            tokens (B, S): A batch (B) of sequences (S) of integer tokens.

        Returns:
            Token embeddings.
        """
        embedded_tokens = np.take(weights, tokens, axis=1)
        embedded_tokens = all_gather(embedded_tokens, axis=1)
        return embedded_tokens

    def init_weights(self, rng):
        """The embeddings are a simple tensor of hidden dim by vocab size.
        We split them along the columns."""
        return {
            "E": scatter_init((self.d_model, self.vocab_size), rng, axis=1),
        }
