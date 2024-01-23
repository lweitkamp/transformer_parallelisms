import pytest
import numpy as np

import numpy_distributed as ndist
from tensor_parallel.embedding import (
    VocabParallelInputEmbedding,
    InputEmbedding,
)


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,d_model,seed",
    [(2, 3, 20, 4, 42)],
)
def test_parallel_input_embedding(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    seed: int,
):
    """Create an embedding layer and a vocab-parallel embedding layer.
    Scatter the embedding layer on the parallel layer and see if
    outputs match on both."""
    world_size = ndist.world_size()

    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + ndist.rank())

    # Create a normal- and a row parallel linear-layer.
    embedding = InputEmbedding(d_model, vocab_size, global_rng)
    parallel_embedding = VocabParallelInputEmbedding(
        d_model,
        vocab_size,
        local_rng,
    )

    # Scatter the embedding layer's weights.
    ndist.scatter(parallel_embedding.e, np.split(embedding.e, world_size, 1))

    # Init the input with the global seed.
    x = global_rng.integers(low=0, high=vocab_size, size=(batch_size, seq_len))

    # An all-reduce is required to combine the results.
    parallel_forward = parallel_embedding.forward(x)
    ndist.all_reduce(parallel_forward)

    np.testing.assert_allclose(embedding.forward(x), parallel_forward)
