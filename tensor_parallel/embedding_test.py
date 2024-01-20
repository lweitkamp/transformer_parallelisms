import pytest
import numpy as np

from tensor_parallel.embedding import InputEmbedding, OutputEmbedding
import numpy_distributed as ndist


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,d_model,seed",
    [(2, 3, 20, 4, 42)],
)
def test_input_embedding(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    seed: int,
):
    """Run the Embedding with an expected input."""
    random_state = np.random.default_rng(seed)
    weights = InputEmbedding(
        d_model=d_model,
        vocab_size=vocab_size,
    ).init_weights(rng=random_state)

    # Init and broadcast input.
    tokens = random_state.integers(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
    )
    tokens = ndist.broadcast(tokens)

    # Init expected output.
    x_out = np.array([
        [[0.6316644,  0.66981399, 0.11453007, 0.78389821],
         [0.82276161, 0.37045971, 0.66485086, 0.23393949],
         [0.12811363, 0.15428949, 0.6824955,  0.03081783]],
        [[0.12811363, 0.15428949, 0.6824955,  0.03081783],
         [0.12811363, 0.15428949, 0.6824955,  0.03081783],
         [0.55458479, 0.12992151, 0.45891578, 0.29359376]]
    ]).astype(np.float32)

    # Forward pass and check only on root.
    out_all = InputEmbedding.forward(weights, tokens)
    if ndist.get_rank() == 0:
        np.testing.assert_almost_equal(out_all, x_out)


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,d_model,seed",
    [(2, 3, 20, 4, 42)],
)
def test_output_embedding(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    seed: int,
):
    """Run the Embedding with an expected input."""
    random_state = np.random.default_rng(seed)
    weights = InputEmbedding(
        d_model=d_model,
        vocab_size=vocab_size,
    ).init_weights(rng=random_state)

    # Init and broadcast input.
    tokens = random_state.random(
        size=(batch_size, seq_len, d_model),
    )
    tokens = ndist.broadcast(tokens)

    # Forward pass and check only on root.
    out_all = OutputEmbedding.forward(weights, tokens)
    ndist.reduce(out_all)

    if ndist.get_rank() == 0:
        np.testing.assert_almost_equal(out_all.sum(), 110.46088402258374)
