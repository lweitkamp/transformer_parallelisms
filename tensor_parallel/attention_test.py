import pytest
import numpy as np

from tensor_parallel.attention import Attention
import numpy_distributed as ndist


@pytest.mark.parametrize(
        "batch_size,seq_len,d_model,n_heads,seed",
        [(1, 2, 16, 4, 42)],
)
def test_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    seed: int,
):
    """Create a sharded attention layer and a non-sharded one,
    scatter the weights of the non-sharded one and ensure that a
    forward pass with both will yield the same outcome."""
    random_state = np.random.default_rng(seed + ndist.rank())

    attention_sharded = Attention(
        d_model=d_model,
        n_heads_chunk=n_heads // ndist.world_size(),
        d_hidden=n_heads // d_model,
        rng=random_state,
    )

    # Create an attention layer that has the 'full' parameter shape
    # and scatter the weights to all devices.
    attention = Attention(
        d_model=d_model,
        n_heads_chunk=n_heads,
        d_hidden=n_heads // d_model,
        rng=random_state,
    )
    ndist.scatter(
        tensor=attention_sharded.q,
        scatter_list=np.split(attention.q, ndist.world_size(), 1)
    )
    ndist.scatter(
        tensor=attention_sharded.k,
        scatter_list=np.split(attention.k, ndist.world_size(), 1)
    )
    ndist.scatter(
        tensor=attention_sharded.v,
        scatter_list=np.split(attention.v, ndist.world_size(), 1)
    )
    ndist.scatter(
        tensor=attention_sharded.b,
        scatter_list=np.split(attention.b, ndist.world_size(), 0)
    )

    # Init and broadcast input.
    x = random_state.random((batch_size, seq_len, d_model))
    x = ndist.broadcast(x)

    # Check that a forward pass with sharded weights yields the same
    # result as that without sharded weights.
    np.testing.assert_almost_equal(
        attention_sharded.forward(x),
        attention.forward(x),
    )
