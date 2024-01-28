import numpy as np
import pytest

import numpy_distributed as npdist
from numpy_sequential import Attention


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads,seed",
    [(1, 2, 4, 2, 42), (2, 4, 8, 4, 42)],
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
    world_size = npdist.world_size()

    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + npdist.rank() + 1)
    # Create a normal- and a head parallel attention-layer.
    attention = Attention(d_model, n_heads, d_model, global_rng)
    head_attention = npdist.HeadParallelAttention(
        d_model,
        n_heads,
        d_model,
        local_rng,
    )

    # Scatter the attention layer's weights.
    npdist.scatter(head_attention.q, np.split(attention.q, world_size, 1))
    npdist.scatter(head_attention.k, np.split(attention.k, world_size, 1))
    npdist.scatter(head_attention.v, np.split(attention.v, world_size, 1))
    npdist.scatter(head_attention.b, np.split(attention.b, world_size, 0))

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    np.testing.assert_allclose(attention.forward(x), head_attention.forward(x))
