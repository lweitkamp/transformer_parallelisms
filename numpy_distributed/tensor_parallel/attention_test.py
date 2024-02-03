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
    """Create a sequential attention layer and scatter the values to a
    head-parallel version. Compare outputs of a generated input,
    the outputs should match."""
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
    npdist.scatter(attention.q, head_attention.q, axis=1)
    npdist.scatter(attention.k, head_attention.k, axis=1)
    npdist.scatter(attention.v, head_attention.v, axis=1)
    npdist.scatter(attention.b, head_attention.b, axis=0)

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    np.testing.assert_allclose(attention.forward(x), head_attention.forward(x))
