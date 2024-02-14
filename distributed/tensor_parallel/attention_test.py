import numpy as np
import pytest

import distributed as npdist
from layers import Attention


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads,seed",
    [
        (1, 1, 8, 2, 42),
        (3, 1, 8, 2, 42),
        (1, 3, 8, 2, 42),
        (3, 3, 8, 2, 42),
    ],
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
    rng = np.random.default_rng(seed)

    # Create a normal- and a head parallel attention-layer.
    attention = Attention(d_model, n_heads, d_model, rng)
    head_attention = npdist.HeadParallelAttention(
        d_model,
        n_heads,
        d_model,
        rng,
    )

    # (d_model, n_heads, d_hidden)
    npdist.scatter(attention.q_proj.weight, head_attention.q_proj.weight, axis=1)
    npdist.scatter(attention.k_proj.weight, head_attention.k_proj.weight, axis=1)
    npdist.scatter(attention.v_proj.weight, head_attention.v_proj.weight, axis=1)
    npdist.scatter(attention.q_proj.bias, head_attention.q_proj.bias, axis=0)
    npdist.scatter(attention.k_proj.bias, head_attention.k_proj.bias, axis=0)
    npdist.scatter(attention.v_proj.bias, head_attention.v_proj.bias, axis=0)
    npdist.scatter(attention.out_proj.weight, head_attention.out_proj.weight, axis=0)
    npdist.scatter(attention.out_proj.bias, head_attention.out_proj.bias, axis=0)

    # Init the input with the global seed.
    x = rng.random((batch_size, seq_len, d_model))
    y = np.ones_like(x)

    np.testing.assert_allclose(attention.forward(x), head_attention.forward(x))
    np.testing.assert_allclose(attention.backward(y), head_attention.backward(y))
