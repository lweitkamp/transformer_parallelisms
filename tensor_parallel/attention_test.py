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
    """Run the MLP with an expected input."""
    random_state = np.random.default_rng(seed)

    weights = Attention(
        d_model=d_model,
        n_heads=n_heads,
    ).init_weights(rng=random_state)

    # Init and broadcast input.
    x = random_state.random((batch_size, seq_len, d_model))
    x = ndist.broadcast(x)

    # Init expected output.
    x_out = np.array([
        [[3.51835810e+01, 3.58623726e+01, 4.01228824e+01, 3.28495483e+01,
          3.54062995e+01, 3.52020623e+01, 3.10022348e+01, 2.47532137e+01,
          3.21282074e+01, 3.55002773e+01, 2.81269164e+01, 2.73963838e+01,
          3.02861454e+01, 3.68035725e+01, 3.81917009e+01, 4.27970480e+01],
         [1.17922054e-03, 8.00294625e-04, 1.12230698e-03, 7.86663864e-04,
          9.03707005e-04, 9.90655111e-04, 7.55695375e-04, 8.36140384e-04,
          9.34871864e-04, 7.22772077e-04, 5.69110131e-04, 6.04552978e-04,
          9.31625256e-04, 1.01709381e-03, 9.57672171e-04, 1.20517226e-03]]
    ]).astype(x.dtype)

    # # Forward pass and check only on root.
    out_all = Attention.forward(weights, x)
    if ndist.get_rank() == 0:
        np.testing.assert_almost_equal(out_all, x_out)


if __name__ == "__main__":
    test_attention(1, 2, 16, 4, 42)