import pytest
from mpi4py import MPI
import numpy as np

from mlp import MLP
from world_info import get_rank


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(2, 3, 4, 42)])
def mlp_test(batch_size: int, seq_len: int, d_model: int, seed: int):
    """Run the MLP with an expected input."""
    comm = MPI.COMM_WORLD
    random_state = np.random.default_rng(seed)
    weights = MLP(d_model=d_model).init_weights(rng=random_state)

    # Init and broadcast input.
    x = random_state.random((batch_size, seq_len, d_model)) if get_rank() == 0 else None
    x = comm.bcast(x, root=0)

    # Init expected output.
    x_out = np.array([
        [[6.5797237,   7.02095552,  5.20528612,  7.60512021],
         [5.99235118,  6.69518111,  4.55373548,  6.98397626],
         [8.94932038, 10.0784911,   6.89457243, 10.4773979 ]],
        [[9.33846795, 10.19174526,  7.07632521, 10.86081999],
         [5.93787353,  6.62879174,  4.71989223,  6.99819881],
         [7.86049214,  8.71594933,  6.1110801,   9.33112637]]
    ]).astype(x.dtype)

    # Forward pass and check only on root.
    out_all = MLP.forward(weights, x)
    if get_rank() == 0:
        np.testing.assert_almost_equal(out_all, x_out)
