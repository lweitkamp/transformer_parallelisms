import pytest
from mpi4py import MPI
import numpy as np

from megatron_lm.attention import Attention
from world_utils.world_info import get_rank


# @pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(2, 3, 4, 42)])
def attention_test(batch_size: int, seq_len: int, d_model: int, n_heads: int, seed: int):
    """Run the MLP with an expected input."""
    comm = MPI.COMM_WORLD
    random_state = np.random.default_rng(seed)
    weights = Attention(d_model=d_model, n_heads=n_heads).init_weights(rng=random_state)
    if get_rank() == 0:
        for key, val in weights.items():
            print(f"{key}: {val.shape}")

    # Init and broadcast input.
    # x = random_state.random((batch_size, seq_len, d_model)) if get_rank() == 0 else None
    # x = comm.bcast(x, root=0)

    # # Forward pass and check only on root.
    # out_all = Attention.forward(weights, x)



if __name__ == "__main__":
    print("Batch size: 2, Sequence length: 3, d_model: 4")
    attention_test(
        batch_size=1,
        seq_len=3,
        d_model=8,
        n_heads=2,
        seed=42,
    )
