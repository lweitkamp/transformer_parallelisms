import pytest
import numpy as np

import numpy_distributed as ndist
from numpy_distributed.tensor_parallel import TensorParallelMLP
from numpy_sequential import MLP


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_parallel_mlp(batch_size: int, seq_len: int, d_model: int, seed: int):
    world_size = ndist.world_size()

    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + ndist.rank())

    # Create a normal- and a row parallel linear-layer.
    mlp = MLP(d_model, d_model * 4, global_rng)
    parallel_mlp = TensorParallelMLP(d_model, d_model * 4, local_rng)

    # Scatter the MLP weights.
    ndist.scatter(
        parallel_mlp.w1.weight,
        np.split(mlp.w1.weight, world_size, 1),
    )
    ndist.scatter(
        parallel_mlp.w2.weight,
        np.split(mlp.w2.weight, world_size, 0),
    )
    ndist.scatter(parallel_mlp.w1.bias, np.split(mlp.w1.bias, world_size, 0))
    parallel_mlp.w2.bias = mlp.w2.bias

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    # An all-reduce is required to combine the results.
    parallel_forward = parallel_mlp.forward(x)
    ndist.all_reduce(parallel_forward)

    np.testing.assert_allclose(mlp.forward(x), parallel_forward)
