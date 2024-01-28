import pytest
import numpy as np

import numpy_distributed as npdist
from numpy_sequential import MLP


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_parallel_mlp(batch_size: int, seq_len: int, d_model: int, seed: int):
    world_size = npdist.world_size()

    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + npdist.rank() + 1)

    # Create a normal- and a row parallel linear-layer.
    mlp = MLP(d_model, d_model * 4, global_rng)
    parallel_mlp = npdist.TensorParallelMLP(d_model, d_model * 4, local_rng)

    # Scatter the MLP weights.
    npdist.scatter(
        parallel_mlp.w1.weight,
        np.split(mlp.w1.weight, world_size, 1),
    )
    npdist.scatter(
        parallel_mlp.w2.weight,
        np.split(mlp.w2.weight, world_size, 0),
    )
    npdist.scatter(parallel_mlp.w1.bias, np.split(mlp.w1.bias, world_size, 0))
    parallel_mlp.w2.bias = mlp.w2.bias

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    np.testing.assert_allclose(mlp.forward(x), parallel_mlp.forward(x))
