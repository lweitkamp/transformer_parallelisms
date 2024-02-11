import numpy as np
import pytest

import distributed as npdist
from layers import MLP


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_parallel_mlp(batch_size: int, seq_len: int, d_model: int, seed: int):
    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + npdist.rank() + 1)

    # Create a normal- and a row parallel linear-layer.
    mlp = MLP(d_model, d_model * 4, global_rng)
    parallel_mlp = npdist.TensorParallelMLP(d_model, d_model * 4, local_rng)

    # Scatter the MLP weights.
    npdist.scatter(mlp.layers[0].weights, parallel_mlp.layers[0].weights, axis=1)
    npdist.scatter(mlp.layers[2].weights, parallel_mlp.layers[2].weights, axis=0)
    npdist.scatter(mlp.layers[0].bias, parallel_mlp.layers[0].bias, axis=0)
    parallel_mlp.layers[2].bias = mlp.layers[2].bias

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    np.testing.assert_allclose(mlp.forward(x), parallel_mlp.forward(x))
