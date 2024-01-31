import numpy as np
import pytest

import numpy_distributed as npdist
from numpy_sequential import Linear


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_row_linear(batch_size: int, seq_len: int, d_model: int, seed: int):
    world_size = npdist.world_size()

    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + npdist.rank() + 1)

    # Create a normal- and a row parallel linear-layer.
    linear = Linear(d_model, d_model, global_rng)
    row_linear = npdist.RowParallelLinear(d_model, d_model, local_rng)

    # Scatter the linear layer's weights
    npdist.scatter(row_linear.weights, np.split(linear.weights, world_size, 0))
    row_linear.bias = linear.bias

    # Init the input. We need to scatter it to devices on the row dim.
    x = global_rng.random((batch_size, seq_len, d_model))
    scatter_x = np.empty((batch_size, seq_len, d_model // world_size))
    npdist.scatter(scatter_x, np.split(x, world_size, 2))

    # An all-reduce is required to combine the results.
    parallel_forward = row_linear.forward(scatter_x)
    npdist.all_reduce(parallel_forward)

    np.testing.assert_allclose(linear.forward(x), parallel_forward)


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_column_linear(batch_size: int, seq_len: int, d_model: int, seed: int):
    world_size = npdist.world_size()

    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + npdist.rank())

    # Create a normal- and a row parallel linear-layer.
    linear = Linear(d_model, d_model, global_rng)
    column_linear = npdist.ColumnParallelLinear(d_model, d_model, local_rng)

    # Scatter the linear layer's weights.
    npdist.scatter(column_linear.weights, np.split(linear.weights, world_size, 1))
    npdist.scatter(column_linear.bias, np.split(linear.bias, world_size, 0))

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    # An all-gather is required to combine the results.
    gathered_forward = np.zeros((batch_size, seq_len, d_model))
    npdist.all_gather(gathered_forward, column_linear.forward(x))

    np.testing.assert_allclose(linear.forward(x), gathered_forward)
