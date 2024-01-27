import pytest
import numpy as np

import numpy_distributed as npdist
from numpy_sequential import SoftmaxCrossEntropy


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 20, 42)])
def test_parallel_softmax(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
):
    world_size = npdist.world_size()
    global_rng = np.random.default_rng(seed)

    # Create a normal- and a row parallel linear-layer.
    smce = SoftmaxCrossEntropy()
    parallel_smce = npdist.ParallelSoftmaxCrossEntropy()

    inputs = global_rng.random((batch_size, seq_len, vocab_size))
    labels = global_rng.integers(0, vocab_size, (batch_size, seq_len))
    
    # Scatter the inputs along vocab dim.
    inputs_scatter = np.zeros((batch_size, seq_len, vocab_size // world_size))
    npdist.scatter(
        inputs_scatter,
        np.split(inputs, world_size, 1)
    )
    

    
    # # Init the input with the global seed.
    # x = global_rng.random((batch_size, seq_len, d_model))

    # # An all-reduce is required to combine the results.
    # parallel_forward = parallel_mlp.forward(x)
    # npdist.all_reduce(parallel_forward)

    # np.testing.assert_allclose(mlp.forward(x), parallel_forward)
