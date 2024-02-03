import numpy as np
import pytest

import numpy_distributed as npdist
from numpy_sequential import SoftmaxCrossEntropy


@pytest.mark.parametrize("batch_size,seq_len,vocab_size,seed", [(1, 2, 20, 42)])
def test_parallel_softmax(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
):
    """The parallel softmax and cross-entropy loss function expects the input
    to be chunked along the vocab dim. We set up a sequential
    softmax cross-"""
    world_size = npdist.world_size()
    global_rng = np.random.default_rng(seed)

    inputs = global_rng.random((batch_size, seq_len, vocab_size))
    labels = global_rng.integers(0, vocab_size, (batch_size, seq_len))

    # Scatter the inputs along vocab dim.
    inputs_scatter = np.zeros((batch_size, seq_len, vocab_size // world_size))
    npdist.scatter(inputs, inputs_scatter, axis=2)

    # Forward through the parallel layer, all-reduce is already occuring
    # inside of it.
    parallel_forward = npdist.ParallelSoftmaxCrossEntropy().forward(
        inputs_scatter,
        labels,
    )

    np.testing.assert_allclose(
        SoftmaxCrossEntropy().forward(inputs, labels),
        parallel_forward,
    )
