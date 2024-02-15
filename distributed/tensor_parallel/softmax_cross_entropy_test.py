import numpy as np
import pytest

import distributed as npdist
from layers import SoftmaxCrossEntropy


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
    ce = SoftmaxCrossEntropy()
    parallel_ce = npdist.ParallelSoftmaxCrossEntropy()

    np.testing.assert_allclose(
        ce.forward(inputs, labels),
        parallel_ce.forward(inputs_scatter, labels),
    )

    # An all-gather is required to combine the results.
    parallel_backward = np.zeros((batch_size, seq_len, vocab_size))
    npdist.all_gather(parallel_ce.backward(), parallel_backward)

    np.testing.assert_allclose(
        ce.backward(),
        parallel_backward,
    )
