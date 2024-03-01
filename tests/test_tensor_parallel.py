import numpy as np
import pytest

from numpitron import nn
from numpitron.parallel import tensor_parallel, distributed as dist


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model,n_heads,seed",
    [
        (1, 1, 8, 2, 42),
        (3, 1, 8, 2, 42),
        (1, 3, 8, 2, 42),
        (3, 3, 8, 2, 42),
    ],
)
def test_attention(
    batch_size: int,
    seq_len: int,
    d_model: int,
    n_heads: int,
    seed: int,
):
    """Create a sequential attention layer and scatter the values to a
    head-parallel version. Compare outputs of a generated input,
    the outputs should match."""
    rng = np.random.default_rng(seed)

    # Create a normal- and a head parallel attention-layer.
    attention = nn.Attention(d_model, n_heads, d_model, rng)
    head_attention = tensor_parallel.HeadParallelAttention(
        d_model,
        n_heads,
        d_model,
        rng,
    )

    # (d_model, n_heads, d_hidden)
    dist.scatter(attention.q_proj.weight, head_attention.q_proj.weight, axis=1)
    dist.scatter(attention.k_proj.weight, head_attention.k_proj.weight, axis=1)
    dist.scatter(attention.v_proj.weight, head_attention.v_proj.weight, axis=1)
    dist.scatter(attention.q_proj.bias, head_attention.q_proj.bias, axis=0)
    dist.scatter(attention.k_proj.bias, head_attention.k_proj.bias, axis=0)
    dist.scatter(attention.v_proj.bias, head_attention.v_proj.bias, axis=0)
    dist.scatter(attention.out_proj.weight, head_attention.out_proj.weight, axis=0)
    dist.scatter(attention.out_proj.bias, head_attention.out_proj.bias, axis=0)

    # Init the input with the global seed.
    x = rng.random((batch_size, seq_len, d_model))
    y = np.ones_like(x)

    np.testing.assert_allclose(attention.forward(x), head_attention.forward(x))
    np.testing.assert_allclose(attention.backward(y), head_attention.backward(y))


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,d_model,seed",
    [(2, 3, 20, 4, 42)],
)
def test_parallel_input_embedding(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    seed: int,
):
    """Create an embedding layer and a vocab-parallel embedding layer.
    Scatter the embedding layer on the parallel layer and see if
    outputs match on both."""
    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + dist.rank() + 1)

    # Create a normal- and a row parallel linear-layer.
    embedding = nn.InputEmbedding(d_model, vocab_size, global_rng)
    parallel_embedding = tensor_parallel.VocabParallelInputEmbedding(
        d_model,
        vocab_size,
        local_rng,
    )

    # Scatter the embedding layer's weights.
    dist.scatter(embedding.weight, parallel_embedding.weight, axis=1)

    # Init the input with the global seed.
    x = global_rng.integers(low=0, high=vocab_size, size=(batch_size, seq_len))

    # An all-reduce is required to combine the results.
    parallel_forward = parallel_embedding.forward(x)
    dist.all_reduce(parallel_forward)

    np.testing.assert_allclose(embedding.forward(x), parallel_forward)


@pytest.mark.parametrize(
    "batch_size,seq_len,vocab_size,d_model,seed",
    [(2, 3, 20, 4, 42)],
)
def test_parallel_output_embedding(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    seed: int,
):
    """The output embedding is tied to the input embedding. We create both
    a normal and a parallel input embedding, copy the weights to the output
    variant and check if the input-output for both parallel and sequential
    models agree."""
    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + dist.rank())

    # Create a normal- and a row parallel linear-layer.
    embedding = nn.InputEmbedding(d_model, vocab_size, global_rng)
    parallel_embedding = tensor_parallel.VocabParallelInputEmbedding(
        d_model,
        vocab_size,
        local_rng,
    )

    # Scatter the embedding layer's weights.
    dist.scatter(embedding.weight, parallel_embedding.weight, axis=1)

    # Create the output embeddings with weight tied from the input embedding.
    output_embedding = nn.OutputEmbedding(weight=embedding.weight)
    parallel_output_embedding = nn.OutputEmbedding(weight=parallel_embedding.weight)

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    # An all-gather is required to combine the results.
    parallel_forward = np.zeros((batch_size, seq_len, vocab_size))
    dist.all_gather(parallel_output_embedding.forward(x), parallel_forward)

    np.testing.assert_allclose(output_embedding.forward(x), parallel_forward)


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_row_linear(batch_size: int, seq_len: int, d_model: int, seed: int):
    world_size = dist.world_size()

    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + dist.rank() + 1)

    # Create a normal- and a row parallel linear-layer.
    linear = nn.Linear(d_model, d_model, global_rng)
    row_linear = tensor_parallel.RowParallelLinear(d_model, d_model, local_rng)

    # Scatter the linear layer's weights
    dist.scatter(linear.weight, row_linear.weight, axis=0)
    row_linear.bias = linear.bias

    # Init the input. We need to scatter it to devices on the row dim.
    x = global_rng.random((batch_size, seq_len, d_model))
    scatter_x = np.empty((batch_size, seq_len, d_model // world_size))
    dist.scatter(x, scatter_x, axis=2)

    # An all-reduce is required to combine the results.
    parallel_forward = row_linear.forward(scatter_x)
    dist.all_reduce(parallel_forward)

    np.testing.assert_allclose(linear.forward(x), parallel_forward)


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_column_linear(batch_size: int, seq_len: int, d_model: int, seed: int):
    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + dist.rank())

    # Create a normal- and a row parallel linear-layer.
    linear = nn.Linear(d_model, d_model, global_rng)
    column_linear = tensor_parallel.ColumnParallelLinear(d_model, d_model, local_rng)

    # Scatter the linear layer's weights.
    dist.scatter(linear.weight, column_linear.weight, axis=1)
    dist.scatter(linear.bias, column_linear.bias, axis=0)

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    # An all-gather is required to combine the results.
    gathered_forward = np.zeros((batch_size, seq_len, d_model))
    dist.all_gather(column_linear.forward(x), gathered_forward)

    np.testing.assert_allclose(linear.forward(x), gathered_forward)


@pytest.mark.parametrize("batch_size,seq_len,d_model,seed", [(1, 2, 4, 42)])
def test_parallel_mlp(batch_size: int, seq_len: int, d_model: int, seed: int):
    global_rng = np.random.default_rng(seed)
    local_rng = np.random.default_rng(seed + dist.rank() + 1)

    # Create a normal- and a row parallel linear-layer.
    mlp = nn.MLP(d_model, d_model * 4, global_rng)
    parallel_mlp = tensor_parallel.TensorParallelMLP(d_model, d_model * 4, local_rng)

    # Scatter the MLP weights.
    dist.scatter(mlp.layers[0].weight, parallel_mlp.layers[0].weight, axis=1)
    dist.scatter(mlp.layers[2].weight, parallel_mlp.layers[2].weight, axis=0)
    dist.scatter(mlp.layers[0].bias, parallel_mlp.layers[0].bias, axis=0)
    parallel_mlp.layers[2].bias = mlp.layers[2].bias

    # Init the input with the global seed.
    x = global_rng.random((batch_size, seq_len, d_model))

    np.testing.assert_allclose(mlp.forward(x), parallel_mlp.forward(x))

    np.testing.assert_allclose(
        mlp.backward(np.ones_like(x)), parallel_mlp.backward(np.ones_like(x))
    )


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
    world_size = dist.world_size()
    global_rng = np.random.default_rng(seed)

    inputs = global_rng.random((batch_size, seq_len, vocab_size))
    labels = global_rng.integers(0, vocab_size, (batch_size, seq_len))

    # Scatter the inputs along vocab dim.
    inputs_scatter = np.zeros((batch_size, seq_len, vocab_size // world_size))
    dist.scatter(inputs, inputs_scatter, axis=2)

    # Forward through the parallel layer, all-reduce is already occuring
    # inside of it.
    ce = nn.SoftmaxCrossEntropy()
    parallel_ce = tensor_parallel.ParallelSoftmaxCrossEntropy()

    np.testing.assert_allclose(
        ce.forward(inputs, labels),
        parallel_ce.forward(inputs_scatter, labels),
    )

    # An all-gather is required to combine the results.
    parallel_backward = np.zeros((batch_size, seq_len, vocab_size))
    dist.all_gather(parallel_ce.backward(), parallel_backward)

    np.testing.assert_allclose(
        ce.backward(),
        parallel_backward,
    )
