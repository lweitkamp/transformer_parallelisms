import numpy as np
import pytest

from numpitron.parallel import distributed as dist


@pytest.mark.parametrize(
    "batch_size,seq_len,d_model",
    [(1, 2, 4), (2, 4, 8)],
)
def test_broadcast(
    batch_size: int,
    seq_len: int,
    d_model: int,
) -> None:
    """Create a tensor on root, broadcast it, assert
    all ranks have the same tensor afterwards."""
    expected_tensor = np.arange((batch_size * seq_len * d_model)).reshape(
        batch_size, seq_len, d_model
    )

    tensor = (
        np.empty((batch_size, seq_len, d_model))
        if dist.rank()
        else np.copy(expected_tensor)
    )

    dist.broadcast(tensor)

    np.testing.assert_array_equal(tensor, expected_tensor)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_reduce(
    batch_size: int,
    seq_len: int,
) -> None:
    """Create an empty tensor and fill specific rank indices with ones.
    After reduction, only the root rank should have a sum equal to
    the entire tensor shape, all others should have a sum equal to whatever
    they filled in."""
    world_size = dist.world_size()
    rank = dist.rank()

    tensor = np.zeros((batch_size, seq_len, world_size))
    tensor[..., rank] = 1.0

    dist.reduce(tensor)

    if rank == 0:
        np.testing.assert_equal(tensor.sum(), batch_size * seq_len * world_size)
    else:
        np.testing.assert_equal(tensor.sum(), batch_size * seq_len)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_all_reduce(
    batch_size: int,
    seq_len: int,
) -> None:
    """Create an empty tensor and fill specific rank indices with ones.
    After all-reduce, all ranks should have a sum equal to
    the entire tensor shape."""
    world_size = dist.world_size()

    tensor = np.zeros((batch_size, seq_len, world_size))
    tensor[..., dist.rank()] = 1.0

    dist.all_reduce(tensor)

    np.testing.assert_equal(tensor.sum(), batch_size * seq_len * world_size)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_scatter(
    batch_size: int,
    seq_len: int,
) -> None:
    """Create a zeros tensor and fill it with ones on root. Scatter it to
    all processes and ensure the sum is expected."""
    world_size = dist.world_size()
    source_tensor = np.zeros((batch_size, seq_len, world_size))
    destination_tensor = np.zeros((batch_size, seq_len, 1))

    if dist.rank() == 0:
        source_tensor = source_tensor + 1.0

    dist.scatter(source_tensor, destination_tensor, axis=-1)

    np.testing.assert_equal(destination_tensor.sum(), batch_size * seq_len)


@pytest.mark.parametrize(
    "batch_size,seq_len",
    [(1, 2), (2, 4)],
)
def test_all_gather(
    batch_size: int,
    seq_len: int,
) -> None:
    """Each process creates a tensor with their rank as value. We all-gather
    the result to all processes. To test if the gather was successful, each
    slice sent by a process should have unique values."""
    world_size = dist.world_size()

    destination_tensor = np.zeros((batch_size, seq_len, world_size))
    source_tensor = np.zeros((batch_size, seq_len, 1)) + dist.rank()

    dist.all_gather(source_tensor, destination_tensor, axis=-1)

    assert set(np.unique(destination_tensor)) == set(range(world_size))
