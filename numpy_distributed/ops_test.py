import numpy as np
import pytest

import numpy_distributed as npdist


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
        if npdist.rank()
        else np.copy(expected_tensor)
    )

    npdist.broadcast(tensor)

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
    world_size = npdist.world_size()
    rank = npdist.rank()

    tensor = np.zeros((batch_size, seq_len, world_size))
    tensor[..., rank] = 1.0

    npdist.reduce(tensor)

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
    world_size = npdist.world_size()

    tensor = np.zeros((batch_size, seq_len, world_size))
    tensor[..., npdist.rank()] = 1.0

    npdist.all_reduce(tensor)

    np.testing.assert_equal(tensor.sum(), batch_size * seq_len * world_size)
