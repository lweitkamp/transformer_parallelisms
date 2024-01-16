import numpy as np
from mpi4py import MPI

from world_utils.world_info import get_rank, get_world_size


def broadcast_init(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    dtype: str = 'float',
) -> np.ndarray:
    """Initiate a tensor with a given shape and broadcast it
    to all devices.

    Args:
        shape (tuple[int, ...]): Desired tensor shape.
        rng (np.random.Generator): NumPy random state.
        dtype (str): desired data type.

    Returns:
        A tensor that is broadcasted to all devices.
    """
    comm = MPI.COMM_WORLD
    x = rng.random(shape, dtype=dtype) if get_rank() == 0 else None
    x = comm.bcast(x, root=0)
    return x


def scatter_init(
    shape: tuple[int, ...],
    rng: np.random.Generator,
    axis: int,
    dtype: str = 'float',
) -> np.ndarray:
    """Initiate a tensor with a given shape and scatter it
    to all devices split on `axis`.

    Args:
        shape (tuple[int, ...]): Desired tensor shape.
        rng (np.random.Generator): NumPy random state.
        dtype (str): desired data type.

    Returns:
        A tensor that is broadcasted to all devices.
    """
    comm = MPI.COMM_WORLD

    tensor = None
    if get_rank() == 0:
        data = rng.random(shape, dtype=dtype)
        arrs = np.split(data, get_world_size(), axis=axis)
        raveled = [np.ravel(arr) for arr in arrs]

        # Join them back up into a 1D array
        tensor = np.concatenate(raveled)

    shape = [
        dim // get_world_size() if i == axis else dim
        for i, dim in enumerate(shape)
    ]

    tensor_scattered = np.empty(shape, dtype=dtype)
    comm.Scatterv(tensor, tensor_scattered, root=0)
    return tensor_scattered


def all_reduce(
    scattered_source: np.ndarray,
    reduction: MPI.Op,
    dtype: str | np.dtype = 'float',
) -> np.ndarray:
    comm = MPI.COMM_WORLD
    gathered_result = np.empty_like(scattered_source, dtype=dtype)
    comm.Reduce(scattered_source, gathered_result, op=reduction, root=0)

    if get_rank() == 0:
        return gathered_result

    return np.zeros_like(gathered_result, dtype=dtype)
