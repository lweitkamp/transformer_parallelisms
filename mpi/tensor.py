import numpy as np
from mpi4py import MPI

from world_info import get_rank, get_world_size


def scatter_init(
    d_in: int,
    d_out: int,
    rng: np.random.Generator,
    axis: int,
    dtype: str = 'float',
) -> np.ndarray:
    """Initiate a tensor and scatter it across an axis."""
    comm = MPI.COMM_WORLD

    tensor = None
    if get_rank() == 0:
        data = rng.random((d_in, d_out))
        arrs = np.split(data, get_world_size(), axis=axis)
        raveled = [np.ravel(arr) for arr in arrs]

        # Join them back up into a 1D array
        tensor = np.concatenate(raveled)

    d_in, d_out = (
        d_in // get_world_size() if axis == 0 else d_in,
        d_out // get_world_size() if axis == 1 else d_out,
    )

    tensor_scattered = np.empty((d_in, d_out), dtype=dtype)
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


def broadcast(input_tensor: np.ndarray):
    """Broadcast an array"""
    comm = MPI.COMM_WORLD
    return comm.bcast(input_tensor, root=0)
