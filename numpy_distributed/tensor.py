import numpy as np
from mpi4py import MPI

import numpy_distributed as ndist

MPI_COMM = MPI.COMM_WORLD


def broadcast(
    tensor: np.ndarray,
    src: int = 0,
) -> np.ndarray:
    """Broadcast tensor to all devices.

    Args:
        tensor (np.ndarray): NumPy array.
        src (int): Source rank from which to broadcast.
    """
    return MPI_COMM.bcast(tensor, root=src)


def reduce(
    tensor: np.ndarray,
    dst: int = 0,
    op: MPI.Op = MPI.SUM,
) -> None:
    """Reduce tensor across all devices and broadcast the result
    back to a single device.

    Args:
        tensor (np.ndarray): NumPy array.
        dst (int): Rank on which we gather the reduction.
        op (MPI.Op): Operation to reduce the tensor.
    """
    if ndist.get_rank() == dst:
        MPI_COMM.Reduce(MPI.IN_PLACE, tensor, op=op, root=dst)
    else:
        MPI_COMM.Reduce(tensor, None, op=op, root=dst)


def all_reduce(
    tensor: np.ndarray,
    op: MPI.Op = MPI.SUM,
) -> None:
    """Reduce tensor across all devices and broadcast the result
    back to all devices.

    Args:
        tensor (np.ndarray): NumPy array.
        op (MPI.Op): Operation to reduce the tensor.
    """
    MPI_COMM.Allreduce(MPI.IN_PLACE, tensor, op=op)


def gather(
    tensor: np.ndarray,
    gather_list=None,
    dst: int = 0,
    group=None,
    async_op=False,
) -> None:
    return None


def all_gather(
    tensor_list,
    tensor,
    group=None,
    async_op=False,
):
    return None


def scatter(
    tensor: np.ndarray,
    scatter_list=None,
    src=0,
    group=None,
    async_op=False,
) -> None:
    comm = MPI.COMM_WORLD
    return None


# TO DELETE #####

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
    if ndist.get_rank() == 0:
        data = rng.random(shape, dtype=dtype)
        arrs = np.split(data, ndist.get_world_size(), axis=axis)
        raveled = [np.ravel(arr) for arr in arrs]

        # Join them back up into a 1D array
        tensor = np.concatenate(raveled)

    shape = [
        dim // ndist.get_world_size() if i == axis else dim
        for i, dim in enumerate(shape)
    ]

    tensor_scattered = np.empty(shape, dtype=dtype)
    comm.Scatterv(tensor, tensor_scattered, root=0)
    return tensor_scattered