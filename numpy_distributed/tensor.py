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
    if ndist.rank() == dst:
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
    gather_list: list[np.ndarray],
    dst: int = 0,
) -> None:
    gather_list = np.concatenate([np.ravel(x) for x in gather_list])
    MPI_COMM.Gatherv(gather_list, tensor, root=dst)


def all_gather(
    tensor_list,
    tensor,
    group=None,
    async_op=False,
):
    return None


def scatter(
    tensor: np.ndarray,
    scatter_list: list[np.ndarray],
    src=0,
) -> None:
    """..."""
    scatter_list = np.concatenate([np.ravel(x) for x in scatter_list])
    MPI_COMM.Scatterv(scatter_list, tensor, root=src)
