import numpy as np
from mpi4py import MPI

import numpy_distributed as ndist

MPI_COMM = MPI.COMM_WORLD


def rank() -> int:
    return MPI_COMM.Get_rank()


def world_size() -> int:
    """Return the world size, i.e. the number of parallel programs."""
    return MPI_COMM.Get_size()


def assert_divisible(dim: int) -> None:
    """A simple assert to check that whatever the size of the
    dimension is, it can be equally divided among devices."""
    assert dim % world_size() == 0, \
        f"Cannot divide the dimension {dim} amongst {world_size()} devices."


def broadcast(
    tensor: np.ndarray,
    src: int = 0,
) -> None:
    """Broadcast tensor to all devices.

    Args:
        tensor (np.ndarray): NumPy array.
        src (int): Source rank from which to broadcast.
    """
    np.copyto(tensor, MPI_COMM.bcast(tensor, root=src))


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
    """Gather data in gather_list and store in tensor, send to dst."""
    gather_list = np.concatenate([np.ravel(x) for x in gather_list])
    MPI_COMM.Gatherv(gather_list, tensor, root=dst)


def all_gather(
    output_tensor,
    tensor_to_gather,
    axis: int = -1
) -> None:
    """Gather data in gather_list and store in tensor, send to all devices.

    TODO: gather is not trivial in ndim>1 so revisit this later. For now,
    it is implemented as an all-reduce with zero padding.
    """
    scatter_size = tensor_to_gather.shape[axis]

    pad_width = [(0, 0)] * len(tensor_to_gather.shape)
    pad_width[axis] = (
        ndist.rank() * scatter_size,
        (ndist.world_size() - 1 - ndist.rank()) * scatter_size,
    )

    z = np.pad(tensor_to_gather, pad_width=pad_width)
    all_reduce(z)
    np.copyto(output_tensor, z)
    # MPI_COMM.Allgatherv(tensor_to_gather.T, output_tensor)


def scatter(
    tensor: np.ndarray,
    scatter_list: list[np.ndarray],
    src=0,
) -> None:
    """..."""
    scatter_list = np.concatenate([np.ravel(x) for x in scatter_list])
    MPI_COMM.Scatterv(scatter_list, tensor, root=src)
