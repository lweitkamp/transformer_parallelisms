import numpy as np
from mpi4py import MPI

import numpy_distributed as npdist

MPI_COMM = MPI.COMM_WORLD


def rank() -> int:
    return MPI_COMM.Get_rank()


def world_size() -> int:
    """Return the world size, i.e. the number of parallel programs."""
    return MPI_COMM.Get_size()


def assert_divisible(dim: int) -> None:
    """A simple assert to check that whatever the size of the
    dimension is, it can be equally divided among devices."""
    assert (
        dim % world_size() == 0
    ), f"Cannot divide the dimension {dim} amongst {world_size()} devices."


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
    """Reduce tensor across all processes and broadcast the result
    back to a single process.

    Args:
        tensor (np.ndarray): NumPy array.
        dst (int): Rank on which we gather the reduction.
        op (MPI.Op): Operation to reduce the tensor.
    """
    if npdist.rank() == dst:
        MPI_COMM.Reduce(MPI.IN_PLACE, tensor, op=op, root=dst)
    else:
        MPI_COMM.Reduce(tensor, None, op=op, root=dst)


def all_reduce(
    tensor: np.ndarray,
    op: MPI.Op = MPI.SUM,
) -> None:
    """Reduce tensor across all processes and broadcast the result
    back to all processes.

    Args:
        tensor (np.ndarray): NumPy array.
        op (MPI.Op): Operation to reduce the tensor.
    """
    MPI_COMM.Allreduce(MPI.IN_PLACE, tensor, op=op)


def all_gather(output_tensor, tensor_to_gather, axis: int = -1) -> None:
    """Gather data in gather_list and store in tensor, send to all devices.

    TODO: gather is not trivial in ndim>1 so revisit this later. For now,
    it is implemented as an all-reduce with zero padding.
    """
    scatter_size = tensor_to_gather.shape[axis]

    pad_width = [(0, 0)] * len(tensor_to_gather.shape)
    pad_width[axis] = (
        npdist.rank() * scatter_size,
        (npdist.world_size() - 1 - npdist.rank()) * scatter_size,
    )

    z = np.pad(tensor_to_gather, pad_width=pad_width)
    all_reduce(z)
    np.copyto(output_tensor, z)
    # MPI_COMM.Allgatherv(tensor_to_gather.T, output_tensor)


def scatter(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    axis: int,
    src: int = 0,
) -> None:
    """We scatter the source tensor along an axis and the scattered result
    will be collected in the destination tensor for each process.

    Args:
        source_tensor (np.ndarray): NumPy array that collects the scattered result.
        destination_tensor (np.ndarray): List of tensors to scatter.
        axis (int): axis to split source_tensor and scatter the results with.
        src (int): Rank from which we scatter the tensor.
    """
    scatter_list = np.split(source_tensor, world_size(), axis=axis)
    scatter_list = np.concatenate([np.ravel(x) for x in scatter_list])
    MPI_COMM.Scatterv(scatter_list, destination_tensor, root=src)
