import numpy as np
from mpi4py import MPI


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
    if rank() == dst:
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


def scatter(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    axis: int,
    src: int = 0,
) -> None:
    """We scatter the source tensor along an axis and the scattered result
    will be collected in the destination tensor for each process.

    Args:
        source_tensor (np.ndarray): Tensor to scatter along axis.
        destination_tensor (np.ndarray): Tensor from each process to
            collect results.
        axis (int): axis to split source_tensor and scatter the results with.
        src (int): Rank from which we scatter the tensor.
    """
    scatter_list = np.split(source_tensor, world_size(), axis=axis)
    scatter_list = np.concatenate([np.ravel(x) for x in scatter_list])
    MPI_COMM.Scatterv(scatter_list, destination_tensor, root=src)


def all_gather(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    axis: int = -1,
) -> None:
    """Gather source tensors from each process and collect it in the
    destination tensor.

    MPI sends a contiguous stream of bytes from each process. To ensure
    the expected shape is returned in destination tensor, we collect
    the contiguous stream per process and reshape each accordingly.

    Args:
        source_tensor (np.ndarray): Source tensor for each process.
        destination_tensor (np.ndarray): Tensor to gather the results.
        axis (int): The axis on which the tensor needs to be concatenated.
    """
    receiving_buffer = np.empty(np.prod(destination_tensor.shape))
    MPI_COMM.Allgather(source_tensor, receiving_buffer)
    receiving_buffer = np.split(receiving_buffer, world_size(), axis)
    receiving_buffer = np.concatenate(
        [x.reshape(source_tensor.shape) for x in receiving_buffer],
        axis=-1,
    )
    np.copyto(destination_tensor, receiving_buffer)


def reduce_scatter(
    source_tensor: np.ndarray,
    destination_tensor: np.ndarray,
    op: MPI.Op = MPI.SUM,
) -> None:
    """Reduce source tensor to root process and scatter the reduction
    back to all processes.

    Again the issue with contiguous data streams - but I could not
    find a workaround for MPI_COMM.Reduce_scatter so I resorted to using
    the `reduce` and `scatter` operations I defined earlier.

    Args:
        source_tensor (np.ndarray): Source tensor for each process.
        destination_tensor (np.ndarray): Tensor to gather the results.
        op (MPI.Op): Operation to reduce the tensor.
    """
    # Maybe soon:
    # MPI_COMM.Reduce_scatter(source_tensor, destination_tensor, op=op)
    reduce(source_tensor, dst=0, op=op)
    scatter(source_tensor, destination_tensor, axis=-1, src=0)
