from mpi4py import MPI


def rank() -> int:
    comm = MPI.COMM_WORLD
    return comm.Get_rank()


def world_size() -> int:
    """Return the world size, i.e. the number of parallel programs."""
    comm = MPI.COMM_WORLD
    return comm.Get_size()
