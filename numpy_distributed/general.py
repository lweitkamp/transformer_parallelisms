from mpi4py import MPI


def get_rank() -> int:
    comm = MPI.COMM_WORLD
    return comm.Get_rank()


def get_world_size() -> int:
    """Return the world size, i.e. the number of parallel programs."""
    comm = MPI.COMM_WORLD
    return comm.Get_size()
