from mpi4py import MPI


def rank() -> int:
    comm = MPI.COMM_WORLD
    return comm.Get_rank()


def world_size() -> int:
    """Return the world size, i.e. the number of parallel programs."""
    comm = MPI.COMM_WORLD
    return comm.Get_size()


def assert_divisible(dim: int) -> None:
    """A simple assert to check that whatever the size of the
    dimension is, it can be equally divided among devices."""
    assert dim % world_size() == 0, \
        f"Cannot divide the dimension {dim} amongst {world_size()} devices."
