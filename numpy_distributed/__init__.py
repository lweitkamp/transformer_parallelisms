from numpy_distributed.general import get_rank, get_world_size
from numpy_distributed.tensor import (
    reduce,
    all_reduce,
    scatter_init,
    broadcast,
)
