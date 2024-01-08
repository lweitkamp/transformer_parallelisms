import numpy as np
from mpi4py import MPI


from world_info import get_rank, get_world_size


def scatter(
    source: np.ndarray,
    d_in: int,
    d_out: int,
    axis: int,
    dtype: str = 'int',
) -> None:
    """Scatter 'source' to 'destination'."""
    comm = MPI.COMM_WORLD

    d_in, d_out = (
        d_in // get_world_size() if axis == 0 else d_in,
        d_out // get_world_size() if axis == 1 else d_out,
    )

    recvbuf = np.empty((d_in, d_out), dtype=dtype)
    comm.Scatterv(source, recvbuf, root=0)
    return recvbuf


def gather(
    scattered_source: np.ndarray,
    d_in: int,
    d_out: int,
    op,
    dtype: str = 'int',
) -> np.ndarray:
    comm = MPI.COMM_WORLD
    gathered_result = np.empty((d_in, d_out), dtype=dtype)
    comm.Reduce(scattered_source, gathered_result, op=op, root=0)

    if get_rank() == 0:
        return gathered_result

    return np.zeros_like(gathered_result, dtype=dtype)


class MLP:
    """A multi-layer perceptron layer"""
    def __init__(self, d_model: int):
        self.weights_a = self.init_weights(
            d_in=d_model,
            d_out=d_model*4,
            scatter_axis=1,
        )
        self.weights_b = self.init_weights(
            d_in=d_model*4,
            d_out=d_model,
            scatter_axis=0,
        )

    def init_weights(
        self,
        d_in: int,
        d_out: int,
        scatter_axis: int,
    ) -> np.ndarray:
        """Initiate weights of size `d_in` by `d_out` and scatter on parallel
        devices on axis `scatter_axis`."""
        assert scatter_axis in (0, 1), \
            "Can only scatter on first or second axis."
        
        weight = None
        if get_rank() == 0:
            weight = np.arange(d_in*d_out).reshape((d_in, d_out))

        scatter_weight = scatter(weight, d_in, d_out, scatter_axis)
        return scatter_weight

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Multiply x by scattered weights and sum the results."""
        y = x @ self.weights_a
        z = y @ self.weights_b
        out = gather(z, z.shape[0], z.shape[1], op=MPI.SUM)
        return out


if __name__ == "__main__":
    mlp = MLP(d_model=4)
    out_all = mlp.forward(np.arange(3 * 4).reshape((3, 4)))
    if get_rank() == 0:
        print(out_all)
