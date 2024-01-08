import numpy as np
from mpi4py import MPI

from tensor import scatter_init, gather
from world_info import get_rank

from typing import Optional


class MLP:
    """A multi-layer perceptron layer"""
    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        name: str = "MLP",
    ):
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model*4
        self.name = name

    def init_weights(self, rng):
        return {
            "A": scatter_init(self.d_model, self.d_hidden, rng, axis=1),
            "B": scatter_init(self.d_hidden, self.d_model, rng, axis=0),
        }

    def forward(self, weights: dict, x: np.ndarray) -> np.ndarray:
        """Multiply x by scattered weights and sum the results."""
        y = x @ weights["A"]
        z = y @ weights["B"]
        out = gather(z, z.shape[0], z.shape[1], op=MPI.SUM)
        return out


if __name__ == "__main__":
    mlp = MLP(d_model=4)
    weights = mlp.init_weights(rng=np.random.default_rng(42))
    out_all = mlp.forward(weights, np.arange(3 * 4).reshape((3, 4)))
    if get_rank() == 0:
        print(out_all)
