import numpy as np
from mpi4py import MPI

from tensor import scatter_init, gather

from typing import Optional


class MLP:
    """A multi-layer perceptron layer."""
    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model*4

    @staticmethod
    def forward(weights: dict, x: np.ndarray) -> np.ndarray:
        """Multiply x by scattered weights and sum the results."""
        y = x @ weights["A"]
        z = y @ weights["B"]
        out = gather(z, z.shape[0], z.shape[1], op=MPI.SUM)
        return out

    def backward(self):
        """Backward pass through the MLP."""
        raise NotImplementedError

    def init_weights(self, rng):
        """Initiate weights for the MLP. This specific MLP has two
        weight matrices, no bias."""
        return {
            "A": scatter_init(self.d_model, self.d_hidden, rng, axis=1),
            "B": scatter_init(self.d_hidden, self.d_model, rng, axis=0),
        }
