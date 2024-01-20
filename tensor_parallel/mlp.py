from typing import Optional

import numpy as np

import numpy_distributed as ndist


class MLP:
    """A multi-layer perceptron layer. We use ReLU instead of GELU
    for simplicity."""
    def __init__(self, d_model: int, d_hidden: Optional[int] = None):
        self.d_model = d_model
        self.d_hidden = d_hidden or d_model*4

    @staticmethod
    def forward(weights: dict, x: np.ndarray) -> np.ndarray:
        """..."""
        y = np.maximum(0, x @ weights["A"])
        z = y @ weights["B"]
        ndist.all_reduce(z)
        return z

    def backward(self, weights: dict):
        """Backward pass through the MLP."""
        raise NotImplementedError

    def init_weights(self, rng):
        """Initiate weights for the MLP. This specific MLP has two
        weight matrices, no bias."""
        return {
            "A": ndist.scatter_init((self.d_model, self.d_hidden), rng, axis=1),
            "B": ndist.scatter_init((self.d_hidden, self.d_model), rng, axis=0),
        }
