import numpy as np

import numpy_distributed as ndist


class Linear:
    """A linear layer."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        self.weight = rng.random((d_model, d_hidden))
        self.bias = rng.random((d_hidden, ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias

    def backward(self):
        raise NotImplementedError


class ColumnParallelLinear:
    """A linear layer scattered along the column dimension."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        ndist.assert_divisible(d_hidden)
        self.weight = rng.random((d_model, d_hidden // ndist.world_size()))
        self.bias = rng.random((d_hidden // ndist.world_size(), ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the matrix product x @ W + b. The bias is also scattered
        and hence we have to add it for every device."""
        return x @ self.weight + self.bias

    def backward(self):
        raise NotImplementedError


class RowParallelLinear:
    """A linear layer scattered along the row dimension."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        ndist.assert_divisible(d_model)
        self.weight = rng.random((d_model // ndist.world_size(), d_hidden))
        self.bias = rng.random((d_hidden, ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute the matrix product x @ W + b. Since the weights are
        scatterd along the row dimension the bias is identical for each
        device. Hence, we only add the bias for a single device."""
        out = x @ self.weight

        if ndist.rank() == 0:
            out = out + self.bias

        return out

    def backward(self):
        raise NotImplementedError
