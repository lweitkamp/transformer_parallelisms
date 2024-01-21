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
    """A linear layer scattered along the columns."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        self.weight = rng.random((d_model, d_hidden // ndist.world_size()))
        self.bias = rng.random((d_hidden // ndist.world_size(), ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.weight @ x + self.bias

    def backward(self):
        raise NotImplementedError


class RowParallelLinear:
    """A linear layer scattered along the rows."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        self.weight = rng.random((d_model // ndist.world_size(), d_hidden))
        self.bias = rng.random((d_hidden, ))

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x @ self.weight

        if ndist.rank() == 0:
            out = out + self.bias

        return out

    def backward(self):
        raise NotImplementedError
