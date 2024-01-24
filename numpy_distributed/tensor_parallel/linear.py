import numpy as np

import numpy_distributed as ndist
from numpy_sequential import Linear


class ColumnParallelLinear(Linear):
    """A linear layer scattered along the column dimension."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        ndist.assert_divisible(d_hidden)
        super().__init__(
            d_model=d_model,
            d_hidden=d_hidden // ndist.world_size(),
            rng=rng,
        )

    def backward(self):
        raise NotImplementedError


class RowParallelLinear(Linear):
    """A linear layer scattered along the row dimension."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        ndist.assert_divisible(d_model)
        super().__init__(
            d_model=d_model // ndist.world_size(),
            d_hidden=d_hidden,
            rng=rng,
        )

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        """Compute the matrix product x @ W + b. Since the weights are
        scatterd along the row dimension the bias is identical for each
        device. Hence, we only add the bias for a single device.
        
        Alternatively, we could have inherited the Linear.forward method
        and subtracted the bias from one of the devices.
        """
        out = inputs_ @ self.weight

        if ndist.rank() == 0:
            out = out + self.bias

        return out

    def backward(self):
        raise NotImplementedError
