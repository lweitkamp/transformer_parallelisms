import numpy as np

import numpy_distributed as ndist
from tensor_parallel.linear import (
    Linear,
    RowParallelLinear,
    ColumnParallelLinear,
)


class MLP:
    """Simple Multi-Layered Perceptron with two layers."""
    def __init__(self, d_model: int, d_hidden: int, rng):
        self.w1 = Linear(d_model=d_model, d_hidden=d_hidden, rng=rng)
        self.w2 = Linear(d_model=d_hidden, d_hidden=d_model, rng=rng)

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        x = self.w1.forward(inputs_)
        x = np.maximum(0, x)
        x = self.w2.forward(x)
        return x
    
    def backward(self):
        raise NotImplementedError


class TensorParallelMLP(MLP):
    """Megatron-LM style Multi-Layered Perceptron.
    
    This module should be followed by an all-reduce layer.
    """

    def __init__(self, d_model: int, d_hidden: int, rng):
        ndist.assert_divisible(d_model)
        ndist.assert_divisible(d_hidden)
        super().__init__(d_model, d_hidden, rng)

        self.w1 = ColumnParallelLinear(
            d_model=d_model,
            d_hidden=d_hidden,
            rng=rng,
        )
        self.w2 = RowParallelLinear(
            d_model=d_hidden,
            d_hidden=d_model,
            rng=rng,
        )
        
    def backward(self):
        raise NotImplementedError
