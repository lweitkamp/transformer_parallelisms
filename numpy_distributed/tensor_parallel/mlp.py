import numpy as np

import numpy_distributed as ndist
from numpy_sequential import MLP
from numpy_distributed.tensor_parallel.linear import (
    RowParallelLinear,
    ColumnParallelLinear,
)


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
