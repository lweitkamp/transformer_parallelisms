import numpy as np

import numpy_distributed as npdist
from numpy_sequential import MLP


class TensorParallelMLP(MLP):
    """Megatron-LM style Multi-Layered Perceptron.

    This module should be followed by an all-reduce layer.
    """

    def __init__(self, d_model: int, d_hidden: int, rng):
        npdist.assert_divisible(d_model)
        npdist.assert_divisible(d_hidden)
        super().__init__(d_model, d_hidden, rng)

        self.w1 = npdist.ColumnParallelLinear(
            d_model=d_model,
            d_hidden=d_hidden,
            rng=rng,
        )
        self.w2 = npdist.RowParallelLinear(
            d_model=d_hidden,
            d_hidden=d_model,
            rng=rng,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = super().forward(inputs)
        npdist.all_reduce(x)
        return x

    def backward(self):
        raise NotImplementedError
