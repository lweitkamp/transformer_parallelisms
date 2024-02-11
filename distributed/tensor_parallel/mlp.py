import numpy as np

import distributed as npdist
from layers import MLP, ReLU


class TensorParallelMLP(MLP):
    """Megatron-LM style Multi-Layered Perceptron.

    This module should be followed by an all-reduce layer.
    """

    def __init__(self, d_model: int, d_hidden: int, rng, dtype=np.float32):
        npdist.assert_divisible(d_model)
        npdist.assert_divisible(d_hidden)
        super().__init__(d_model, d_hidden, rng)

        self.layers = [
            npdist.ColumnParallelLinear(d_model, d_hidden, rng, dtype),
            ReLU(),
            npdist.RowParallelLinear(d_hidden, d_model, rng, dtype),
        ]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = super().forward(inputs)
        npdist.all_reduce(x)
        return x
