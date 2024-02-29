import numpy as np

from numpitron import nn
from numpitron.parallel import tensor_parallel, distributed as dist


class TensorParallelMLP(nn.MLP):
    """Megatron-LM style Multi-Layered Perceptron.

    This module should be followed by an all-reduce layer.
    """

    def __init__(self, d_model: int, d_hidden: int, rng, dtype=np.float32):
        dist.assert_divisible(d_model)
        dist.assert_divisible(d_hidden)
        super().__init__(d_model, d_hidden, rng)

        self.layers = [
            tensor_parallel.ColumnParallelLinear(d_model, d_hidden, rng, dtype),
            nn.ReLU(),
            tensor_parallel.RowParallelLinear(d_hidden, d_model, rng, dtype),
        ]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """A forward pass through the MLP. Since the layers themselves
        are already parallel, here we only need to ensure the all-reduce
        is performed after the forward pass."""

        # f(x) -->
        x = super().forward(inputs)

        # g(x) -->
        dist.all_reduce(x)
        return x

    def backward(self, grads: np.ndarray) -> np.ndarray:
        # g(x) -->
        grads = super().backward(grads)

        # f(x) -->
        dist.all_reduce(grads)
        return grads
