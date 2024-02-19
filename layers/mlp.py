import numpy as np

import layers
from layers.core import Block


class MLP(Block):
    """Simple Multi-Layered Perceptron with two layers."""

    def __init__(self, d_model: int, d_hidden: int, rng, dtype=np.float32):
        super().__init__()

        self.layers.extend(
            [
                layers.Linear(d_model, d_hidden, rng=rng, dtype=dtype),
                layers.ReLU(),
                layers.Linear(d_hidden, d_model, rng=rng, dtype=dtype),
            ]
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        return grads
