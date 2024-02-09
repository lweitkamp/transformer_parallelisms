import numpy as np

import numpy_sequential as nseq


class MLP:
    """Simple Multi-Layered Perceptron with two layers."""

    def __init__(self, d_model: int, d_hidden: int, rng, dtype=np.float32):
        self.layers = [
            nseq.Linear(d_model, d_hidden, rng=rng, dtype=dtype),
            nseq.ReLU(),
            nseq.Linear(d_hidden, d_model, rng=rng, dtype=dtype),
        ]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        return grads
