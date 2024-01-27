import numpy as np

import numpy_sequential as nseq


class MLP(nseq.Layer):
    """Simple Multi-Layered Perceptron with two layers."""

    def __init__(self, d_model: int, d_hidden: int, rng):
        self.w1 = nseq.Linear(d_model=d_model, d_hidden=d_hidden, rng=rng)
        self.w2 = nseq.Linear(d_model=d_hidden, d_hidden=d_model, rng=rng)

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        x = self.w1.forward(inputs_)
        x = np.maximum(0, x)
        x = self.w2.forward(x)
        return x

    def backward(self):
        raise NotImplementedError
