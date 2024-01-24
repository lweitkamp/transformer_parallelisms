import numpy as np

from numpy_sequential import Linear


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