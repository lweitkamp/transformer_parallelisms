import numpy as np


class LayerNorm:
    """Layer normalization - normalize the inputs over the last dimension."""
    def __init__(self):
        self.weight = ...
        self.bias = ...

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        """Calculate mean and standard deviation of the inputs along the
        last dimension and normalize the inputs. Additionally,
        multiply the normalized input with weights and add a bias."""
        mean = inputs_.mean(dim=-1, keepdims=True)
        var = ((inputs_ - mean) ** 2).mean(dim=-1, keepdims=True)
        std = np.sqrt(var + 1e-05)

        normed = (inputs_ - mean) / std
        return self.weight * normed + self.bias
