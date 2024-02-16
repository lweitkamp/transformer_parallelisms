import numpy as np


class LayerNorm:
    """Layer normalization - normalize the inputs over the last dimension."""

    def __init__(self, d_model: int, rng, dtype=np.float32):
        self.weight = rng.random((d_model,), dtype=dtype)
        self.bias = np.zeros((d_model,), dtype=dtype)

        self.ctx: dict = {"input_normalized": None}
        self.grads: dict = {"weight": None, "bias": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate mean and standard deviation of the inputs along the
        last dimension and normalize the inputs. Additionally,
        multiply the normalized input with weights and add a bias."""
        mean = inputs.mean(axis=-1, keepdims=True)
        var = ((inputs - mean) ** 2).mean(axis=-1, keepdims=True)
        std = np.sqrt(var + 1e-05)

        normed = (inputs - mean) / std

        self.ctx["input_normalized"] = normed

        return self.weight * normed + self.bias

    def backward(self, grads: np.ndarray) -> np.ndarray:
        self.grads["bias"] = grads.sum(axis=(0, 1))
        self.grads["weight"] = np.sum(grads * self.ctx["input_normalized"], axis=(0, 1))

        self.ctx["input_normalized"] = None

        # Modify grads

        return grads
