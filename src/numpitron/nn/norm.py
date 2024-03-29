import numpy as np

from numpitron.nn.core import Layer


class LayerNorm(Layer):
    """Layer normalization - normalize the inputs over the last dimension."""

    def __init__(self, d_model: int, rng, dtype=np.float32):
        super().__init__()
        self.add_parameter("weight", (d_model,), dtype, rng=rng)
        self.add_parameter("bias", (d_model,), dtype, dtype, init_fn=np.zeros)

        self.d_model = d_model
        self.eps = 1e-5

        self.ctx: dict = {"input_normalized": None, "std": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Calculate mean and standard deviation of the inputs along the
        last dimension and normalize the inputs. Additionally,
        multiply the normalized input with weights and add a bias."""
        mean = inputs.mean(axis=-1, keepdims=True)
        var = inputs.var(axis=-1, keepdims=True)
        normed = (inputs - mean) / np.sqrt(var + self.eps)

        self.ctx["inputs"] = inputs
        self.ctx["mean"] = mean
        self.ctx["var"] = var

        return self.weight.data * normed + self.bias.data

    def backward(self, grads: np.ndarray) -> np.ndarray:
        """The most straightforward reference is surpisingly from the Triton tutorial
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html."""

        inputs = self.ctx["inputs"]
        mean = self.ctx["mean"]
        var = self.ctx["var"]
        inputs_normed = (inputs - mean) / np.sqrt(var + self.eps)

        self.weight.gradient = np.sum(grads * inputs_normed, axis=(0, 1))
        self.bias.gradient = grads.sum(axis=(0, 1))

        wdy = self.weight.data * grads
        c1 = np.sum(inputs_normed * wdy, axis=-1) / self.d_model
        c2 = wdy.sum(axis=-1) / self.d_model
        grads = (wdy - c1[..., None] * inputs_normed - c2[..., None]) / inputs.std(
            axis=-1, keepdims=True
        )

        return grads
