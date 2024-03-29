import numpy as np

import string
from numpitron.nn.core import Layer


class Linear(Layer):
    """Linear layer mimicking the DenseGeneral from flax - flexible in and out axes."""

    def __init__(
        self,
        input_dim: tuple | int,
        output_dim: tuple | int,
        rng,
        dtype=np.float32,
    ):
        super().__init__()
        input_dim = tuple([input_dim]) if isinstance(input_dim, int) else input_dim
        output_dim = tuple([output_dim]) if isinstance(output_dim, int) else output_dim

        self.add_parameter("weight", input_dim + output_dim, dtype, rng=rng)
        self.add_parameter("bias", output_dim, dtype, dtype, init_fn=np.zeros)

        self.ctx: dict = {"inputs": None}

        # format the einsums for this layer.
        ascii_options = list(string.ascii_letters)
        self.in_chr = "".join(ascii_options.pop() for _ in range(len(input_dim)))
        self.out_chr = "".join(ascii_options.pop() for _ in range(len(output_dim)))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.ctx["inputs"] = np.copy(inputs)
        outputs = np.einsum(
            f"...{self.in_chr}, ...{self.in_chr}{self.out_chr} -> ...{self.out_chr}",
            inputs,
            self.weight.data,
        )
        return outputs + self.bias.data

    def backward(self, grads: np.ndarray):
        """Perform a backward pass, calculating the gradients."""
        weight_gradient = np.einsum(
            f"...{self.in_chr}, ...{self.out_chr} -> ...{self.in_chr}{self.out_chr}",
            self.ctx["inputs"],
            grads,
        )

        self.weight.gradient = weight_gradient.sum(axis=(0, 1))
        self.bias.gradient = grads.sum(axis=(0, 1))

        grads = np.einsum(
            f"...{self.out_chr}, {self.in_chr}{self.out_chr} -> ...{self.in_chr}",
            grads,
            self.weight.data,
        )
        self.ctx["inputs"] = None
        return grads
