import numpy as np

import string


class Linear:
    """Generalized Linear Layer."""

    ctx: dict = {"inputs": None}
    grads: dict = {"weight": None, "bias": None}

    def __init__(
        self, input_dim: tuple | int, output_dim: tuple | int, rng, dtype=np.float32
    ):
        input_dim = tuple([input_dim]) if isinstance(input_dim, int) else input_dim
        output_dim = tuple([output_dim]) if isinstance(output_dim, int) else output_dim

        self.weight = rng.random(size=input_dim + output_dim, dtype=dtype)
        self.bias = np.zeros(output_dim, dtype=dtype)

        # format the einsums for this layer.
        ascii_options = list(string.ascii_letters)
        self.in_chr = "".join(ascii_options.pop() for _ in range(len(input_dim)))
        self.out_chr = "".join(ascii_options.pop() for _ in range(len(output_dim)))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.ctx["inputs"] = inputs
        outputs = np.einsum(
            f"...{self.in_chr}, ...{self.in_chr}{self.out_chr} -> ...{self.out_chr}",
            inputs,
            self.weight,
        )
        return outputs + self.bias

    def backward(self, grads: np.ndarray):
        """Perform a backward pass, calculating the gradients."""
        sum_dims = tuple([i for i in range(grads.ndim - len(self.in_chr))])

        weight_gradient = np.einsum(
            f"...{self.in_chr}, ...{self.out_chr} -> ...{self.in_chr}{self.out_chr}",
            self.ctx["inputs"],
            grads,
        ).sum(axis=sum_dims)
        bias_gradient = grads.sum(axis=sum_dims)

        self.grads["weight"] = weight_gradient
        self.grads["bias"] = bias_gradient

        self.ctx["inputs"] = None
        grads = np.einsum(
            f"...{self.out_chr}, {self.out_chr}{self.in_chr} -> ...{self.in_chr}",
            grads,
            self.weight.T,
        )
        return grads
