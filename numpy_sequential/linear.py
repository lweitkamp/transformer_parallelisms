import numpy as np

import string

class Linear:
    """A linear layer."""

    # This linear layer is very annoying
    # it should be possible to give it arbitrary shape
    # What is the worst case we got
    # input (B, S, N, M)
    # weight: N, M, D
    # matmul there should be also on N, M x D
    # If the input has 4 dim it has to be that there is a head and hidden section
    # if the input has 3 dim it is only hidden
    # b: batch, s: seq len, d: d_model, h: num heads, m: d_hidden

    def __init__(self, d_model: int, d_hidden: int, rng, dtype=np.float32):
        self.weights = rng.random((d_model, d_hidden)).astype(dtype)
        self.bias = rng.random((d_hidden,)).astype(dtype)

        self.ctx = 
        

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the matrix product x @ W + b."""
        einsum_map = {1: "d", 2: "hm"}

        "...d, d"

        forward = np.einsum("...{einsum_map[dims]}, ", inputs, self.weights)

        forward = inputs @ self.weights + self.bias
        self.ctx["inputs"] = inputs
        return forward

    def backward(self, grads: np.ndarray):
        """Perform a backward pass, calculating the gradients."""
        self.grads["weights"] = np.einsum("bsm, bsd -> md", self.ctx["inputs"], grads)
        self.grads["bias"] = grads.sum(axis=(0, 1))
        self.ctx["inputs"] = None
        return grads @ self.weights.T


class GeneralizedLinear:
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
        
        ascii_options = list(string.ascii_letters)

        # map input dim to chars
        # map output dim to chars
        einsum = "...{x}, {x}{y} -> ...{y}".format(x=..., y=...)
        # x == a or abc or abcd? idk

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.ctx["inputs"] = inputs
        return np.einsum(...) + self.bias
