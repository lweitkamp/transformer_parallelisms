import numpy as np

from numpitron import nn
from numpitron.parallel import distributed as dist


class ColumnParallelLinear(nn.Linear):
    """A linear layer scattered along the column dimension. If the output
    dimensions is a tuple, scatter it along the last output dim."""

    def __init__(
        self, input_dim: tuple | int, output_dim: tuple | int, rng, dtype=np.float32
    ):
        output_dim = (
            list([output_dim]) if isinstance(output_dim, int) else list(output_dim)
        )
        dist.assert_divisible(output_dim[-1])
        output_dim[-1] = output_dim[-1] // dist.world_size()

        super().__init__(
            input_dim=input_dim, output_dim=tuple(output_dim), rng=rng, dtype=dtype
        )


class RowParallelLinear(nn.Linear):
    """A linear layer scattered along the row dimension. If the input
    dimensions is a tuple, scatter it along the last input dim."""

    def __init__(
        self, input_dim: tuple | int, output_dim: tuple | int, rng, dtype=np.float32
    ):
        input_dim = list([input_dim]) if isinstance(input_dim, int) else list(input_dim)
        dist.assert_divisible(input_dim[-1])
        input_dim[-1] = input_dim[-1] // dist.world_size()

        super().__init__(
            input_dim=tuple(input_dim),
            output_dim=output_dim,
            rng=rng,
            dtype=dtype,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the matrix product x @ W + b. Since the weights are
        scatterd along the row dimension the bias is identical for each
        device. Hence, we only add the bias for a single device.
        In order to make code more readable we inherin the Linear.forward
        method and subtracted the bias from one of the devices."""
        out = super().forward(inputs)
        if dist.rank() != 0:
            out = out - self.bias
        return out
