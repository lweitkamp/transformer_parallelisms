import numpy as np

import distributed as npdist
from layers import Attention


class HeadParallelAttention(Attention):
    """A Multi-headed self-Attention (decoder-only) layer. We split the
    weights over multiple devices along the head dimension."""

    def __init__(self, d_model: int, n_heads: int, d_hidden: int, rng):
        npdist.assert_divisible(n_heads)
        super().__init__(
            d_model=d_model,
            n_heads=n_heads // npdist.world_size(),
            d_hidden=d_hidden,
            rng=rng,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """..."""
        # f(x) -->
        x = super().forward(inputs)

        # g(x) -->
        npdist.all_reduce(x)
        return x

    def backward(self, grads: np.ndarray):
        """Backward pass through the Attention layer."""
        # g(x) -->
        grads = super().backward(grads)

        # f(x) -->
        npdist.all_reduce(np.ascontiguousarray(grads))
        return grads
