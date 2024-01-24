import numpy_distributed as ndist
from numpy_sequential import Attention


class HeadParallelAttention(Attention):
    """A Multi-headed self-Attention (decoder-only) layer. We split the
    weights over multiple devices along the head dimension."""

    def __init__(self, d_model: int, n_heads: int, d_hidden: int, rng):
        ndist.assert_divisible(n_heads)
        super().__init__(
            d_model=d_model,
            n_heads=n_heads // ndist.world_size(),
            d_hidden=d_hidden,
            rng=rng,
        )

    def backward(self):
        """Backward pass through the Attention layer."""
        raise NotImplementedError
