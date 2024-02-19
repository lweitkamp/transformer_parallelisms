import numpy as np


class Layer:
    def __init__(self):
        self.grads: dict = {}

    def expose(self):
        """Returns the layer and the weight.

        # Linear:
        self.expose()[0] == (Linear, "weight")
        self.expose()[1] == (Linear, "bias")
        # So we can access Linear.weight, Linear.grads["weight"] --> all we need
        """
        return [(self, key) for key in self.grads]

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)


class Block(Layer):
    def __init__(self):
        self.layers: list = []

    def expose(self):
        """A block has layers that have weights, but no weights itself. These
        layers are added to self.list. Blocks cannot be defined recursively."""
        data = []
        for layer in self.layers:
            data.append(layer.expose())
        return data
    