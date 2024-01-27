from abc import abstractmethod, ABC


import numpy as np


class Layer(ABC):
    """Abstract layer class."""

    @abstractmethod
    def forward(self, inputs_: np.ndarray, **kwargs):
        """Forward pass through the layer."""
        ...

    @abstractmethod
    def backward(self, grad: np.ndarray):
        """Backward pass through the layer."""
        ...

    def __call__(self, inputs_: np.ndarray, **kwargs):
        """Overwrite the call method."""
        return self.forward(inputs_, **kwargs)
