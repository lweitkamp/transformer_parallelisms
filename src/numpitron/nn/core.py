import numpy as np
from dataclasses import dataclass


@dataclass
class Parameter:
    data: np.ndarray
    gradient: None | np.ndarray = None

    def __post_init__(self):
        self.gradient = np.zeros_like(self.data)


class Layer:
    def __init__(self) -> None:
        self.params: list[Parameter] = []

    def parameters(self) -> list[Parameter]:
        return self.params

    def add_parameter(self, name: str, shape, dtype, rng=None, init_fn=None) -> None:
        assert rng or init_fn
        init_fn = init_fn or rng.random
        setattr(self, name, Parameter((init_fn(shape) * 0.02).astype(dtype)))
        self.params.append(getattr(self, name))

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.gradient = np.zeros_like(parameter.gradient)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        return self.forward(*args, **kwargs)


class Block(Layer):
    "_sequential_ block of compute."

    def __init__(self) -> None:
        super().__init__()
        self.layers: list = []

    def parameters(self):
        return [layer.parameters() for layer in self.layers]

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
