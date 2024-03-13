import numpy as np
from dataclasses import dataclass

from numpitron.nn.core import Parameter


@dataclass
class ParamStatistics:
    parameter: Parameter
    velocity: np.ndarray | None = None
    momentum: np.ndarray | None = None

    def __post_init__(self):
        self.velocity = np.zeros_like(self.parameter.data)
        self.momentum = np.zeros_like(self.parameter.data)


class Adam:
    def __init__(
        self,
        model,
        loss_fn: callable,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ):
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.model = model
        self.state = self.init(model.parameters())

        self.timestep = 0
        self.betas = betas
        self.eps = eps

    def init(self, model_parameters):
        optimizer_state = []

        for layer in model_parameters:
            if isinstance(layer, Parameter):
                optimizer_state.append(ParamStatistics(layer))
            elif isinstance(layer, list):
                optimizer_state.append(self.init(layer))

        return optimizer_state

    def update(self):
        def _update(state):
            if isinstance(state, list):
                for s in state:
                    _update(s)
                return

            assert isinstance(state, ParamStatistics)
            b1, b2 = self.betas
            gradient = state.parameter.gradient

            state.momentum = b1 * state.momentum + (1 - b1) * gradient
            state.velocity = b2 * state.velocity + (1 - b2) * np.power(gradient, 2)

            momentum = state.momentum / (1 - b1**self.timestep)
            velocity = state.velocity / (1 - b2**self.timestep)
            update = self.learning_rate * momentum / (np.sqrt(velocity) + self.eps)
            state.parameter.data = state.parameter.data - update

        _update(self.state)

    def step(self, inputs: np.ndarray, labels: np.ndarray) -> dict:
        self.timestep += 1

        # Forward pass & calculate the loss
        logits = self.model.forward(inputs)
        loss = self.loss_fn.forward(logits, labels)

        # Backward pass and update
        self.model.backward(self.loss_fn.backward())
        self.update()
        self.model.zero_grad()

        return {
            "loss": loss,
        }
