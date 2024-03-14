from pathlib import Path
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

    def save(self, path: Path):
        save_state = {}

        save_state["embedding"] = {
            "velocity": self.state[0][0].velocity,
            "momentum": self.state[0][0].momentum,
        }

        for i in range(2, len(self.state) - 1):
            save_state[f"layer_{i}"] = {
                "attention": {
                    "q": {
                        0: {
                            "velocity": self.state[i][0][0][0].velocity,
                            "momentum": self.state[i][0][0][0].momentum,
                        },
                        1: {
                            "velocity": self.state[i][0][0][1].velocity,
                            "momentum": self.state[i][0][0][1].momentum,
                        },
                    },
                    "k": {
                        0: {
                            "velocity": self.state[i][0][1][0].velocity,
                            "momentum": self.state[i][0][1][0].momentum,
                        },
                        1: {
                            "velocity": self.state[i][0][1][1].velocity,
                            "momentum": self.state[i][0][1][1].momentum,
                        },
                    },
                    "v": {
                        0: {
                            "velocity": self.state[i][0][2][0].velocity,
                            "momentum": self.state[i][0][2][0].momentum,
                        },
                        1: {
                            "velocity": self.state[i][0][2][1].velocity,
                            "momentum": self.state[i][0][2][1].momentum,
                        },
                    },
                    "o": {
                        0: {
                            "velocity": self.state[i][0][3][0].velocity,
                            "momentum": self.state[i][0][3][0].momentum,
                        },
                        1: {
                            "velocity": self.state[i][0][3][1].velocity,
                            "momentum": self.state[i][0][3][1].momentum,
                        },
                    },
                },

                "norm1": {
                    0: {
                        "velocity": self.state[i][1][0].velocity,
                        "momentum": self.state[i][1][0].momentum,
                    },
                    1: {
                        "velocity": self.state[i][1][1].velocity,
                        "momentum": self.state[i][1][1].momentum,
                    },
                },

                "mlp": {
                    "linear1": {
                        0: {
                            "velocity": self.state[i][2][0][0].velocity,
                            "momentum": self.state[i][2][0][0].momentum,
                        },
                        1: {
                            "velocity": self.state[i][2][0][1].velocity,
                            "momentum": self.state[i][2][0][1].momentum,
                        },
                    },
                    "linear2": {
                        0: {
                            "velocity": self.state[i][2][2][0].velocity,
                            "momentum": self.state[i][2][2][0].momentum,
                        },
                        1: {
                            "velocity": self.state[i][2][2][1].velocity,
                            "momentum": self.state[i][2][2][1].momentum,
                        },
                    },
                },

                "norm2": {
                    0: {
                        "velocity": self.state[i][3][0].velocity,
                        "momentum": self.state[i][3][0].momentum,
                    },
                    1: {
                        "velocity": self.state[i][3][1].velocity,
                        "momentum": self.state[i][3][1].momentum,
                    },
                },
            }

        np.save(path, save_state, allow_pickle=True)

    def load(self, path: Path):
        save_state = np.load(path, allow_pickle=True)[()]

        self.state[0][0].velocity = save_state["embedding"]["velocity"]
        self.state[0][0].momentum = save_state["embedding"]["momentum"]

        for i in range(2, len(self.state) - 1):
            layer = save_state[f"layer_{i}"]
            self.state[i][0][0][0].velocity = layer["attention"]["q"][0]["velocity"]
            self.state[i][0][0][0].momentum = layer["attention"]["q"][0]["momentum"]
            self.state[i][0][0][1].velocity = layer["attention"]["q"][1]["velocity"]
            self.state[i][0][0][1].momentum = layer["attention"]["q"][1]["momentum"]

            self.state[i][0][1][0].velocity = layer["attention"]["k"][0]["velocity"]
            self.state[i][0][1][0].momentum = layer["attention"]["k"][0]["momentum"]
            self.state[i][0][1][1].velocity = layer["attention"]["k"][1]["velocity"]
            self.state[i][0][1][1].momentum = layer["attention"]["k"][1]["momentum"]

            self.state[i][0][2][0].velocity = layer["attention"]["v"][0]["velocity"]
            self.state[i][0][2][0].momentum = layer["attention"]["v"][0]["momentum"]
            self.state[i][0][2][1].velocity = layer["attention"]["v"][1]["velocity"]
            self.state[i][0][2][1].momentum = layer["attention"]["v"][1]["momentum"]

            self.state[i][0][3][0].velocity = layer["attention"]["o"][0]["velocity"]
            self.state[i][0][3][0].momentum = layer["attention"]["o"][0]["momentum"]
            self.state[i][0][3][1].velocity = layer["attention"]["o"][1]["velocity"]
            self.state[i][0][3][1].momentum = layer["attention"]["o"][1]["momentum"]

            self.state[i][1][0].velocity = layer["norm1"][0]["velocity"]
            self.state[i][1][0].momentum = layer["norm1"][0]["momentum"]
            self.state[i][1][1].velocity = layer["norm1"][1]["velocity"]
            self.state[i][1][1].momentum = layer["norm1"][1]["momentum"]

            self.state[i][2][0][0].velocity = layer["mlp"]["linear1"][0]["velocity"]
            self.state[i][2][0][0].momentum = layer["mlp"]["linear1"][0]["momentum"]
            self.state[i][2][0][1].velocity = layer["mlp"]["linear1"][1]["velocity"]
            self.state[i][2][0][1].momentum = layer["mlp"]["linear1"][1]["momentum"]

            self.state[i][2][2][0].velocity = layer["mlp"]["linear2"][0]["velocity"]
            self.state[i][2][2][0].momentum = layer["mlp"]["linear2"][0]["momentum"]
            self.state[i][2][2][1].velocity = layer["mlp"]["linear2"][1]["velocity"]
            self.state[i][2][2][1].momentum = layer["mlp"]["linear2"][1]["momentum"]

            self.state[i][3][0].velocity = layer["norm2"][0]["velocity"]
            self.state[i][3][0].momentum = layer["norm2"][0]["momentum"]
            self.state[i][3][1].velocity = layer["norm2"][1]["velocity"]
            self.state[i][3][1].momentum = layer["norm2"][1]["momentum"]

