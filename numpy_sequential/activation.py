import numpy as np


class ReLU:
    ctx: dict = {"inputs": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.ctx["inputs"] = inputs
        inputs = np.maximum(0, inputs)
        return inputs

    def backward(self, grads):
        grads = (self.ctx["inputs"] > 0) * grads
        self.ctx["inputs"] = None
        return grads
