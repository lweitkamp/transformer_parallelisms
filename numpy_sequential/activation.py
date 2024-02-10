import numpy as np


class ReLU:
    def __init__(self):
        self.ctx: dict = {"inputs": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.ctx["inputs"] = np.copy(inputs)
        inputs = np.maximum(0, inputs)
        return inputs

    def backward(self, grads):
        grads = (self.ctx["inputs"] > 0) * grads
        self.ctx["inputs"] = None
        return grads


def softmax(inputs: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the softmax of x along the given axis."""
    x_ = np.exp(inputs - np.max(inputs, axis=axis, keepdims=True))
    return x_ / x_.sum(axis=axis, keepdims=True)


class Softmax:
    def __init__(self, axis: int = -1):
        assert axis == -1, "no support for any other axis right now."
        self.axis = axis

        self.ctx: dict = {"inputs": None}
        self.grads: dict = {"weight": None, "bias": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = softmax(inputs)
        self.ctx["inputs_sm"] = np.copy(outputs)
        return outputs

    def backward(self, grads: np.ndarray):
        inputs_sm = self.ctx["inputs_sm"]
        _, _, seq_len, _ = inputs_sm.shape

        left = np.einsum("...ij, jk -> ...ijk", inputs_sm, np.eye(seq_len))
        right = np.einsum("...ij, ...ik -> ...ijk", inputs_sm, inputs_sm)
        grads = (grads[..., None, :] @ (left - right)).reshape(inputs_sm.shape)
        return grads
