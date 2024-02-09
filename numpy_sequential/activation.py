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
        self.axis = axis

        self.ctx: dict = {"inputs": None}
        self.grads: dict = {"weight": None, "bias": None}

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = softmax(inputs)
        self.ctx["outputs"] = np.copy(outputs)
        return outputs

    def backward(self, grads):
        batch_size, n_heads, seq_len, _ = self.ctx["outputs"].shape
        attn = self.ctx["outputs"].reshape(batch_size * n_heads, seq_len, seq_len)
        X = attn * np.diag(np.ones(seq_len))
        Y = np.matmul(attn.transpose(0, 2, 1), attn)
        grads = (X - Y).reshape(batch_size, n_heads, seq_len, seq_len)
        return grads
