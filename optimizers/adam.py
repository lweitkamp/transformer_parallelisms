import numpy as np
import itertools


class Adam:
    def __init__(self, model, loss_fn: callable, learning_rate: float):
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.model = model

        self.parameters = list(itertools.chain.from_iterable(model.expose()))

    def update(self, norm: int):
        for layer, tensor_name in self.parameters:
            grads = layer.grads[tensor_name]
            weight = getattr(layer, tensor_name)
            update = weight + self.learning_rate * grads / norm
            setattr(layer, tensor_name, update)

    def step(self, inputs: np.ndarray, labels: np.ndarray) -> dict:
        batch_size, seq_len, *_ = inputs.shape

        # Forward pass & calculate the loss
        logits = self.model.forward(inputs)
        loss = self.loss_fn.forward(logits, labels)

        # Backward pass and update
        self.model.backward(self.loss_fn.backward())
        self.update(batch_size * seq_len)

        return {
            "loss": loss,
        }
