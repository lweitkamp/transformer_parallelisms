import numpy as np
import itertools


class Adam:
    def __init__(
        self,
        model,
        loss_fn: callable,
        learning_rate: float,
        betas=(0.9, 0.999),
        eps=1e-08,
    ):
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.model = model

        self.parameters = [
            (
                layer,
                tensor_name,
                np.zeros_like(getattr(layer, tensor_name)),  # velocity
                np.zeros_like(getattr(layer, tensor_name)),  # momentum
            )
            for layer, tensor_name in list(
                itertools.chain.from_iterable(model.expose())
            )
        ]
        self.timestep = 0
        self.betas = betas
        self.eps = eps

    def update(self, norm: int):
        self.timestep += 1

        b1, b2 = self.betas

        for layer, tensor_name, velocity, momentum in self.parameters:
            grads = layer.grads[tensor_name]
            weight = getattr(layer, tensor_name)

            # in-place update of momentum and velocity
            np.add(b1 * momentum, (1 - b1) * grads, out=momentum)
            np.add(b2 * velocity, (1 - b2) * np.power(grads, 2), out=velocity)

            m_hat = momentum / (1 - b1 ** self.timestep)
            v_hat = velocity / (1 - b2 ** self.timestep)
            update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            # TODO: we essentially sum up the grads for batch and seq dim, shouldn't
            # we avg it at some point?
            setattr(layer, tensor_name,  weight - update)

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
