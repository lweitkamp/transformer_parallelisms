import numpy as np

import numpy_distributed as npdist


class AllReduceForward:
    """Perform an all-reduce on the inputs."""

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        """All-reduce the inputs in place and return the tensor."""
        npdist.all_reduce(inputs_)
        return inputs_

    def backward(self):
        raise NotImplementedError
