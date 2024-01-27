import numpy as np

import numpy_distributed as npdist


class AllReduceForward:
    """Perform an all-reduce on the inputs."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """All-reduce the inputs in place and return the tensor."""
        npdist.all_reduce(inputs)
        return inputs

    def backward(self):
        raise NotImplementedError
