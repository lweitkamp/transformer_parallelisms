import pytest
import numpy as np

from mlp import MLP


@pytest.mark.parametrize("batch_size,d_model,seed", [(3, 4, 42)])
def mlp_test(batch_size: int, d_model: int, seed: int):
    random_state = np.random.default_rng(seed)
    mlp = MLP(d_model=d_model)
    weights = mlp.init_weights(rng=random_state)
    out_all = mlp.forward(weights, random_state.random((batch_size, d_model)))
    print(out_all)
