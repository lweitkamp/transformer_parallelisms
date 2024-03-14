from pathlib import Path
import numpy as np


from numpitron import nn
from numpitron.nn.core import Block


class Transformer(Block):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        rng,
        dtype=np.float32,
    ):
        """..."""
        super().__init__()

        self.layers.extend(
            [
                nn.InputEmbedding(d_model, vocab_size, rng),
                nn.PositionalEmbedding(d_model, seq_len),
            ]
        )
        self.layers.extend(
            [
                nn.TransformerBlock(d_model, n_heads, rng, dtype)
                for _ in range(n_layers)
            ]
        )
        self.layers.append(nn.OutputEmbedding(self.layers[0].weight))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def backward(self, grads: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            grads = layer.backward(grads)
        return grads

    def save(self, path: Path):
        # save embeddings
        state = {
            "embedding": self.layers[0].weight.data,
        }

        # Transformer blocks
        for i, layer in enumerate(self.layers[2:-1]):
            state[f"layer_{i}"] = {
                "attention": {
                    "q_proj_weight": layer.attention.q_proj.weight.data,
                    "q_proj_bias": layer.attention.q_proj.bias.data,
                    "k_proj_weight": layer.attention.k_proj.weight.data,
                    "k_proj_bias": layer.attention.k_proj.bias.data,
                    "v_proj_weight": layer.attention.v_proj.weight.data,
                    "v_proj_bias": layer.attention.v_proj.bias.data,
                    "out_proj_weight": layer.attention.out_proj.weight.data,
                    "out_proj_bias": layer.attention.out_proj.bias.data,
                },
                "norm1": {
                    "weight": layer.norm1.weight.data,
                    "bias": layer.norm1.bias.data,
                },
                "mlp": {
                    "linear1_weight": layer.mlp.layers[0].weight.data,
                    "linear1_bias": layer.mlp.layers[0].bias.data,
                    "linear2_weight": layer.mlp.layers[2].weight.data,
                    "linear2_bias": layer.mlp.layers[2].bias.data,
                },
                "norm2": {
                    "weight": layer.norm2.weight.data,
                    "bias": layer.norm2.bias.data,
                }
            }
        np.save(path, state, allow_pickle=True)

    def load(self, path: Path):
        state = np.load(path, allow_pickle=True)[()]
        self.layers[0].weight.data = state["embedding"]
        self.layers[-1].weight.data = state["embedding"]

        for i, layer in enumerate(self.layers[2:-1]):
            params = state[f"layer_{i}"]

            layer.attention.q_proj.weight.data = params["attention"]["q_proj_weight"]
            layer.attention.q_proj.bias.data = params["attention"]["q_proj_bias"]
            layer.attention.k_proj.weight.data = params["attention"]["k_proj_weight"]
            layer.attention.k_proj.bias.data = params["attention"]["k_proj_bias"]
            layer.attention.v_proj.weight.data = params["attention"]["v_proj_weight"]
            layer.attention.v_proj.bias.data = params["attention"]["v_proj_bias"]
            layer.attention.out_proj.weight.data = params["attention"]["out_proj_weight"]
            layer.attention.out_proj.bias.data = params["attention"]["out_proj_bias"]

            layer.norm1.weight.data = params["norm1"]["weight"]
            layer.norm1.bias.data = params["norm1"]["bias"]

            layer.mlp.layers[0].weight.data = params["mlp"]["linear1_weight"]
            layer.mlp.layers[0].bias.data = params["mlp"]["linear1_bias"]
            layer.mlp.layers[2].weight.data = params["mlp"]["linear2_weight"]
            layer.mlp.layers[2].bias.data = params["mlp"]["linear2_bias"]

            layer.norm2.weight.data = params["norm2"]["weight"]
            layer.norm2.bias.data = params["norm2"]["bias"]
