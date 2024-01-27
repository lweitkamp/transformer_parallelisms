import numpy as np

import numpy_distributed as npdist
import tensor_parallel as tp


class MegatronLM:
    """Create a Megatron-LM style transformer."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        d_hidden: int,
        n_heads: int,
        vocab_size: int,
        rng,
    ) -> None:
        self.input_embedding = tp.VocabParallelInputEmbedding(
            d_model,
            vocab_size,
            rng,
        )

        self.transformer_blocks = [
            {
                "attention": tp.HeadParallelAttention(d_model, n_heads, d_hidden, rng),
                "mlp": tp.TensorParallelMLP(d_model, d_hidden, rng),
            }
            for _ in range(n_layers)
        ]

        self.output_embedding = tp.VocabParallelOutputEmbedding(
            weights=self.input_embedding.e,
        )

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs = self.input_embedding.forward(inputs)

        for block in self.transformer_blocks:
            # Attention part
            x_ = inputs
            inputs = block["attention"].forward(layer_norm.forward(inputs))
            npdist.all_reduce(inputs)
            inputs += x_

            x_ = inputs
            inputs = block["mlp"].forward(layer_norm.forward(inputs))
            npdist.all_reduce(inputs)

        output_embedding = self.output_embedding.forward(inputs)
        return output_embedding
