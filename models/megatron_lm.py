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

        self.transformer_blocks = [{
            "attention": tp.HeadParallelAttention(d_model, n_heads, d_hidden, rng),
            "mlp": tp.TensorParallelMLP(d_model, d_hidden, rng),
        } for _ in range(n_layers)]

        self.output_embedding = tp.VocabParallelOutputEmbedding(
            weights=self.input_embedding.e,
        )

    def forward(self, inputs_: np.ndarray) -> np.ndarray:
        inputs_ = self.input_embedding.forward(inputs_)

        for block in self.transformer_blocks:
            # Attention part
            x_ = inputs_
            inputs_ = block["attention"].forward(layer_norm.forward(inputs_))
            npdist.all_reduce(inputs_)
            inputs_ += x_

            x_ = inputs_
            inputs_ = block["mlp"].forward(layer_norm.forward(inputs_))
            npdist.all_reduce(inputs_)

        output_embedding = self.output_embedding.forward(inputs_)
        return output_embedding
