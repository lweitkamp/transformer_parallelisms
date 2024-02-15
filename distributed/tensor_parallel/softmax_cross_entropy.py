import numpy as np

import distributed as npdist


class ParallelSoftmaxCrossEntropy:
    """Softmax Cross-entropy loss of logits."""

    def __init__(self):
        self.ctx: dict = {"softmax": None, "mask": None, "labels_masked": None}

    def forward(self, inputs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate the cross-entropy loss given logits (`inputs`).

        Logits are parallelized along the vocab dim. To avoid large
        communication of the vocab dim, we communicate both the logits
        of shape (B, S) and the sum of the exponents of vocab predictions.
        That's 2 communications in total, it will be efficient only if the
        vocab dim is large.

        Arguments:
            logits (B, S, V_): A batch (B) of logits S for parallelized vocab
                chunk V_.
            labels (B, S, 1): A dense set of labels for each batch & sequence.

        Returns:
            Cross-entropy loss.
        """
        batch_size, seq_len, vocab_chunk_shape = inputs.shape

        # Reshape to make our life easier.
        inputs = inputs.reshape(batch_size * seq_len, vocab_chunk_shape)
        labels = labels.reshape(batch_size * seq_len)

        # Figure out token valid range for this specific embedding chunk.
        chunk_start = npdist.rank() * vocab_chunk_shape
        chunk_end = chunk_start + vocab_chunk_shape
        mask = np.logical_or(labels < chunk_start, labels >= chunk_end)

        # Set labels to chunk range, mask inputs and labels outside range.
        labels = labels - chunk_start
        labels[mask] = 0

        # Gather logits, mask them and communicate the (B, S) values.
        logits = inputs[np.arange(batch_size * seq_len), labels]
        logits[mask] = 0
        npdist.all_reduce(logits)

        # Calculate log-sum-exp for the CE loss. Here we first calculate
        # sum-exp and communicate the (B, S) values.
        exp_inputs = np.exp(inputs)
        sum_exp = np.sum(exp_inputs, axis=-1)
        npdist.all_reduce(sum_exp)

        cross_entropy = -logits + np.log(sum_exp)
        cross_entropy = cross_entropy.reshape((batch_size, seq_len))

        self.ctx["softmax"] = (exp_inputs / np.expand_dims(sum_exp, -1)).reshape(
            batch_size, seq_len, vocab_chunk_shape
        )
        self.ctx["mask"] = mask.reshape(batch_size, seq_len)
        self.ctx["labels_masked"] = labels.reshape(batch_size, seq_len)

        return cross_entropy

    def backward(self):
        """Backwards pass of the softmax-ce loss."""
        softmax = self.ctx["softmax"]
        mask = self.ctx["mask"]
        labels_masked = self.ctx["labels_masked"]

        batch_size, seq_len, vocab_chunk_shape = softmax.shape

        softmax = softmax.reshape(batch_size * seq_len, vocab_chunk_shape)
        labels_masked = labels_masked.reshape(batch_size * seq_len)
        mask = mask.reshape(batch_size * seq_len)

        softmax[np.arange(batch_size * seq_len), labels_masked] -= 1 - mask

        # Clear cache.
        self.ctx["softmax"] = None
        self.ctx["mask"] = None
        self.ctx["labels_masked"] = None

        return softmax.reshape(batch_size, seq_len, vocab_chunk_shape)
