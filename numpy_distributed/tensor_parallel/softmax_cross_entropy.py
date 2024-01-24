import numpy as np

import numpy_distributed as ndist


class ParallelSoftmaxCrossEntropy:
    """Softmax Cross-entropy loss of logits."""

    def forward(self, inputs_: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Calculate the cross-entropy loss given logits (`inputs_`).

        Logits are parallelized along the vocab dim. To avoid large
        communication of the vocab dim, keep this distributed and calculate
        the cross entropy loss.

        Arguments:
            logits (B, S, V_): A batch (B) of logits S for parallelized vocab
                chunk V_.
            labels (B, S, 1): A dense set of labels for each batch & sequence.

        Returns:
            Cross-entropy loss.
        """
        batch_size, seq_len, vocab_chunk_shape = inputs_.shape

        # Figure out token valid range for this specific embedding chunk.
        chunk_start = ndist.rank() * vocab_chunk_shape
        chunk_end = chunk_start + vocab_chunk_shape
        mask = np.logical_or(labels < chunk_start, labels >= chunk_end)

        # Set tokens to chunk range, mask tokens outside range.
        labels_masked = labels - chunk_start
        labels_masked[mask] = 0.0

        # Look up the predicted logit value for correct labels.
        y_pred = inputs_.reshape(-1, vocab_chunk_shape)[
            np.arange(batch_size * seq_len),
            labels_masked.reshape(-1),
        ].reshape(labels_masked.shape)

        # Mask the predictions outside the valid range.
        y_pred[mask] = 0.0

        # We need to all-reduce these predictions.
        # todo: check if the numpy lookup/reshape is a view
        # or a copy. (Seems like a copy from the strides).
        return
