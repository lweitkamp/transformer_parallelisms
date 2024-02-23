""""""
from pathlib import Path

import numpy as np

from data.tokenizer import BPETokenizer


class DataLoader:
    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer: BPETokenizer,
        rng,
        seq_len: int,
        batch_size: int,
    ):
        """

        Args:
            dataset_path (str | Path): Path to tokenized data.
        """
        self.tokenizer = tokenizer

        # Convert to numpy integers.
        self.data = np.asarray(
            [
                int(x)
                for x in Path(dataset_path)
                .open(mode="r", encoding="utf-8")
                .read()
                .strip()
                .split(" ")
            ],
            dtype=int,
        )
        self.rng = rng

        # Calculate how quickly we go through one epoch.
        self.batches_per_epoch = len(self.data) / (seq_len * batch_size)

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seq_range = np.arange(seq_len)

    def iter_epoch(self):
        batch_idx = 0
        while batch_idx <= self.batches_per_epoch:
            start_idx = self.rng.integers(
                0, len(self.data) - 1 - self.seq_len, size=self.batch_size
            )[:, None]
            inputs = self.data[start_idx + self.seq_range]
            labels = self.data[start_idx + 1 + self.seq_range]
            yield inputs, labels

            batch_idx += 1
