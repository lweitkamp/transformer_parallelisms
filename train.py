from pathlib import Path
import models
import data

import numpy as np


def train(num_epoch: int = 100, seq_len: int = 32, batch_size: int = 16):
    rng = np.random.default_rng(42)
    tokenizer = data.BPETokenizer.from_saved("data/tokenizer.model")
    dataloader = data.DataLoader(
        dataset_path=Path("data") / "shakespeare.tokens",
        tokenizer=tokenizer,
        rng=rng,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    # transformer = models.Transformer(
    #     seq_len=128,
    #     vocab_size=512,
    #     n_layers=2,
    #     d_model=256,
    #     n_heads=8,
    #     rng=rng,
    # )

    for epoch in range(num_epoch):
        for input_data, labels in dataloader.iter_epoch():
            ...


if __name__ == "__main__":
    train()
