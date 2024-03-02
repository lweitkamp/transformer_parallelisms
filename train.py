from pathlib import Path
import argparse

import numpy as np

from numpitron import nn, optimizers, models
from examples.config import Config, load_config
from data import DataLoader


def train(config: Config):
    rng = np.random.default_rng(config.seed)

    train_dataloader = DataLoader(
        dataset_path=config.dataset_train_path,
        tokenizer=config.tokenizer,
        rng=rng,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
    )

    transformer = models.Transformer(
        seq_len=config.seq_len,
        vocab_size=len(config.vocab),
        n_layers=config.n_layers,
        d_model=config.d_model,
        n_heads=config.n_heads,
        rng=rng,
    )

    optimizer = optimizers.Adam(
        transformer,
        nn.SoftmaxCrossEntropy(),
        learning_rate=config.learning_rate,
        betas=config.betas,
    )

    for epoch in range(config.num_epoch):
        for step, (input_data, labels) in enumerate(train_dataloader.iter_epoch()):
            out = optimizer.step(input_data, labels)
            print(f"{epoch} {step} {out['loss'].mean()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=Path, required=True, help="Path to json config file."
    )
    args = parser.parse_args()
    train(load_config(args.config_file))
