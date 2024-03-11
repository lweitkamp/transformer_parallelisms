from pathlib import Path
import argparse

import numpy as np

from numpitron import nn, optimizers, models, data
from examples.config import Config, load_config


def train(config: Config):
    rng = np.random.default_rng(config.seed)

    train_dataloader = data.DataLoader(
        dataset_path=config.dataset_train_path,
        rng=rng,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
    )

    transformer = models.Transformer(
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
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

    for epoch in range(config.num_epochs):
        for step, (input_data, labels) in enumerate(train_dataloader.iter_epoch()):
            out = optimizer.step(input_data, labels)
            loss = out["loss"].mean()
            print(f"{epoch} : {step}/{train_dataloader.batches_per_epoch} - {loss:.3f}")

    print("....")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=Path,
        required=False,
        default="examples/single.json",
        help="Path to json config file.",
    )
    args = parser.parse_args()
    train(load_config(args.config_file))
