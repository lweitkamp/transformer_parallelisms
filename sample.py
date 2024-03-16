from pathlib import Path
import argparse

import numpy as np
from numpy.random import Generator

from numpitron import nn, models
from examples.config import load_config


def sample(
    model,
    initial_prompt: str,
    meta: dict,
    max_len: int,
    seq_len: int,
    rng: Generator,
    temperature: float = 1.0,
    top_k: int | None = None,
):
    predicted_text = ""
    tokens = [meta["stoi"][x] for x in initial_prompt]

    for _ in range(max_len):
        tokens = tokens[-seq_len:]
        logits = model(np.asarray([tokens]))
        probabilities = nn.softmax(logits[0, -1] / temperature)

        next_token = np.argmax(rng.multinomial(n=1, pvals=probabilities))
        tokens.append(next_token)

        predicted_text += meta["itos"][next_token]

        print(f"\033[1m{initial_prompt}\033[0m{predicted_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=Path,
        default="examples/single.json",
        help="Path to json config file.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default="examples/model.npy",
        help="Path to json model file.",
    )
    parser.add_argument(
        "--vocab-meta-path",
        type=Path,
        default="examples/meta.pkl",
        help="Path to metadata for vocab.",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default="\n",  # "And cowards it be strawn to my bed,",
        help="Starting prompt.",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)

    rng = np.random.default_rng(config.seed)

    transformer = models.Transformer(
        seq_len=config.seq_len,
        vocab_size=config.vocab_size,
        n_layers=config.n_layers,
        d_model=config.d_model,
        n_heads=config.n_heads,
        rng=rng,
    )

    transformer.load(args.model_path)

    vocab_meta = np.load(args.vocab_meta_path, allow_pickle=True)

    sample(
        transformer,
        args.initial_prompt,
        vocab_meta,
        200,
        config.seq_len,
        rng,
        temperature=1.0,
        top_k=5,
    )
