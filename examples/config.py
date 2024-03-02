import dataclasses
import json
from pathlib import Path


@dataclasses.dataclass
class Config:
    batch_size: int
    num_epochs: int

    learning_rate: float
    beta0: float
    beta1: float
    seed: int = 42

def load_config(config_path: Path | str) -> Config:
    config = json.load(Path(config_path).open(mode="r"))
    return Config(**config)
