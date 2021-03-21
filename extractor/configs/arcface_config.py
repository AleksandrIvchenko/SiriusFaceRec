from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


@dataclass
class CommonArguments:
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    neptune_api_token: str = open('./neptune_api_token.txt').readline()[:-1]
    neptune_experiment_name: str = 'arcface'
    neptune_project_name: str = 'sergevkim/arcface'
    seed: int = 9
    verbose: int = 1
    version: str = 'arcface0.1'


@dataclass
class DataArguments:
    batch_size: int = 128
    data_path: Path = Path('./data/celeba')
    learning_rate: float = 3e-4
    num_workers: int = 8
    val_ratio: float = 0.1


@dataclass
class TrainArguments:
    max_epoch: int = 10
    new_size: Tuple[int, int] = (128, 128)
    one_batch_overfit: int = 0
    save_period: int = 1
    scheduler_gamma: float = 0.5
    scheduler_step_size: int = 10


@dataclass
class SpecificArguments:
    n_classes: int = 10178


print(CommonArguments.device)

