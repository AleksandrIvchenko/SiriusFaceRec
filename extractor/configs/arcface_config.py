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
    version: str = 'arcface0.2'


@dataclass
class DataArguments:
    batch_size: int = 128
    data_path: Path = Path('./data/celeba')
    new_size: Tuple[int, int] = (128, 128)
    num_workers: int = 8
    val_ratio: float = 0.1


@dataclass
class TrainArguments:
    learning_rate: float = 0.1
    max_epoch: int = 100
    momentum: float = 0.9
    new_size: Tuple[int, int] = (128, 128)
    one_batch_overfit: int = 0
    save_period: int = 5
    scheduler_gamma: float = 0.5
    scheduler_step_size: int = 10
    weight_decay: float = 5e-4


@dataclass
class SpecificArguments:
    n_classes: int = 10178


print(CommonArguments.device)

