import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

from cface.models import BaseModule
from cface.utils import ParametersCounter


class ProtostarModel(BaseModule):
    def __init__(
            self,
            device: torch.device,
            learning_rate: float,
            scheduler_step_size: int,
            scheduler_gamma: float,
            verbose: bool,
        ):
        super().__init__()

    def forward(
            self,
            x,
        ):
        pass

    def training_step(
            self,
            batch,
            batch_idx,
            optimizer_idx,
        ):
        pass

    def validation_step(
            self,
            batch,
            batch_idx,
        ):
        loss = self.training_step(
            batch=batch,
            batch_idx=batch_idx,
            optimizer_idx=0,
        )

        return loss

    def configure_optimizers(
            self,
        ):
        optimizer = Adam(
            params=self.parameters(),
            lr=self.learning_rate,
        )

        return [optimizer], []


if __name__ == '__main__':
    model = ProtostarModel()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(n_params)

