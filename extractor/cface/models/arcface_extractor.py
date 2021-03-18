import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from cface.models import BaseModule
from cface.models.components import Extractor, Head
from cface.utils import ParametersCounter


class ArcFaceExtractor(BaseModule):
    def __init__(
            self,
            family: str = 'resnet',
            n_layers: int = 18,
            n_hiddens: int = 512,
            n_classes: int = 10,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 3e-4,
            verbose: bool = True,
        ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.verbose = verbose

        self.extractor = Extractor(
            family=family,
            n_layers=n_layers,
            out_features=n_hiddens,
        )
        self.head = Head(
            mode='arc',
            in_features=n_hiddens,
            out_features=n_classes,
        )
        self.criterion = CrossEntropyLoss()

    def forward(
            self,
            x,
        ):
        return self.extractor(x)

    def training_step(
            self,
            batch,
            batch_idx,
            optimizer_idx,
        ):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        embeddings = self.extractor.forward(images)
        angle = self.head.forward(embeddings)

        loss = self.criterion(angle, labels)

        info = {
            'loss': loss,
            'accuracy': accuracy,
        }

        return info

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
    model = ArcFaceExtractor()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(model)
    print(n_params)

