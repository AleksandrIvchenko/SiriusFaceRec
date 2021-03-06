import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from cface.models import BaseModule
from cface.models.components import Extractor, Head
from cface.utils import ParametersCounter


class ArcFaceExtractor(BaseModule):
    def __init__(
            self,
            extractor_family: str = 'resnet',
            head_mode: str = 'arcface',
            n_layers: int = 50,
            n_hiddens: int = 512,
            n_classes: int = 10,
            device: torch.device = torch.device('cpu'),
            learning_rate: float = 0.1,
            momentum: float = 0.9,
            weight_decay: float = 5e-4,
            verbose: bool = True,
        ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.extractor = Extractor(
            family=extractor_family,
            n_layers=n_layers,
            out_features=n_hiddens,
        )
        self.head = Head(
            mode=head_mode,
            in_features=n_hiddens,
            out_features=n_classes,
        )
        self.criterion = CrossEntropyLoss()

    def forward(
            self,
            x,
        ):
        return self.extractor(x)

    def predict(
            self,
            image,
        ):
        batch = torch.tensor(image)
        return self.extractor(image)

    def training_step(
            self,
            batch,
            batch_idx,
            optimizer_idx,
        ):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        embeddings = self.extractor(images)
        outputs = self.head(embeddings, labels)
        predictions = torch.max(outputs, dim=1).values

        loss = self.criterion(outputs, labels)
        accuracy = (predictions == labels).sum().item() / len(batch)

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
        optimizer = SGD(
            params=self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        def lr_step_func(epoch):
            if epoch < -1:
                return ((epoch + 1) / 5) ** 2
            else:
                return 0.1 ** len([m for m in [8, 14] if m - 1 <= epoch])
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lr_step_func,
        ) 

        return [optimizer], [scheduler]


if __name__ == '__main__':
    model = ArcFaceExtractor()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    print(model)
    print(n_params)

