from torch.nn import BatchNorm1d, Identity, Linear, Module, Sequential
from torchvision.models import (
    resnet18,
    resnet50,
)


class Extractor(Module):
    def __init__(
            self,
            family: str = 'resnet',
            n_layers: int = 50,
            out_features: int = 512,
        ):
        super().__init__()
        if family == 'resnet':
            if n_layers == 18:
                self.backbone = resnet18(pretrained=False)
            elif n_layers == 50:
                self.backbone = resnet50(pretrained=True)
            self.backbone.fc = Identity()

        self.classifier = Sequential(
            Linear(
                in_features=2048,
                out_features=out_features,
                bias=False,
            ),
            BatchNorm1d(out_features),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    model = Extractor()
    print(model)

