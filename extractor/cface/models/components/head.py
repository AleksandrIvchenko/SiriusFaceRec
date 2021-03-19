from collections import OrderedDict
import math

from torch import BoolTensor, FloatTensor
from torch.nn import (
    Linear,
    Module,
    Parameter,
    Sequential,
    Softmax,
)


class ArcFaceLayer(Module):
    def __init__(
            self,
            in_features: int = 512,
            out_features: int = 10,
            m: float = 0.5,
            s: float = 64,
        ):
        super().__init__()
        self.weights = Parameter(FloatTensor(
            out_features,
            in_features,
        ))
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(
            self,
            embeddings,
            labels,
        ):
        cos_theta = torch.mm(
            self.weight,
            embeddings,
        ).clamp(-1, 1)
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))
        #cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(
            condition=cos_theta>0,
            x=cos_theta_m,
            y=cos_theta,
        )

        mask = BoolTensor(cos_theta.shape[0], cos_theta.shape[1])
        mask.zero_()
        mask.scatter_(
            dim=1,
            index=labels,
            src=1,
        )
        output = torch.where(
            condition=mask,
            x=cos_theta_m,
            y=cos_theta,
        )
        output *= self.s

        return output


class Head(Module):
    def __init__(
            self,
            mode: str = 'arcface',
            in_features: int = 512,
            out_features: int = 10,
            m: float = 0.5,
            s: float = 64,
        ):
        super().__init__()
        head_dict = OrderedDict()
        if mode == 'linear':
            head_dict['linear'] = Linear(
                in_features=in_features,
                out_features=out_features,
            )
        elif mode == 'arcface':
            head_dict['arcface'] = ArcFaceLayer(
                in_features=in_features,
                out_features=out_features,
            )
        self.head = Sequential(head_dict)

    def forward(
            self,
            embeddings,
            labels,
        ):
        return self.head(embeddings, labels)


if __name__ == '__main__':
    model = Head()
    print(model)
