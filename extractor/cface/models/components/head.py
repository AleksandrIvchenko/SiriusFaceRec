from collections import OrderedDict
import math

import torch
import torch.nn as nn
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
            in_features,
            out_features,
        ))
        nn.init.xavier_uniform_(self.weights)
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

    def forward(
            self,
            embeddings,
            labels,
        ):
        print(self.weights.shape, embeddings.shape)
        cos_theta = torch.mm(
            embeddings,
            self.weights,
        ).clamp(-1, 1)
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2))
        #cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(
            condition=cos_theta>0,
            x=cos_theta_m,
            y=cos_theta,
        )

        mask = BoolTensor(len(labels), self.n_classes)
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
        if mode == 'linear':
            self.head = Linear(
                in_features=in_features,
                out_features=out_features,
            )
        if mode == 'arcface':
            self.head = ArcFaceLayer(
                in_features=in_features,
                out_features=out_features,
            )

    def forward(
            self,
            embeddings,
            labels,
        ):
        return self.head(embeddings, labels)


if __name__ == '__main__':
    model = Head()
    print(model)
