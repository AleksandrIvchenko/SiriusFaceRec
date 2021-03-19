from collections import OrderedDict
import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import BoolTensor, FloatTensor, LongTensor
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
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.out_features = out_features
        self.weights = Parameter(FloatTensor(
            in_features,
            out_features,
        ))
        nn.init.xavier_uniform_(self.weights)

    def forward(
            self,
            embeddings,
            labels,
        ):
        cos_theta = torch.mm(
            F.normalize(embeddings),
            F.normalize(self.weights),
        ).clamp(-1, 1)
        sin_theta = torch.sqrt(1 - torch.pow(cos_theta, 2)).clamp(0, 1)
        #cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        cos_theta_m = torch.where(
            cos_theta>0,
            cos_theta_m,
            cos_theta,
        )
        one_hot = torch.zeros(
            cos_theta.size(),
            device='cuda',
        )
        one_hot.scatter_(
            1,
            labels.view(-1, 1).long(),
            1,
        )
        output = (
            one_hot * cos_theta_m +
            (1.0 - one_hot) * cos_theta
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
