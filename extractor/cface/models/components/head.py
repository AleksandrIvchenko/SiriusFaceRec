from collections import OrderedDict

from torch.nn import (
    Linear,
    Module,
    Sequential,
    Softmax,
)


class Head(Module):
    def __init__(
            self,
            mode: str = 'softmax',
            in_features: int = 512,
            out_features: int = 10,
        ):
        super().__init__()
        head_dict = OrderedDict()
        head_dict['linear'] = Linear(
            in_features=in_features,
            out_features=out_features,
        )
        if mode == 'softmax':
            head_dict['last'] = Softmax()
        elif mode == 'arc':
            head_dict['last'] = Softmax()

        self.head = Sequential(head_dict)

    def forward(self, x):
        return self.head(x)


if __name__ == '__main__':
    model = Head()
    print(model)

