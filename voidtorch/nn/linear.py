from torch import Tensor
import torch.nn.functional as F

from .voidmodule import VoidModule
from .xparam import ExternalParameter


class VoidLinear(VoidModule):

    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        assert isinstance(in_features, int) and isinstance(out_features, int)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ExternalParameter(shape=(out_features, in_features))
        if bias:
            self.bias = ExternalParameter(shape=(out_features,))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, "bias" in self._external_params
        )

