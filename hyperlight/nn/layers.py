from typing import Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor, nn

from .module import HyperModule
from .xparam import ExternalParameter


class HyperLinear(HyperModule):
    """Layer that implements a nn.Linear layer but with external parameters
    that will be predicted by a external hypernetwork"""

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
            self.in_features, self.out_features, "bias" in self._external_parameters
        )


class _HyperConvNd(HyperModule):
    """Layer that implements a nn.ConvNd layer but with external parameters
    that will be predicted by a external hypernetwork"""

    _conv_mod = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Union[int, Tuple[int, ...]] = 0,
        dilation: Union[int, Tuple[int, ...]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        base = self._conv_mod(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.in_channels = base.in_channels
        self.out_channels = base.out_channels
        self.kernel_size = base.kernel_size
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.groups = base.groups
        self.padding_mode = base.padding_mode

        self.weight = ExternalParameter(tuple(base.weight.shape))
        self.bias = ExternalParameter(tuple(base.bias.shape)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_mod._conv_forward(self, input, self.weight, self.bias)

    def extra_repr(self) -> str:
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if "bias" not in self._external_parameterss:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


class HyperConv1d(_HyperConvNd):
    """Layer that implements a nn.Conv1d layer but with external parameters
    that will be predicted by a external hypernetwork"""

    _conv_mod = nn.Conv1d


class HyperConv2d(_HyperConvNd):
    """Layer that implements a nn.Conv2d layer but with external parameters
    that will be predicted by a external hypernetwork"""

    _conv_mod = nn.Conv2d


class HyperConv3d(_HyperConvNd):
    """Layer that implements a nn.Conv3d layer but with external parameters
    that will be predicted by a external hypernetwork"""

    _conv_mod = nn.Conv3d
