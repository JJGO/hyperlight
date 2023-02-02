from typing import Optional, Tuple, Union

from torch import nn, Tensor

from .voidmodule import VoidModule
from .xparam import ExternalParameter


class _VoidConvNd(VoidModule):

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
        if "bias" not in self._external_params:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)


class VoidConv1d(_VoidConvNd):
    _conv_mod = nn.Conv1d


class VoidConv2d(_VoidConvNd):
    _conv_mod = nn.Conv2d


class VoidConv3d(_VoidConvNd):
    _conv_mod = nn.Conv3d

