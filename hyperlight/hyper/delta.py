from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
from torch import nn, Tensor

from .base import HyperNet
from ...nn.init import initialize_weight, initialize_bias


@dataclass(eq=False, repr=False)
class DeltaHyperNet(HyperNet):

    factor: float = 1
    main_init_distribution: str = "kaiming_normal"
    main_init_bias: Union[float, str] = 0.0
    main_requires_grad: bool = True
    main_nonlinearity: Optional[str] = "LeakyReLU"

    def __post_init__(self):
        super().__post_init__(reset=False)

        self.params = nn.ParameterDict()
        for name, shape in self.output_sizes.items():
            self.params[name.replace(".", ":")] = nn.Parameter(torch.Tensor(*shape))

        if not self.main_requires_grad:
            self.grad_main(False)

        self.reset_parameters()

    def forward(self, **inputs: Dict[str, Tensor]):
        outputs = super().forward(**inputs)
        for name in self.output_sizes:
            outputs[name] = (
                self.params[name.replace(".", ":")] + self.factor * outputs[name]
            )
        return outputs

    def grad_main(self, requires_grad=True):
        if not self.main_requires_grad:
            for param in self.params.values():
                param.requires_grad = requires_grad

    def reset_parameters(self):
        super().reset_parameters()
        if self.main_requires_grad:
            for name, param in self.params.items():
                if "weight" in name:
                    initialize_weight(
                        param,
                        self.main_init_distribution,
                        nonlinearity=self.main_nonlinearity,
                    )
                if "bias" in name:
                    initialize_bias(
                        param,
                        self.main_init_bias,
                        nonlinearity=self.main_nonlinearity,
                    )

