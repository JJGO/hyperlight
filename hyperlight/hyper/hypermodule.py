import copy
from typing import Dict, Any, Union

import torch
from torch import nn


JsonDict = Dict[str, Any]


class HyperModule(nn.Module):

    ZOO = {}

    def __init__(self):
        super().__init__()
        self._parametrized = False
        self.input_params = nn.ParameterDict()

    def build_hypernet(self, input_sizes, output_sizes, **hypernet_kws):
        hypernet_kws = copy.deepcopy(hypernet_kws)

        hypernet_model = self.ZOO[hypernet_kws.pop("model", "HyperNet")]
        self.hypernet = hypernet_model(
            input_sizes=input_sizes, output_sizes=output_sizes, **hypernet_kws
        )

        for name, shape in input_sizes.items():
            self.input_params[name] = nn.Parameter(torch.zeros(shape))

    @property
    def parametrized(self):
        return self._parametrized

    def parametrize(self):
        self._parametrized = True

    def deparametrize(self):
        self._parametrized = False

    @classmethod
    def register_hypernet(cls, hypernet_type):
        cls.ZOO[hypernet_type.__name__] = hypernet_type

