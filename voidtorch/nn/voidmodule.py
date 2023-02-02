from collections import OrderedDict
from contextlib import contextmanager

from typing import Tuple, Dict, Iterator
from torch import nn, Tensor

from .xparam import ExternalParameter


class VoidModule(nn.Module):
    def __init__(self):
        super().__init__()
        self._external_params: Dict[str, ExternalParameter] = OrderedDict()

    def __setattr__(self, name: str, value):
        if isinstance(value, ExternalParameter):
            self._external_params[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str):
        if "_external_params" in self.__dict__:
            _external_params = self.__dict__["_external_params"]
            if name in _external_params:
                return _external_params[name].data
        return super().__getattr__(name)

    def get_external(self, name: str) -> ExternalParameter:
        if name not in self._external_params:
            raise LookupError(f"{name} is not a external parameter of module")
        return self._external_params[name]

    def set_external(self, name: str, value: Tensor):
        if name not in self._external_params:
            raise LookupError(f"{name} is not a external parameter of module")
        self._external_params[name].data = value

    def set_externals(self, **params):
        for name, value in params.items():
            self.set_external(name, value)

    def named_externals(self) -> Iterator[Tuple["str", ExternalParameter]]:
        for modname, module in self.named_modules():
            if isinstance(module, VoidModule):
                for param_name, external_param in module._external_params.items():
                    if modname != "":
                        param_name = modname + "." + param_name
                    yield (param_name, external_param)

    def externals(self) -> Iterator[ExternalParameter]:
        for _, external_param in self.named_externals():
            yield external_param

    def external_shapes(self) -> Iterator[Tuple[str, Tuple[int, ...]]]:
        for name, external_param in self.named_externals():
            yield name, external_param.shape

    def _groupby_module(
        self, param_dict: Dict[str, Tensor]
    ) -> Dict["VoidModule", Dict[str, Tensor]]:
        grouped_params = {}
        for modname, module in self.named_modules():
            if isinstance(module, VoidModule):
                grouped_params[module] = {}
                for param_name in module._external_params:
                    full_name = (
                        modname + "." + param_name if modname != "" else param_name
                    )
                    if full_name in param_dict:
                        grouped_params[module][param_name] = param_dict[full_name]
                if len(grouped_params[module]) == 0:
                    grouped_params.pop(module)
        return grouped_params

    def reset_externals(self):
        for module in self.modules():
            if isinstance(module, VoidModule):
                for external_param in module._external_params.values():
                    external_param.reset()

    def apply_externals(self, param_dict):
        first_key = next(iter(param_dict.keys()))
        if isinstance(first_key, str):
            param_dict = self._groupby_module(param_dict)
        for module, external_params in param_dict.items():
            module.set_externals(**external_params)

    @contextmanager
    def using_externals(self, param_dict):
        self.apply_externals(param_dict)
        yield
        self.reset_externals()

    # def parameters(self):
    #     for external in self.externals():
    #         yield external.data

    # def named_parameters(self):
    #     for name, external in self.named_externals():
    #         yield (name, external.data)

