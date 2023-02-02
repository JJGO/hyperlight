import copy
from collections import OrderedDict
from functools import lru_cache
from fnmatch import fnmatch

from torch import nn

from nn import VoidModule, ExternalParameter

def allbut(mapping, keys):
    mapping = copy.deepcopy(mapping)
    for k in keys:
        if k in mapping:
            del mapping[k]
    return mapping


def _init_from_regular(self, module: nn.Module):
    self._external_params = OrderedDict()
    self.__dict__.update(allbut(module.__dict__, ["_parameters"]))
    self._parameters = {}
    for name, value in module._parameters.items():
        if isinstance(value, nn.Parameter):
            setattr(self, name, ExternalParameter(value.shape))
        else:
            setattr(self, name, value)


@lru_cache(maxsize=None)
def _voided_class(module_type) -> type:

    return type(
        f"Void{module_type.__name__}",
        (module_type, VoidModule),
        {"__init__": _init_from_regular, "extra_repr": lambda x: "",},
    )


def _voidify(
    module: nn.Module, recurse=True, memo=None, module_types=None, glob="*", prefix=""
):

    # Skip non parametric modules
    if not any(True for _ in module.parameters()):
        return module

    if module not in memo:
        valid_type = module_types is None or isinstance(module, module_types)
        if len(memo) == 0:
            memo[module] = _voided_class(module.__class__)(module)
        # elif len(module._parameters) == 0:
        #     memo[module] = module
        elif valid_type and fnmatch(prefix, glob):
            memo[module] = _voided_class(module.__class__)(module)
        else:
            memo[module] = copy.deepcopy(module)

    voided_module = memo[module]

    if recurse:
        for name, submodule in module.named_children():
            full_name = f"{prefix}.{name}"
            submodule = _voidify(
                submodule,
                memo=memo,
                module_types=module_types,
                glob=glob,
                prefix=full_name,
            )
            setattr(voided_module, name, submodule)
    return voided_module


def voidify(module, recurse=True, module_types=None, glob="*"):
    memo = {}
    module_types = tuple(module_types) if module_types is not None else None
    return _voidify(
        module, recurse=recurse, module_types=module_types, memo=memo, glob=glob
    )

