import copy
from collections import OrderedDict
from fnmatch import fnmatch
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Type

from torch import Tensor, nn

from .nn import ExternalParameter, HyperModule


def _init_from_regular(
    self, module: nn.Module, external_parameters: Optional[List[str]] = None
) -> None:
    """
    Helper function that will override the __init__ of modules that
    are dynamically hypernetized. Instead of being initialized from the
    original arguments, it takes in a non-hypernetized module and copies the
    state.

    Args:
        self: The instance of the class.
        module (nn.Module): The module to initialize from.
        external_parameters (Optional[List[str]], optional): List of parameters to externalize. Defaults to None.
    """
    self._external_parameters = OrderedDict()
    # We don't do a deepcopy to allow reusing the original model state if preferred
    self.__dict__.update(module.__dict__.copy())
    if external_parameters is not None:
        self.convert_externals(external_parameters)


# We lru_cache so we don't create multiple hypernetized classes for the same original nn.Module type
@lru_cache(maxsize=None)
def _hyper_class(module_type: Type[nn.Module]) -> Type[HyperModule]:
    """
    Given a nn.Module type, returns a HyperModule subtype that can
    externalizes the specified parameters so they can be computed
    externally (e.g. via a hypernetwork)

    Args:
        module_type (Type[nn.Module]): The module type to create a hypernetized subtype for.

    Returns:
        Type[HyperModule]: The hypernetized subtype.
    """
    return type(
        f"Hypernetized{module_type.__name__}",
        (module_type, HyperModule),
        {"__init__": _init_from_regular, "extra_repr": lambda x: "",},
        # module_type.extra_repr,}
    )


def hypernetize_single(
    module: nn.Module,
    parameters: Optional[List[nn.Parameter]] = None,
    return_parameters: bool = False,
) -> HyperModule:
    """
    hypernetize a single module.

    Args:
    - module (nn.Module): The module to hypernetize.
    - parameters (Optional[List[nn.Parameter]], optional): A list of parameters to be converted. Defaults to None.
    - return_parameters (bool, optional): Whether to return hyper module and converted parameters or just hyper module. Defaults to False.

    Returns:
    - HyperModule: The hypernetized module.
    or
    - Tuple[HyperModule, Dict[str, torch.Tensor]]: A tuple containing the hypernetized module and the converted parameters.
    """
    if not isinstance(module, HyperModule):
        constructor = _hyper_class(module.__class__)
        module = constructor(module)

    if parameters is not None:
        values = module.convert_externals(parameters)

    if return_parameters:
        return module, values

    return module


def hypernetize(
    model: nn.Module,
    modules: Optional[List[nn.Module]] = None,
    parameters: Optional[List[nn.Parameter]] = None,
    return_values: bool = False,
    inplace: bool = True,
):
    """
    Recursively replace children modules with hyper modules as needed.
     parameters of a module and its children with None, except for
    those specified in `parameters` and `modules`.

    Args:
        module (nn.Module): PyTorch module to hypernetize.
        mapping (Dict[nn.Module, nn.Module]): Dictionary to store hypernetized modules.
        modules (Optional[List[nn.Module]]): List of PyTorch modules whose parameters should be hypernetized. Defaults to None.
        parameters (Optional[List[nn.Parameter]]): List of PyTorch parameters that should not be hypernetized. Defaults to None.
        inplace (bool): Whether to hypernetize inplace

    Returns:
      Tuple[HyperModule, Dict[str, torch.Tensor]]: A tuple containing the hypernetized module and the converted parameters.
    """
    if not inplace:
        model = copy.deepcopy(model)

    if isinstance(modules, dict):
        modules = list(modules.values())
    if isinstance(parameters, dict):
        parameters = list(parameters.values())

    parameters = parameters or []

    if modules is not None:
        parameters += sum(
            [list(module._parameters.values()) for module in modules], start=[]
        )

    parameters = set(parameters)

    if len(parameters) > 0:
        parameter_values = {}
        # need to pass the module names mapping to return human-readable parameter values
        module_names = {module: name for name, module in model.named_modules()}
        model = _hypernetize(
            model,
            {},
            parameters=parameters,
            parameter_values=parameter_values,
            module_names=module_names,
        )

    # top-level module should support HyperModule API
    if not isinstance(model, HyperModule):
        model = hypernetize_single(model)

    if return_values:
        return model, parameter_values

    return model


def _hypernetize(
    module: nn.Module,
    memo: Dict[nn.Module, nn.Module],
    parameters: List[nn.Parameter],
    parameter_values: Dict[str, Tensor],
    module_names: Dict[nn.Module, str],
):
    """
    Helper for recursion with memoization
    memo is required so if we encounter the same module multiple
    times we don't make multiple  independent copies of it
    """

    if module in memo:
        return memo[module]

    # hypernetize current module
    external_parameters = []
    for name, param in module._parameters.items():
        if param in parameters:
            external_parameters.append(name)

    if len(external_parameters) > 0:
        memo[module] = hypernetize_single(module)
        parameter_data = memo[module].convert_externals(external_parameters)
        module_name = module_names[module]
        for param, value in parameter_data.items():
            parameter_values[f"{module_name}.{param}"] = value
    else:
        memo[module] = module

    # hypernetize children
    for name, submodule in module.named_children():
        submodule = _hypernetize(
            submodule,
            memo=memo,
            parameters=parameters,
            module_names=module_names,
            parameter_values=parameter_values,
        )
        setattr(memo[module], name, submodule)

    return memo[module]
