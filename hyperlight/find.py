import fnmatch
import re
from typing import Dict, List, Optional, Sequence, Tuple, Type

from torch import nn


def _ensure_module(module: nn.Module):
    if not isinstance(module, nn.Module):
        msg = f"module must be a nn.Module, got {type(module)} instead"
        raise TypeError(msg)


def _ensure_module_types(
    module_types: Sequence[Type[nn.Module]],
) -> Tuple[Type[nn.Module]]:
    if isinstance(module_types, list):
        module_types = tuple(module_types)

    if not isinstance(module_types, tuple):
        msg = f"module_types must be Tuple of nn.Module types, got {type(module_types)} instead"
        raise TypeError(msg)

    for type_ in module_types:
        if not issubclass(type_, nn.Module):
            msg = f"{type_} in module_types is not subclass of nn.Module"
            raise TypeError(msg)
    return module_types


def find_modules_of_type(
    model: nn.Module, module_types: Sequence[Type[nn.Module]]
) -> Dict[str, nn.Module]:
    """
    Find all modules of given types in a PyTorch model.

    Args:
      model (nn.Module): The PyTorch model to search for modules.
      module_types (Sequence[Type[nn.Module]]): The types of modules to search for.

    Returns:
      Dict[str, nn.Module]: A list of tuples containing the name and module for each found module.

    Raises:
      TypeError: If model is not an instance of nn.Module or if module_types is not comprised of nn.Module types.
    """
    _ensure_module(model)
    module_types = _ensure_module_types(module_types)
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, module_types)
    }


def find_modules_from_patterns(
    model: nn.Module,
    globs: Optional[List[str]] = None,
    regex: Optional[List[str]] = None,
) -> Dict[str, nn.Module]:
    """
    Find modules in a PyTorch model that match any of the given glob or regex patterns.

    Args:
        model (nn.Module): The PyTorch model to search for modules.
        globs (Optional[List[str]]): A list of glob-style patterns to match module names against.
        regex (Optional[List[str]]): A list of regular expressions to match module names against.

    Returns:
        Dict[str, nn.Module]: A dictionary of names to modules that match the given patterns.
    """
    globs = globs or []
    regex = regex or []

    named_modules = dict(model.named_modules())

    pattern_matches = set()
    for glob_pattern in globs:
        pattern_matches |= set(
            filter(
                lambda name: fnmatch.fnmatch(name, glob_pattern), named_modules.keys()
            )
        )
    for regex_pattern in regex:
        pattern_matches |= set(
            filter(lambda name: re.search(regex_pattern, name), named_modules.keys())
        )
    return {name: named_modules[name] for name in pattern_matches}


def find_parameters_from_patterns(
    model: nn.Module,
    globs: Optional[List[str]] = None,
    regex: Optional[List[str]] = None,
) -> Dict[str, nn.Parameter]:
    """
    Find parameters in a PyTorch model that match any of the given glob or regex patterns.

    Args:
        model (nn.Module): The PyTorch model to search for parameters.
        globs (Optional[List[str]]): A list of glob-style patterns to match module names against.
        regex (Optional[List[str]]): A list of regular expressions to match module names against.

    Returns:
        Dict[str, nn.Parameter]: A dictionary of names to parameters that match the given patterns.
    """
    globs = globs or []
    regex = regex or []

    named_parameters = dict(model.named_parameters())

    pattern_matches = set()
    for glob_pattern in globs:
        pattern_matches |= set(
            filter(
                lambda name: fnmatch.fnmatch(name, glob_pattern),
                named_parameters.keys(),
            )
        )
    for regex_pattern in regex:
        pattern_matches |= set(
            filter(lambda name: re.search(regex_pattern, name), named_parameters.keys())
        )
    return {name: named_parameters[name] for name in pattern_matches}
