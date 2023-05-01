from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Tuple, Union

from torch import Tensor, nn

from .xparam import ExternalParameter


class HyperModule(nn.Module):
    """
    A HyperModule class. HyperModule are nn.Modules that directly or indirectly
    (via their children submodules) contain external parameters that need to
    be set with the hypernetwork predictions (or by any other compatible means).

    HyperModules are still nn.Modules so they can have a mixture of regular
    parameters and external parameters
    """

    _external_parameters: Dict[str, ExternalParameter]

    def __init__(self):
        super().__init__()
        self._external_parameters: Dict[str, ExternalParameter] = OrderedDict()

    def register_external(self, name: str, param: ExternalParameter) -> None:
        """
        Registers an external parameter with the module.

        Args:
            name (str): The name of the external parameter.
            param (ExternalParameter): The external parameter to be registered.

        Raises:
            TypeError: If name is not a string or param is not an instance of ExternalParameter.
        """
        if not isinstance(name, str):
            name_err = f"Name must be of type str, got {type(name)} instead"
            raise TypeError(name_err)
        if not isinstance(param, ExternalParameter):
            param_err = f"param must be an ExternalParameter, got {type(param)} instead"
            raise TypeError(param_err)
        self._external_parameters[name] = param

    def __setattr__(self, name: str, value: Any) -> None:
        """
        nn.Module-like overload to detect ExternalParemeter assignment

        Args:
            name (str): The name of the attribute.
            value (Any): The value of the attribute.
        """
        if isinstance(value, ExternalParameter):
            self.register_external(name, value)
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """
        nn.Module-like overload since ExternalParemeters are not true
        atributes but tracked via _external_parameters

        Args:
            name (str): The name of the attribute.
        """
        if "_external_parameters" in self.__dict__:
            _external_parameters = self.__dict__["_external_parameters"]
            if name in _external_parameters:
                return _external_parameters[name].data
        return super().__getattr__(name)

    def get_external(self, name: str) -> ExternalParameter:
        """
        Gets an external parameter of the module.

        Args:
            name (str): The name of the external parameter.

        Raises:
            LookupError: If the `name` is not an external parameter of module.

        Return:
            ExternalParameter: The external parameter object.
        """
        if name not in self._external_parameters:
            raise LookupError(f"{name} is not a external parameter of module")
        return self._external_parameters[name]

    def set_external(self, name: str, value: Tensor) -> None:
        """
        Sets the value of an external parameter of the module.

        Args:
            name (str): The name of the external parameter.
            value (Tensor): The value to be set.

        Raises:
            LookupError: If the `name` is not an external parameter of module.
        """
        if name not in self._external_parameters:
            raise LookupError(f"{name} is not a external parameter of module")
        self._external_parameters[name].data = value

    def set_externals(self, **parameters: Tensor) -> None:
        """
        Sets the values of multiple external parameters of the module.

        Args:
            **parameters (Tensor): The parameter values to be set.
        """
        for name, value in parameters.items():
            self.set_external(name, value)

    def named_externals(self) -> Iterator[Tuple[str, ExternalParameter]]:
        """
        Returns an iterator over all external parameters in the module and its submodules
        as tuples (name, external_parameter)

        Yields:
            Tuple[str, ExternalParameter]: A tuple containing the name of the external
            parameter and the external parameter object.
        """
        for modname, module in self.named_modules():
            if isinstance(module, HyperModule):
                for (
                    param_name,
                    external_param,
                ) in module._external_parameters.items():
                    if modname != "":
                        param_name = modname + "." + param_name
                    yield (param_name, external_param)

    def externals(self) -> Iterator[ExternalParameter]:
        """
        Returns an iterator over all external parameters in the module and its submodules

        Yields:
            Iterator[ExternalParameter]: An iterator over all external parameters in the
            module and its submodules
        """
        for _, external_param in self.named_externals():
            yield external_param

    def external_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Returns an dictionary of the shapes of all external parameters in the module
        and its submodules

        Returns:
            Dict[str, Tuple[int, ...]]: An iterator over the shapes of all
            external parameters in the module and its submodules
            as tuples (name, shape)
        """
        return {
            name: external_param.shape
            for name, external_param in self.named_externals()
        }

    def _groupby_module(
        self, param_dict: Dict[str, Tensor]
    ) -> Dict["HyperModule", Dict[str, Tensor]]:
        """
        Groups parameter dictionary by module containing the parameter
        """
        grouped_params = {}
        for modname, module in self.named_modules():
            if isinstance(module, HyperModule):
                grouped_params[module] = {}
                for param_name in module._external_parameters:
                    full_name = (
                        modname + "." + param_name if modname != "" else param_name
                    )
                    if full_name in param_dict:
                        grouped_params[module][param_name] = param_dict[full_name]
                if len(grouped_params[module]) == 0:
                    grouped_params.pop(module)
        return grouped_params

    def reset_externals(self) -> None:
        """
        Resets all external parameters in the module and its submodules
        """
        for module in self.modules():
            if isinstance(module, HyperModule):
                for external_param in module._external_parameters.values():
                    external_param.reset()

    def apply_externals(
        self,
        param_dict: Union[Dict["HyperModule", Dict[str, Tensor]], Dict[str, Tensor]],
    ) -> None:
        """
        Sets external parameters in the module and its submodules from a dictionary
        of the form {module: {param_name: value}} or from a dictionary of the form {full_key: value}
        where full_key uses the notation from nn.Module.named_parameters

        Args:
            param_dict (Union[Dict['HyperModule', Dict[str, Tensor]], Dict[str, Tensor]]):
                A dictionary of the form {module: {param_name: value}} or {full_key: value}
        """

        # Convert to module-key format from str-key format if needed
        first_key = next(iter(param_dict.keys()))
        if isinstance(first_key, str):
            param_dict = self._groupby_module(param_dict)

        # Set param values
        for module, external_params in param_dict.items():
            module.set_externals(**external_params)

    @contextmanager
    def using_externals(self, param_dict: Dict[str, Tensor]) -> None:
        """
        Context manager for using external parameters in a with block

        Args:
            param_dict (Dict[str, Tensor]): Dictionary of parameters
        """
        self.apply_externals(param_dict)
        yield
        self.reset_externals()

    def convert_external(self, parameter_name: str) -> Tensor:
        """
        Convert the given parameter to an external parameter and return the parameter data.

        Args:
            parameter_name (str): Name of the parameter to be converted.

        Returns:
            Tensor: Data of the converted external parameter.
        """
        parameter = getattr(self, parameter_name)
        if isinstance(parameter, nn.Parameter):
            delattr(self, parameter_name)
            shape = tuple(parameter.shape)
            self.register_external(parameter_name, ExternalParameter(shape))
            return parameter.data

    def convert_externals(self, parameter_names: List[str]) -> Dict[str, Tensor]:
        """
        Convert the given list of parameters to external parameters and return a dictionary of their data.

        Args:
            parameter_names (List[str]): Names of the parameters to be converted.

        Returns:
            Dict[str, Tensor]: Dictionary containing the data of the converted external parameters.
        """
        parameters = {name: self.convert_external(name) for name in parameter_names}
        return {name: param for name, param in parameters.items() if param is not None}
