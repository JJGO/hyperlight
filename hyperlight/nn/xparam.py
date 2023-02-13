from typing import Tuple, Optional

from torch import Tensor

# TODO Enforce DataType?
# TODO ExternalParameterList and ExternalParameterDict


class ExternalParameter:
    def __init__(self, shape: Tuple[int, ...], data: Optional[Tensor] = None):
        self.shape = shape
        self._data = None
        if data is not None:
            self.data = data

    @property
    def data(self):
        if self._data is None:
            raise AttributeError(
                "Uninitialized External Parameter, please set the value first"
            )
        return self._data

    @data.setter
    def data(self, value):
        assert isinstance(
            value, Tensor
        ), f"data should be torch.Tensor but is {value.__class__.__name__} instead"
        assert (
            value.shape == self.shape
        ), f"parameter tensor shape mismatch, expected {tuple(self.shape)}, got {tuple(value.shape)}"
        self._data = value

    def reset(self):
        self._data = None

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={tuple(self.shape)})"

    def __str__(self):
        return repr(self)

