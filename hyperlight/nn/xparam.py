from typing import Optional, Tuple

import torch
from torch import Tensor


class ExternalParameter:
    """
    External parameters owned by HyperModules.
    ExternalParameter classes are a container that
    will report the required shape/dtype and check
    those when data is assigned.

    Attributes:
        shape (Tuple[int, ...]): The required shape of the data.
        data (Optional[Tensor]): The actual data.
        dtype (Optional[torch.dtype]): The required dtype of the data.

    Note:
        Only .data can be modified, shape and dtype are protected
    """

    __slots__ = ("_shape", "_data", "_dtype")

    def __init__(
        self,
        shape: Tuple[int, ...],
        data: Optional[Tensor] = None,
        dtype: torch.dtype = None,
    ) -> None:
        """
        Constructs a parameter.

        Args:
            shape (Tuple[int, ...]): The shape of the parameter.
            data (Optional[Tensor], optional): The data of the parameter. Defaults to None.
            dtype (torch.dtype, optional): The datatype of the parameter. Defaults to None.
        """

        self._shape = shape
        self._dtype = dtype
        self._data = None
        self._validate_attrs()
        if data is not None:
            self.data = data

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the external parameter.

        Returns:
            Tuple[int, ...]: Expected shape of the tensor.
        """
        return self._shape

    @property
    def dtype(self) -> Optional[torch.dtype]:
        """Returns the data type of the external parameter.

        Returns:
            Optional[torch.dtype]: Data type of the tensor as a torch.dtype object or None if data type is not set.
        """
        return self._dtype

    def _validate_attrs(self) -> None:
        """
        Raises:
            TypeError: If the shape is not a tuple of integers.
            TypeError: If the dtype is not a valid Tensor type.
        """

        # Validate shape
        if isinstance(self.shape, list):
            self.shape = tuple(self.shape)
        shape = self.shape
        shape_err = f"Shape must be Tuple[int, ...], but got {type(shape)} instead"
        if not isinstance(shape, tuple) and all(isinstance(d, int) for d in shape):
            raise TypeError(shape_err)

        # Validate dtype
        if self._dtype is not None:
            type_err = f"dtype must be a valid Tensor type, got {self._dtype} instead"
            if not isinstance(self._dtype, torch.dtype):
                raise TypeError(type_err)

    @property
    def data(self) -> Tensor:
        """
        Returns the data of the parameter.

        Raises:
            AttributeError: If the parameter is uninitialized.

        Returns:
            Tensor: The data of the parameter.
        """

        if self._data is None:
            raise AttributeError(
                "Uninitialized External Parameter, please set the value first"
            )
        return self._data

    @data.setter
    def data(self, value: Tensor) -> None:
        """
        Sets the data of the parameter, validating shape and dtype

        Args:
            value (Tensor): The data of the parameter.

        Raises:
            TypeError: If the datatype or shape of the parameter is incorrect.
        """

        assert isinstance(
            value, Tensor
        ), f"data should be torch.Tensor but is {value.__class__.__name__} instead"
        assert (
            value.shape == self.shape
        ), f"parameter tensor shape mismatch, expected {tuple(self.shape)}, got {tuple(value.shape)}"
        if self._dtype is not None:
            assert (
                value.dtype == self._dtype
            ), f"parameter tensor dtype mismatch, expected {self._dtype}, got {value.dtype}"
        self._data = value

    def reset(self) -> None:
        """
        Resets the data of the parameter to None.
        """

        self._data = None

    def __repr__(self) -> str:
        """
        Returns a string representation of the parameter.

        Returns:
            str: A string representation of the parameter.

        """

        s = f"{self.__class__.__name__}(shape={tuple(self.shape)}"
        if self._dtype:
            s += f", dtype={repr(self._dtype)}"
        if self._data is not None:
            s += f", data={repr(self._data)}"
        s += ")"
        return s
