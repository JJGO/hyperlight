import warnings
from typing import Dict, Literal

import torch
from torch import Tensor

InputMode = Literal[None, "cos|sin", "z|1-z"]


encoding_multiplier: Dict[InputMode, int] = {
    None: 1,
    "cos|sin": 2,
    "z|1-z": 2,
}


def _check_tensor_range(tensor: Tensor, low: float, high: float) -> None:
    """
    Check if the input tensor is within specified range.

    Args:
        tensor (Tensor): The input tensor.
        low (float): The lower bound of the range.
        high (float): The upper bound of the range.

    Raises:
        UserWarning: If the input tensor is outside the specified range.
    """
    if (tensor < low).any() or (tensor > high).any():
        warnings.warn(f"Input tensor has dimensions outside of [{low},{high}].")


def encode_input(input: Tensor, mode: InputMode = "cos|sin") -> Tensor:
    """
    Encodes the input tensor based on the specified mode.

    Args:
        input (Tensor): The input tensor.
        mode (InputMode): The encoding mode.

    Returns:
        Tensor: The encoded tensor.

    Raises:
        UserWarning: If the input tensor is outside the specified range in cos|sin mode.
    """
    z = input

    if mode is None:
        return input

    if mode == "cos|sin":
        _check_tensor_range(input, low=0, high=1)
        scaled_z = torch.pi * z / 2

        return torch.cat([torch.cos(scaled_z), torch.sin(scaled_z)], dim=-1)

    if mode == "z|1-z":
        return torch.cat([z, 1 - z], dim=-1)
