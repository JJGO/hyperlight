from typing import Optional

import torch
from torch import nn
from torch.nn import init


def initialize_weight(
    weight: torch.Tensor,
    distribution: Optional[str],
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """
    Initialize weight tensor using the specified distribution and nonlinearity function.

    Args:
        weight (torch.Tensor): Tensor to be initialized.
        distribution (str, optional): Distribution to use for initialization.
        nonlinearity (str, optional): Nonlinearity function to use. Defaults to "LeakyReLU".

    Raises:
        ValueError: When the specified distribution is not supported.
    """
    if distribution is None:
        return

    if nonlinearity:
        nonlinearity = nonlinearity.lower()
        if nonlinearity == "leakyrelu":
            nonlinearity = "leaky_relu"

    if nonlinearity is None:
        nonlinearity = "linear"

    if nonlinearity in ("silu", "gelu"):
        nonlinearity = "leaky_relu"

    gain = 1 if nonlinearity is None else init.calculate_gain(nonlinearity)

    if distribution == "zeros":
        init.zeros_(weight)
    elif distribution == "kaiming_normal":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_uniform":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity)
    elif distribution == "kaiming_normal_fanout":
        init.kaiming_normal_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "kaiming_uniform_fanout":
        init.kaiming_uniform_(weight, nonlinearity=nonlinearity, mode="fan_out")
    elif distribution == "glorot_normal":
        init.xavier_normal_(weight, gain=gain)
    elif distribution == "glorot_uniform":
        init.xavier_uniform_(weight, gain)
    elif distribution == "orthogonal":
        init.orthogonal_(weight, gain)
    else:
        raise ValueError(f"Unsupported weight distribution '{distribution}'")


def initialize_bias(bias: torch.Tensor, distribution: Optional[float] = 0.0) -> None:
    """
    Initializes the bias tensor of a layer using the given distribution.

    Args:
        bias (nn.Parameter): the bias tensor to be initialized
        distribution (float): the distribution to use for initialization, default is 0 (constant)

    Raises:
        ValueError: When the specified distribution is not supported.
    """
    if distribution is None:
        return

    if isinstance(distribution, (int, float)):
        init.constant_(bias, distribution)
        return

    raise ValueError(f"Unsupported bias distribution '{distribution}'")


def initialize_layer(
    layer: nn.Module,
    distribution: Optional[str] = "kaiming_normal",
    init_bias: Optional[float] = 0.0,
    nonlinearity: Optional[str] = "LeakyReLU",
) -> None:
    """
    Initializes the weight and bias tensors of a linear or convolutional layer using the given distribution.

    Args:
    - layer (nn.Module): the linear or convolutional layer to be initialized
    - distribution (str): the distribution to use for initialization, default is 'kaiming_normal'
    - init_bias (float): the initial value of the bias tensor, default is 0
    - nonlinearity (str): the nonlinearity function to use for initialization, default is "LeakyReLU"

    Returns:
    - None
    """
    assert isinstance(
        layer, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
    ), f"Can only be applied to linear and conv layers, given {layer.__class__.__name__}"

    initialize_weight(layer.weight, distribution, nonlinearity)
    if layer.bias is not None:
        initialize_bias(layer.bias, init_bias)
