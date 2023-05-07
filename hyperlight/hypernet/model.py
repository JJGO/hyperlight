import math
from collections import OrderedDict
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .encoding import InputMode, encode_input, encoding_multiplier
from .initialization import (initialize_bias, initialize_layer,
                             initialize_weight)

Shape = Tuple[int, ...]


def _same_keys(
    required: Dict[str, Any], provided: Dict[str, Any], name: str = ""
) -> None:
    """
    Checks if two dictionaries have the same keys.

    Args:
        required: A dictionary with required keys.
        provided: A dictionary with provided keys.
        name: A string name to identify the dictionaries.

    Raises:
        ValueError: If the dictionaries have missing or unexpected keys.
    """
    missing = required.keys() - provided.keys()
    if len(missing) > 0:
        raise ValueError(f"Missing {name}: {list(missing)}")

    unexpected = provided.keys() - required.keys()
    if len(unexpected) > 0:
        raise ValueError(f"Unexpected {name}: {list(missing)}")


class FullyConnectedNet(nn.Sequential):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_prob: float = 0,
        nonlinearity: str = "LeakyReLU",
        out_bias: bool = True,
        init_distribution: Optional[str] = "kaiming_normal_fanout",
        init_bias: Optional[float] = 0.0,
    ) -> None:
        """
        Creates a fully connected neural network with specified input size, hidden sizes,
        output size, dropout probability and activation function.

        The default choice of 'fan out' init is done on purpose as hypernetworks tend to
        substantially increase the number of neurons, specially in the last layer

        Args:
            input_size (int): Dimension of input
            hidden_sizes (List[int]): List of hidden layer dimensions
            output_size (int): Dimension of output
            dropout_prob (float): Dropout probability (default=0)
            nonlinearity (str): Nonlinear activation function (default='LeakyReLU')
            out_bias (bool): Whether last layer should have bias (default=True)
            init_distribution (Optional[str]): Initialization distribution (default='kaiming_normal_fanout')
            init_bias (Optional[float]): Initialization bias (default=0.0)
        """

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.nonlinearity = nonlinearity
        self.init_distribution = init_distribution
        self.init_bias = init_bias
        self.out_bias = out_bias

        layers = OrderedDict()

        nonlinearity_fn = getattr(nn, nonlinearity)

        in_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers[f"block{i}_lin"] = nn.Linear(in_size, hidden_size)
            layers[f"block{i}_act"] = nonlinearity_fn()

            if dropout_prob > 0:
                layers[f"block{i}_drop"] = nn.Dropout(p=dropout_prob)
            in_size = hidden_size

        layers["output"] = nn.Linear(in_size, output_size, bias=out_bias)

        super().__init__(layers)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the network layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                initialize_layer(
                    module,
                    self.init_distribution,
                    init_bias=self.init_bias,
                    nonlinearity=self.nonlinearity,
                )

            initialize_layer(
                self.output,
                self.init_distribution,
                init_bias=self.init_bias,
                nonlinearity=None,
            )


# TODO support for categoricals via 'C' shape and frozen embeddings
class HyperNetMixin:

    input_shapes: Dict[str, Shape]
    output_shapes: Dict[str, Shape]
    encoding: InputMode

    def encode_inputs(self, inputs: Dict[str, Shape]) -> Dict[str, Shape]:
        """Encodes the inputs using the specified `encoding` mode.

        Args:
            inputs: A dictionary of input names and shapes.

        Returns:
            Dict[str, Shape]: A dictionary of encoded input names and shapes.
        """
        return {k: encode_input(x, mode=self.encoding) for k, x in inputs.items()}

    def flat_input_size(self) -> int:
        """Returns the flat input size after encoding.

        Returns:
            int: The flat input size.
        """
        flat_input_size = sum(int(math.prod(v)) for v in self.input_shapes.values())
        flat_input_size *= encoding_multiplier[self.encoding]
        return flat_input_size

    def flat_output_size(self) -> int:
        """Returns the flat output size.

        Returns:
            int: The flat output size.
        """
        flat_output_size = sum(int(math.prod(v)) for v in self.output_shapes.values())
        return flat_output_size

    def output_offsets(self) -> Dict[str, Tuple[int, int]]:
        """Returns the output offsets.

        Returns:
            Dict[str, Tuple[int, int]]: A dictionary of output names and 
            their corresponding offsets and sizes.
        """
        offset = 0
        offsets = {}
        for name, shape in self.output_shapes.items():
            size = int(math.prod(shape))
            offsets[name] = (offset, size)
            offset += size
        return offsets

    def _validate_inputs(self, inputs: Dict[str, Tensor]) -> int:
        """
        Validates the inputs.

        Args:
            inputs (Dict[str, Tensor]): A dictionary of input names and their corresponding tensors

        Returns:
            int: The batch dimension.

        Raises:
            ValueError: If the input shapes are not as expected or if different batch dimensions are found.
        """
        _same_keys(self.input_shapes, inputs, name="inputs")

        for name, required_shape in self.input_shapes.items():

            shape = inputs[name].shape

            # If batch dimension is already provided
            if shape[1:] == required_shape:
                continue

            # If no batch dimension is provided, but shape is correct
            if shape == required_shape:
                # Add batch dimension
                inputs[name] = inputs[name][None]
                continue

            error_msg = f"Wrong shape for {name}, expected {required_shape} or {('B',*required_shape)}, got {shape}"
            raise ValueError(error_msg)

        # All batch dims must match
        batch_dims = [x.shape[0] for x in inputs.values()]
        if len(set(batch_dims)) > 1:
            raise ValueError(f"Multiple batch dimensions found: {batch_dims}")

        return batch_dims[0]

    def flatten_inputs(self, inputs: Dict[str, Tensor]) -> Tensor:
        """
        Flattens the input tensors into a single tensor along the last dimension.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary of input tensor names and tensor values.

        Returns:
            Tensor: The flattened input tensor.
        """
        batch_dim = self._validate_inputs(inputs)

        flat_input = torch.cat(
            [inputs[hp].view(batch_dim, -1) for hp in sorted(self.input_shapes)], dim=-1
        )
        return flat_input

    def unflatten_output(self, flat_output: Tensor) -> Dict[str, Tensor]:
        """
        Unflattens the output tensor into a dictionary of named tensors.

        Args:
            flat_output (Tensor): The flattened output tensor.

        Returns:
            Dict[str, Tensor]: A dictionary of named tensors.
        """
        outputs = {}
        batch_dim = flat_output.shape[0]
        for name, shape in self.output_shapes.items():
            if batch_dim > 1:
                shape = (batch_dim, *shape)
            outputs[name] = flat_output.narrow(1, *self._output_offsets[name]).view(
                shape
            )
        return outputs


class HyperNet(nn.Module, HyperNetMixin):
    """A hypernetwork that generates weights for a target network. Shape is Dict[str, Tuple[int, ...]]

    Args:
        input_shapes (Dict[str, Shape]): The shapes of the inputs that the hypernetwork takes
        output_shapes (Dict[str, Shape]): The shapes of the primary network weights being predicted,
        hidden_sizes (List[int]): The sizes of the hidden layers of the hypernetwork.
        encoding (InputMode, optional): The input encoding mode. Defaults to "cos|sin".
        init_independent_weights (bool, optional): Whether to init weights independently. Defaults to True.
        output_split_init (bool, optional): Whether to initialize the output split. Defaults to True.
        fc_kws (Dict[str, Any], optional): Keyword arguments for the fully connected layers. Defaults to None.
    """
    def __init__(
        self,
        input_shapes: Dict[str, Shape],
        output_shapes: Dict[str, Shape],
        hidden_sizes: List[int],
        encoding: InputMode = "cos|sin",
        init_independent_weights: bool = True,
        output_split_init: bool = True,
        fc_kws: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        if not isinstance(input_shapes, dict):
            input_shapes = dict(input_shapes)
        if not isinstance(output_shapes, dict):
            output_shapes = dict(output_shapes)

        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.hidden_sizes = hidden_sizes
        self.encoding = encoding

        # Cache this property to avoid recomputation
        self._output_offsets = self.output_offsets()

        self.backbone = FullyConnectedNet(
            input_size=self.flat_input_size(),
            hidden_sizes=self.hidden_sizes,
            output_size=self.flat_output_size(),
            **(fc_kws or {}),
        )

        if init_independent_weights:
            self.init_independent_weights()

        if output_split_init:
            self._init_output_split()

    def _init_output_split(self):
        """
        Helper function that initializes the MLP last fully connected layer
        as N independent layers. This is important because the fan_out scheme
        will substantially change depending on the number of predicted neurons

        Moreover, if the layer is predicting a bias, we initialize it to zero
        """

        output_weight = self.backbone.output.weight
        nonlinearity = self.backbone.nonlinearity
        init_distribution = self.backbone.init_distribution

        for name in self.output_shapes:
            offset, length = self._output_offsets[name]
            tensor = output_weight[offset:offset+length,:]
            if name.endswith('.bias'):
                # Bias should initialize to zero
                initialize_weight(tensor, 'zeros')
            elif name.endswith('.weight'):
                initialize_weight(tensor, init_distribution, nonlinearity=nonlinearity)
            else:
                initialize_weight(tensor, init_distribution, nonlinearity=nonlinearity)


    @classmethod
    def from_existing(cls, weights: Dict[str, Tensor], **kwargs) -> "HyperNet":
        output_shapes = {k: v.shape for k, v in weights.items()}
        hypernet = cls(output_shapes=output_shapes, **kwargs)
        hypernet.set_independent_weights(weights)
        return hypernet

    def set_independent_weights(self, weights: Dict[str, Tensor]) -> None:
        """
        Sets the weights of the model's backbone independent of the hypernetwork

        Args:
            weights (Dict[str, Tensor]): A dictionary containing weights to be set.

        Raises:
            ValueError: If backbone has no bias or if the shape of `weights` does not match the required shape.
        """
        if not self.backbone.out_bias:
            raise ValueError("backbone's last layer has no bias")

        _same_keys(self.output_shapes, weights, name="weights")

        for name, required_shape in self.output_shapes.items():
            shape = weights[name].shape
            if shape != required_shape:
                raise ValueError(
                    f"Wrong shape for {name}, expected {required_shape}. got {shape} instead"
                )

        flat_weights = torch.cat([tensor.flatten() for tensor in weights.values()])

        self.backbone.output.bias.data = flat_weights

    def get_independent_weights(self) -> Dict[str, Tensor]:
        """
        Returns the weights of the model's backbone independent of the hypernetwork

        Returns:
            Dict[str, Tensor]: A dictionary containing independent weights
        Raises:
            ValueError: If backbone has no bias.
        """
        if not self.backbone.out_bias:
            raise ValueError("backbone's last layer has no bias")

        flat_values: Tensor = self.backbone.output.bias.data
        weights = {}
        for name, shape in self.output_shapes.items():
            weights[name] = flat_values.narrow(0, *self._output_offsets[name]).view(
                shape
            )
        return weights

    def init_independent_weights(
        self,
        init_distribution: str = "kaiming_normal",
        init_bias: Optional[float] = 0.0,
        init_nonlinearity: Optional[str] = "LeakyReLU",
    ) -> None:
        """Initializes the weights of the neural network.

        Args:
            init_distribution: The initialization distribution to use for the weights.
            init_bias: The initialization value or distribution to use for the biases.
            init_nonlinearity: The nonlinearity to use for weight initialization.
        """
        weights = {}

        for name, shape in self.output_shapes.items():
            param = torch.zeros(shape)

            if name.endswith(".weight"):
                initialize_weight(
                    param, init_distribution, nonlinearity=init_nonlinearity,
                )
            if name.endswith(".bias"):
                initialize_bias(param, init_bias)

            weights[name] = param

        self.set_independent_weights(weights)

    def freeze_independent_weights(self) -> None:
        """
        Freezes the independent set of parameters,
        i.e. the bias of the last layer

        Raises:
            ValueError: If the backbone's last layer has no bias.
        """
        if not self.backbone.out_bias:
            raise ValueError("backbone's last layer has no bias")
        self.backbone.output.bias.requires_grad_(False)

    def unfreeze_independent_weights(self) -> None:
        """
        Unfreezes the independent set of parameters,
        i.e. the bias of the last layer

        Raises:
            ValueError: If the backbone's last layer has no bias.
        """
        if not self.backbone.out_bias:
            raise ValueError("backbone's last layer has no bias")
        self.backbone.output.bias.requires_grad_(True)

    def forward(self, **inputs: Tensor) -> Dict[str, Tensor]:
        """Performs a forward pass of the neural network.

        Args:
            inputs: The input tensors.

        Returns:
            A dictionary of output tensors.
        """
        flat_input = self.flatten_inputs(inputs)
        flat_input = encode_input(flat_input, self.encoding)
        flat_output = self.backbone(flat_input)
        outputs = self.unflatten_output(flat_output)
        return outputs
