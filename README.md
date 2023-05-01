# HyperLight

> Hypernetworks in Pytorch made easy

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&amp;logo=PyTorch&amp;logoColor=white)](https://pytorch.org)
[![Supported Python Versions](https://img.shields.io/pypi/pyversions/hyperlight)](https://pypi.org/project/hyperlight/) 
[![PyPI version](https://badge.fury.io/py/hyperlight.svg)](https://badge.fury.io/py/hyperlight)
[![Downloads](https://pepy.tech/badge/hyperlight)](https://pepy.tech/project/hyperlight)
[![license](https://img.shields.io/github/license/JJGO/hyperlight.svg)](https://github.com/JJGO/hyperlight/blob/main/LICENSE)

## TL;DR

HyperLight is a Pytorch library designed to make implementing hypernetwork models easy and painless.
What sets HyperLight apart from other hypernetwork implementations:

- **Bring your own architecture** – HyperLight lets you reuse your existing model code.
- **Principled Parametrizations and Initializations** – Default networks can have unstable training dynamics, HyperLight has good defaults that lead to improved training [1].
- **Work with pretrained models** – Pretrained weights can be used as part of the hypernetwork initialization.
- **Seemless Composability** – It's hypernets all the way down! You can hypernetize hypernet models without issue.
- **_Pytorch-nic_ API design** – Parameters are not treated as inputs to layers, preventing lots of code rewriting from happening.
<!-- - **Easy weight reuse** – Once a model has its weights set, it can be used many times. -->

[1] [Non-Proportional Parametrizations for Stable Hypernetwork Learning](https://arxiv.org/abs/2304.07645)

## Installation

HyperLight can be installed via `pip`. For the **stable** version:

```shell
pip install hyperlight
```

Or for the **latest** version:

```shell
pip install git+https://github.com/JJGO/hyperlight.git
```

You can also **manually** install it by cloning it, installing dependencias and adding it to your `PYTHONPATH`:


```shell
git clone https://github.com/JJGO/hyperlight
python -m pip install -r ./hyperlight/requirements.txt # only dependency is PyTorch

# Put this on your .bashrc/.zshrc
export PYTHONPATH="$PYTHONPATH:/path/to/hyperlight)"
```


## Getting Started

The main advantage of HyperLight is that it allows to easily reuse existing networks without having to redo the model code.

For example, here's how we can write a Bayesian Neural Hypernetwork for the resnet18 architecture.

```python
from torchvision.models import resnet18
from hyperlight import hypernetize, Hypernet, find_modules_of_type

# First we instantiate the main network and
# hyperparametrize all convolutional weights
mainnet = resnet18()
modules = find_modules_of_type(mainnet, [nn.Conv2d])

# Replace nn.Parameter objects with ExternalParameters
mainnet = hypernetize(mainnet, modules=modules)

# Now, we get the spec of the weights we need to predict
parameter_shapes = mainnet.external_shapes()

# We can predict these shapes any way we want,
# but hyperlight provides hypernetwork models
hyperparam_shape = {'h': (10,)} # 10-dim input
hypernet = Hypernet(
    input_shapes=hyperparam_shape,
    output_shapes=parameter_shapes,
    hidden_sizes=[16,64,128],
)

# Now, instead of model(input) we first predict the main network weights
parameters = hypernet(h=hyperpameter_input)

# and then use the main network
with mainnet.using_externals(parameters):
    # within this context manager, the weights are accesible
    prediction = mainnet(input)

    # after this point, weights are removed
```

We can also wrap this into `nn.Module` to pair-up the hypernet with the main network and have a nicer API

```python

class HyperResNet18(nn.Module):

    def __init__(self,
        hypernet_layers: List[]
        ):
        super().__init__()
        mainnet = resnet18()
        modules = find_modules_of_type(mainnet, [nn.Conv2d])
        mainnet = hypernetize(mainnet, modules=modules)

        hypernet = Hypernet(
            input_shapes={'h': (10,)},
            output_shapes=parameter_shapes,
            layer_sizes=[16,64,128],
        )

    def forward(self, input, hyper_input):
        parameters = hypernet(h=hyper_input)

        with mainnet.using_externals(parameters):
            prediction = mainnet(input)

        return input
```


HyperLight let us reuse the pretrained weights by setting them as independent weights


```python

class HyperResNet18(nn.Module):

    def __init__(self,
        hypernet_layers: List[]
        ):
        super().__init__()
        # Load pretrained weights
        mainnet = resnet18(pretrained=True)
        modules = find_modules_of_type(mainnet, [nn.Conv2d])
        mainnet, weights = hypernetize(mainnet, modules=modules, return_values=True)

        # Construct from existing
        hypernet = Hypernet.from_existing(
            weights, # The weights encode shape and intialization
            input_shapes={'h': (10,)},
            output_shapes=parameter_shapes,
            layer_sizes=[16,64,128],
        )

    def forward(self, input, hyper_input):
        parameters = hypernet(h=hyper_input)

        with mainnet.using_externals(parameters):
            prediction = mainnet(input)

        return input
```

## Tutorial

### Concepts

In Hyperlight there are a few new concepts:

- `HyperModule` - Specialized `nn.Module` objects that can hold both regular parameters
and `ExternalParameters` to be predicted by a external hypernetwork.
- `ExternalParameter` - `nn.Parameter` replacements that only stores the required shape of the
externalized parameter. Parameter data can be set and reset with the hypernetwork predictions.
- `HyperNetwork` - `nn.Module` that predicts a main network parameters for a given input

### Defining a `HyperModule` with `ExternalParameter`s

Here is an example of how we can define a hypernetized Linear layer. We need to make sure to
define the `ExternalParameter` properties with their correct shapes

```python
import torch.nn.functional as F
from hyperlight import HyperModule, ExternalParameter

class HyperLinear(HyperModule):
    """Layer that implements a nn.Linear layer but with external parameters
    that will be predicted by a external hypernetwork"""

    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        assert isinstance(in_features, int) and isinstance(out_features, int)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = ExternalParameter(shape=(out_features, in_features))
        if bias:
            self.bias = ExternalParameter(shape=(out_features,))
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)
```

Once defined, we can make use of this module in the following way


```python
>>> layer = HyperLinear(in_features=8, out_features=16)
>>> layer.external_shapes()

```

### Dynamically hypernetizing modules

HyperLight supports **dynamic** HyperModule creation using the `hypernetize` and `hypernetize_single`
routines. The former is recursive while the latter is not.

```python
from hyperlight import hypernetize, hypernetize_single


```


###


## Citation

If you find our work or any of our materials useful, please cite our paper:

```
@article{ortiz2023nonproportional,
  title={Non-Proportional Parametrizations for Stable Hypernetwork Learning},
  author={Jose Javier Gonzalez Ortiz and John Guttag and Adrian Dalca},
  year={2023},
  journal={arXiv:2304.07645},
}
```
