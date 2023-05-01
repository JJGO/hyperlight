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
{'weight': (16, 8), 'bias': (16,)}
>>> x = torch.zeros(1, 8)

# Layer cannot be used until weights are set
>>> layer(x)
[...]
AttributeError: Uninitialized External Parameter, please set the value first

# We need to set the external weights first
>>> layer.set_externals(weight=torch.rand(size=(16,8)), bias=torch.zeros((16,)))
>>> layer(x).shape
torch.Size([1, 16])

# Once we are done, we reset the external parameter values
>>> layer.reset_externals()
```

Alternatively, we can use the `using_externals` contextmanager that will set and reset
the parameters accordingly

```python
params(weight=torch.rand(size=(16,8)), bias=torch.zeros((16,)))
with layer.using_externals(params):
    y = layer(x)
```

### Dynamically hypernetizing modules

HyperLight supports **dynamic** HyperModule creation using the `hypernetize` helper.
We need to specify what parameters we want to remove fromt the module and convert to
`ExternalParameter` objects.

```python
>>> from torch import nn
>>> from hyperlight import hypernetize, hypernetize_single

>>> layer = nn.Linear(in_features=8, out_features=16)
>>> layer = hypernetize(layer, parameters=[layer.weight, layer.bias])
>>> layer
HypernetizedLinear()
>>> layer.external_shapes()
{'weight': (16, 8), 'bias': (16,)}
```

`hypernetize` is recursive, and supports entire modules being specified


```python
>>> model = nn.Sequential(OrderedDict({
    'conv': nn.Conv2d(3,128,3),
    'norm': nn.BatchNorm2d(128),
    'relu': nn.ReLU(),
    'pool': nn.AdaptiveAvgPool2d((1,1)),
    'out': nn.Linear(128, 10)
}))

>>> model = hypernetize(model, modules=[model.conv, model.out])
>>>  model.external_shapes()
{'conv.weight': (128, 3, 3, 3),
 'conv.bias': (128,),
 'out.weight': (10, 128),
 'out.bias': (10,)}
```

### Finding modules and parameters

Additionally, Hyperlight provides several routines to recursively search for parameters and modules to feed
into `hypernetize`:

- `find_modules_of_type(model, module_types)` – To find modules from a certain type,
e.g. `nn.Linear` or `nn.Conv2d`
- `find_modules_from_patterns(model, globs=None, regex=None)` – To find modules matching
specific patterns usingglobs, e.g. `*.conv`; or regexes `e.g. layer[1-3].*conv`
- `find_parameters_from_patterns(model, globs=None, regex=None)` – To find parameter
matching specific patterns.

Some examples on a ResNet18 architecture:

```python
>>> from torchvision.models import resnet18
>>> from hyperlight import find_modules_of_type
>>> model = resnet18()
# All convolutions
>>> find_modules_of_type(model, [nn.Conv2d])
{'conv1': Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
 'layer1.0.conv1': Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
 'layer1.0.conv2': Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
 ...

# First convolution of every ResNet block
>>> find_modules_from_patterns(model, regex=['^layer\d.0.conv1'])
{'layer1.0.conv1': Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
 'layer2.0.conv1': Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
 'layer3.0.conv1': Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
 'layer4.0.conv1': Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)}

# Getting just the convolutional weights of the first block (no biases)
>>> find_parameters_from_patterns(model, globs=['layer1*conv*.weight']).keys()
dict_keys(['layer1.0.conv2.weight', 'layer1.0.conv1.weight', 'layer1.1.conv1.weight', 'layer1.1.conv2.weight'])
```


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
