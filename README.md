# HyperLight

> Hypernetworks in Pytorch made easy

## TL;DR

HyperLight is a Pytorch library designed to make implementing hypernetwork models easy and painless.
What sets HyperLight apart from other hypernetwork implementations:

- **Bring your own architecture**. HyperLight lets you reuse your existing model code.
- **Principled Parametrizations and Initializations**. Default networks can have unstable training dynamics, HyperLight has good defaults that lead to improved training [1].
- **Easy weight reuse**. Once a model has its weights set, it can be used many times.
- **_Pytorch-nic_** API design. Parameters are not treated as inputs to layers, preventing lots of code rewriting from happening.

[1] [Non-Proportional Parametrizations for Stable Hypernetwork Learning](https://arxiv.org/abs/2304.07645)

## Installation

HyperLight can be installed via `pip`. For the stable version:

```
pip install hyperlight
```

Or for the latest version:

```
pip install git+https://github.com/JJGO/hyperlight.git
```


## Getting Started

The main advantage of HyperLight is that it allows to easily reuse existing networks without having to redo the model code.

For example, here's how we can write a Bayesian Neural Hypernetwork for the resnet50 architecture.

```python
from torchvision.models import resnet50
from hyperlight import voidify, Hypernet

# First we instantiate the main network and
# hyperparametrize all convolutional weights
mainnet = resnet50()
mainnet = voidify(mainnet, module_types=[nn.Conv2d])

# Now, we get the spec of the weights we need to predict
parameter_shapes = mainnet.external_shapes()

# We can predict these shapes any way we want,
# but hyperlight provides hypernetwork models
hyperparam_shape = {'h': (10,)} # 10-dim input
hypernet = Hypernet(
    input_shapes=hyperparam_shape,
    output_shapes=parameter_shapes,
    layer_sizes=[16,64,128],
)

# Now, instead of model(input) we first predict the main network weights
parameters = hypernet(h=hyperpameter_input)
# and then use the main network
with mainnet.using_externals(parameters):
    # within this context manager, the weights are accesible
    prediction = mainnet(input)

    # after this point, weights are removed
```

We can also wrap this into `nn.Module` to hide this complexity

```python

class HyperResNet50(nn.Module):

    def __init__(self,
        hypernet_layers: List[]
        ):
        super().__init__()
        original = resnet50()
        self.main = voidify(original, module_types=[nn.Conv2d])

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

## Introduction

In Hyperlight there are a few new concepts

A `VoidModule` is a pytorch module that requires some parameters

> TODO: more to come

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
