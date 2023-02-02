# Basic Building blocks for VoidModules
from .xparam import ExternalParameter
from .voidmodule import VoidModule

# Void reimplementations of nn modules
from .linear import VoidLinear
from .conv import VoidConv1d, VoidConv2d, VoidConv3d
