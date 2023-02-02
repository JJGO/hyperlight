from .hypermodule import HyperModule

from .base import HyperNet
from .delta import DeltaHyperNet

HyperModule.register_hypernet(HyperNet)
HyperModule.register_hypernet(DeltaHyperNet)

