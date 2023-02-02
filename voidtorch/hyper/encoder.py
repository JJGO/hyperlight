import math
import torch


class Encoder:

    _MODES = {
        None: 1,
        "z": 1,
        "1-z": 1,
        "1+z": 1,
        "2z-1": 1,
        "z|z": 2,
        "z|1-z": 2,
        "cos|sin": 2,
    }

    def __init__(self, mode: str):
        assert mode in self._MODES, "Mode not recognized"
        self.mode = mode

    @property
    def factor(self):
        return self._MODES[self.mode]

    def __call__(self, flat_input):
        z = flat_input
        if self.mode is None or self.mode == "z":
            return z
        if self.mode == "1-z":
            return 1 - z
        if self.mode == "1+z":
            return 1 + flat_input
        if self.mode == "2z-1":
            return 2 * flat_input - 1
        if self.mode == "z|z":
            return torch.cat([z, z], dim=-1)
        if self.mode == "z|1-z":
            return torch.cat([z, 1 - z], dim=-1)
        if self.mode == "cos|sin":
            a = math.pi * z / 2
            return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        raise NotImplementedError(f"Mode {self.mode} not implemented")

