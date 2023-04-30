# from .module import VoidModule

# from .xparam import ExternalParameter

# from torch import nn

# nn.ParameterList

# class ExternalParameterList(VoidModule):

#     def __len__(self) -> int:
#         return len(self._external_params)

#     def __iter__(self) -> Iterator[str]:
#         return iter(self._external_parameters.values())

#     # def __iadd__(self, para

#     def append(self, parameter: ExternalParameter):
#         self.register_external(str(len(self)), parameter)

#     # def extend

# class ExternalParameterDict(VoidModule):
#     # TODO
