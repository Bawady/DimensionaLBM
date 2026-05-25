import typing

from dimensional_lbm.unit_system_if import ScalarT, VectorT


class VectorCallback(typing.Protocol, typing.Generic[ScalarT, VectorT]):
	def __call__(self, time: ScalarT) -> VectorT:
		...

class ScalarCallback(typing.Protocol, typing.Generic[ScalarT]):
	def __call__(self, time: ScalarT) -> ScalarT:
		...
