from dimensional_lbm.conversion_mode import Dimensional, MagnitudeOnly, NonDimensional
from dimensional_lbm.unit_system_if import UnitSystem


class UnitConfig[Mode: (Dimensional, NonDimensional, MagnitudeOnly)]:
	__us: UnitSystem

	def __init__(self, us: UnitSystem[Mode]) -> None:
		self.__us = us


class Simulator[Mode: (Dimensional, NonDimensional, MagnitudeOnly)]:
	def __init__(self, us: UnitSystem[Mode]) -> None:
		pass

	@staticmethod
	def basic_unit_config(cls, us: UnitSystem[Dimensional]) -> UnitConfig[Dimensional]:
		return UnitConfig(UnitSystem())


class MySimulator(Simulator):
	pass
