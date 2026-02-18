import pathlib
from abc import ABC, abstractmethod
from typing import overload

from dimensional_lbm.conversion_mode import ConversionMode, Dimensional, MagnitudeOnly, NonDimensional
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import (
	PintQuantityScalar,
	PintQuantityVector,
	ScalarMagnitude,
	ScalarQuantityDefinition,
	UnitSystem,
	VectorMagnitude,
)


class Scenario(ABC):
	_lbm: LBM
	characteristic_quantities: list[ScalarQuantityDefinition] | None = None

	def __init__(
		self, characteristic_quantities: list[ScalarQuantityDefinition] | None = None, conversion_mode: type[ConversionMode] | None = None
	) -> None:
		mode = conversion_mode
		self.characteristic_quantities = characteristic_quantities

		if mode is None:
			mode = NonDimensional if characteristic_quantities else Dimensional

		self._lbm = self._create(conversion_mode=mode)

		self.setup(self._lbm)
		self._lbm.init_sim_params()
		self.define_scenario(self._lbm)
		self._lbm.initialize_distribution_function()
		self._lbm.boundary.setup()

	# TODO: Make init wrapped / decorated
	@abstractmethod
	def setup(self, lbm: LBM) -> None:
		pass

	@abstractmethod
	def define_scenario(self, lbm: LBM) -> None:
		pass

	def post_run(self, lbm: LBM) -> None:
		pass

	def run(self, runs: int, dump_period: int, dump_dir: pathlib.Path) -> None:
		self._lbm.run(runs, dump_period, dump_dir)
		self.post_run(self._lbm)

	# Factory functionality for creating properly typed LBM
	@overload
	def _create(self, conversion_mode: type[Dimensional]) -> LBM[Dimensional, PintQuantityScalar, PintQuantityVector]: ...

	@overload
	def _create(self, conversion_mode: type[NonDimensional]) -> LBM[NonDimensional, ScalarMagnitude, VectorMagnitude]: ...

	@overload
	def _create(self, conversion_mode: type[MagnitudeOnly]) -> LBM[MagnitudeOnly, ScalarMagnitude, VectorMagnitude]: ...

	def _create(self, conversion_mode: type[ConversionMode] = Dimensional) -> LBM:
		lbm = LBM()

		us = UnitSystem()
		if self.characteristic_quantities and conversion_mode == NonDimensional:
			us = us.with_characteristic_quantities(self.characteristic_quantities)
		elif conversion_mode == MagnitudeOnly:
			us = us.with_magnitude_only()

		lbm._set_unit_system(us)
		return lbm
