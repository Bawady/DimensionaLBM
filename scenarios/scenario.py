import os
import pathlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import matplotlib.image as plt_img
import numpy as np
from matplotlib import cm

from dimensional_lbm.conversion_mode import ConversionMode, Dimensional, MagnitudeOnly, NonDimensional
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import (
	ScalarQuantityDefinition,
	UnitSystem,
)

T = TypeVar("T", bound=LBM)

class Scenario(ABC, Generic[T]):
	_lbm: T
	characteristic_quantities: list[ScalarQuantityDefinition] | None = None

	def __init__(
		self, lbm: type[T], characteristic_quantities: list[ScalarQuantityDefinition] | None = None, conversion_mode: type[ConversionMode] | None = None
	) -> None:
		mode = conversion_mode
		self.characteristic_quantities = characteristic_quantities

		if mode is None:
			mode = NonDimensional if characteristic_quantities else Dimensional

		self._lbm = self._create(lbm, conversion_mode=mode)

		self.define_scenario(self._lbm)
		self._lbm.check_parameters_set()
		self._lbm.initialize_populations()
		self._lbm.boundary.setup()

	@abstractmethod
	def define_scenario(self, lbm: T) -> None:
		pass

	def run(self, runs: int, dump_period: int=1, dump_dir: pathlib.Path=pathlib.Path(".")) -> None:
		for i in range(runs):
			if i % dump_period == 0 or i == runs-1:
				self.dump(self._lbm, dump_dir)
			self._lbm.single_step()
		self.post_run(self._lbm)

	def dump(self, lbm: T, dir: os.PathLike) -> None:
		dump_dir_p = pathlib.Path(dir)
		dump_dir_p.mkdir(exist_ok=True)

		density_mag = lbm.us.magnitude(lbm.us.dim(lbm.density, "kg/m**3"))
		velocity_mag = lbm.us.magnitude(lbm.us.dim(lbm.u, "m/s"))

		cmap = cm.get_cmap("viridis")
		density_rgba = cmap(density_mag / np.max(density_mag))

		vel_abs = np.sqrt(velocity_mag[:, :, 0] ** 2 + velocity_mag[:, :, 1] ** 2)
		vel_rgba = cmap(vel_abs / np.max(vel_abs)) if np.max(vel_abs) > 0 else cmap(vel_abs)

		dump_data = {"density": density_rgba, "velocity": vel_rgba}

		for name, rgba in dump_data.items():
			plt_img.imsave(dump_dir_p / f"{name}_{lbm._runs}.png", rgba, dpi=600)

	def post_run(self, lbm: T) -> None:
		pass

	def _create(self, lbm: type[T], conversion_mode: type[ConversionMode] = Dimensional) -> T:
		_lbm = lbm()

		us = UnitSystem()
		if self.characteristic_quantities and conversion_mode == NonDimensional:
			us = us.with_characteristic_quantities(self.characteristic_quantities)
		elif conversion_mode == MagnitudeOnly:
			us = us.with_magnitude_only()

		_lbm._set_unit_system(us)  # noqa: SLF001 Justification: Both is "internal" framework code and in here it is known what the private method does
		return _lbm
