import pathlib
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Generic, TypeVar

import matplotlib as mpl
import matplotlib.image as plt_img
import numpy as np

from dimensional_lbm.conversion_mode import ConversionMode, Dimensional, NonDimensional
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import QuantityVector, ScalarQuantityDefinition, UnitSystem

T = TypeVar("T", bound=LBM[Any, Any, Any])


class Scenario(ABC, Generic[T]):
	_lbm: T
	characteristic_quantities: list[ScalarQuantityDefinition] | None = None

	# Subclasses can override this to introduce domain-specific units.
	custom_units: ClassVar[list[str]] = []

	def __init__(
		self,
		lbm: type[T],
		characteristic_quantities: list[ScalarQuantityDefinition] | None = None,
		conversion_mode: type[ConversionMode] | None = None,
	) -> None:
		mode = conversion_mode
		self.characteristic_quantities = characteristic_quantities

		if mode is None:
			mode = NonDimensional if characteristic_quantities else Dimensional

		self._lbm = self._create(lbm, conversion_mode=mode)

		self.define(self._lbm)
		self._lbm.check_parameters_set()
		self._lbm.initialize_populations()
		for boundary in self._lbm.boundaries:
			boundary.setup()

	@abstractmethod
	def define(self, lbm: T) -> None:
		pass

	def run(self, runs: int, dump_period: int=1, dump_dir: pathlib.Path=pathlib.Path()) -> None:
		for i in range(runs):
			if i % dump_period == 0 or i == runs-1:
				self.dump(self._lbm, dump_dir)
			self._lbm.single_step()
		self.post_run(self._lbm)

	def dump(self, lbm: T, dump_dir: pathlib.Path) -> None:
		dump_dir_p = pathlib.Path(dump_dir)
		dump_dir_p.mkdir(exist_ok=True)

		dens: QuantityVector = lbm.density
		dim_dens = lbm.us.dim(dens, "kg/m**3")
		density_mag = lbm.us.magnitude(dim_dens)

		vel: QuantityVector = lbm.u
		vel_dim = lbm.us.dim(vel, "m/s")
		velocity_mag = lbm.us.magnitude(vel_dim)

		cmap = mpl.colormaps["viridis"]
		density_rgba = cmap(density_mag / np.max(density_mag))

		vel_abs = np.sqrt(velocity_mag[:, :, 0] ** 2 + velocity_mag[:, :, 1] ** 2)
		vel_rgba = cmap(vel_abs / lbm.lattice.q)

		dump_data = {"density": density_rgba, "velocity": vel_rgba}

		for name, rgba in dump_data.items():
			rgba_masked = np.where((lbm.boundary_geometry() > 0)[..., np.newaxis], (0, 0, 0, 1), rgba)
			plt_img.imsave(dump_dir_p / f"{name}_{lbm.runs}.png", rgba_masked, dpi=600)

	def post_run(self, lbm: T) -> None:
		pass

	def _create(self, lbm: type[T], conversion_mode: type[ConversionMode] = Dimensional) -> T:
		_lbm = lbm()

		us: UnitSystem[Any] = UnitSystem()
		for unit_definition in self.custom_units:
			us.define_unit(unit_definition)

		if self.characteristic_quantities and conversion_mode == NonDimensional:
			us = us.with_characteristic_quantities(self.characteristic_quantities)

		# Justification: Both is "internal" framework code and in here it is known what the private method does
		_lbm._set_unit_system(us) # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
		return _lbm
