import math
import sys
from pathlib import Path

import numpy as np

from dimensional_lbm.bgk_lbm import BGKLBM, from_viscosity
from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import ConversionMode, Dimensional, NonDimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class LidDrivenCavity(Scenario[BGKLBM]):

	def __init__(
		self,
		lbm: type[BGKLBM],
		size: int,
		characteristic_quantities: list[ScalarQuantityDefinition] | None = None,
		conversion_mode: type[ConversionMode] | None = None
	) -> None:
		mode = conversion_mode
		self.characteristic_quantities = characteristic_quantities

		self.size = size

		if mode is None:
			mode = NonDimensional if characteristic_quantities else Dimensional

		self._lbm = self._create(lbm, conversion_mode=mode)

		self.define(self._lbm)
		self._lbm.check_parameters_set()
		self._lbm.initialize_populations()
		for boundary in self._lbm.boundaries:
			boundary.setup()


	def define(self, lbm: BGKLBM) -> None:
		lbm.width = lbm.us.quantity(self.size, "m")
		lbm.height = lbm.us.quantity(self.size, "m")

		dx = lbm.us.quantity(1, "m")
		dt = lbm.us.quantity(1, "s")
		lbm.lattice = D2Q9(dx, dt)

		initial_density = lbm.us.quantity(1, "kg/m**3")
		lbm.density[:, :] = initial_density

		reynolds = 100
		viscosity = lbm.us.quantity(0.256, "m/s") * lbm.width / reynolds
		lbm.tau = from_viscosity(viscosity, lbm.lattice)

		zou_he = ZouHe(lbm)
		zou_he.geometry[:, 0] = 1
		zou_he.geometry[:, -1] = 1
		zou_he.geometry[-1, :] = 1

		self.max_speed = lbm.us.quantity(0.256, "m/s")
		t_ramp = lbm.us.quantity(30, "us")
		for x in range(lbm.x):
			zou_he.velocity_profile[0, x] = lambda step, s=self.max_speed, t=t_ramp: (
				s * (1 - math.exp(-((step / t) ** 2))) * np.array([1, 0])
			)

		lbm.boundaries += zou_he

if __name__ == "__main__":

	if len(sys.argv) != 3:
		msg: str = f"Expected 2 CLI arguments (mode, size) but got {len(sys.argv)-1}: {sys.argv[1:] if len(sys.argv) > 1 else []}"
		raise ValueError(msg)

	mode = NonDimensional if sys.argv[1] == "NonDimensional" else Dimensional
	size = int(sys.argv[2])

	characteristic_quantities: list[ScalarQuantityDefinition] = [(1, "m"), (1, "s"), (1, "kg/m**3")]

	sim = LidDrivenCavity(BGKLBM, size, characteristic_quantities, conversion_mode=mode)
	sim.run(2000, 2000, Path("test/ldc"))
