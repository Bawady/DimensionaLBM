import math
import pathlib

import numpy as np

from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional, NonDimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class LidDrivenCavity(Scenario):

	def define_scenario(self, lbm: LBM) -> None:
		lbm.dx = lbm.us.quantity(1, "mm")
		lbm.dt = lbm.us.quantity(1, "us")

		lbm.width = lbm.us.quantity(100, "mm")
		lbm.height = lbm.us.quantity(150, "mm")

		lbm.lattice = D2Q9(lbm.dx, lbm.dt)

		reynolds = 100
		lbm.bgk_tau = lbm.viscosity_to_bgk_tau(lbm.us.quantity(250, "m/s") * lbm.width / reynolds)

		initial_density = lbm.us.quantity(1, "kg/m**3")
		lbm.density[:, :] = initial_density

		lbm.boundary = ZouHe(lbm)
		lbm.boundary.geometry[:, 0] = 1
		lbm.boundary.geometry[:, -1] = 1

		max_speed = lbm.us.quantity(0.1, "m/s")
		for x in range(lbm.x):
			lbm.boundary.velocity_profile[0, x] = lambda step: max_speed * (1 - math.exp(-step**2 / (2 * 100))) * np.array([1, 0])
		for y in range(lbm.y):
			lbm.boundary.velocity_profile[y, 0] = lambda step: -.75*max_speed * (1 - math.exp(-step**2 / (2 * 100))) * np.array([0, 1])


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(10, "mm"), (4, "us"), (1, "kg/m**3")]

	sim = LidDrivenCavity(characteristic_quantities, conversion_mode=NonDimensional)
	sim.run(5000, 100, pathlib.Path("test/ldc"))
