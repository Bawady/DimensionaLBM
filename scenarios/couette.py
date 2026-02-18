import pathlib

import numpy as np

from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class Couette(Scenario):
	def setup(self, lbm: LBM) -> None:
		lbm.dx = lbm.us.quantity(1, "m")
		lbm.dt = lbm.us.quantity(1, "s")

		lbm.width = lbm.us.quantity(50, "m")
		lbm.height = lbm.us.quantity(20, "m")

		lbm.lattice = D2Q9(lbm.dx, lbm.dt)
		lbm.bgk_tau = lbm.viscosity_to_bgk_tau(lbm.us.quantity(0.5, "m**2/s"))

	def define_scenario(self, lbm: LBM) -> None:
		lbm.density[:, :] = lbm.us.quantity(1, "kg/m**3")

		lbm.stream = lbm.stream_periodic

		lbm.boundary = ZouHe(lbm)

		max_speed = lbm.lattice.q
		for x in range(lbm.x):
			lbm.boundary.velocity_profile[0, x] = max_speed * np.array([1, 0])
			lbm.boundary.geometry[-1, x] = 1


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(1, "m"), (1, "s"), (1, "kg/m**3")]

	Couette(characteristic_quantities, conversion_mode=Dimensional).run(100, 5, pathlib.Path("test/couette"))
