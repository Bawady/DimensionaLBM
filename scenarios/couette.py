from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dimensional_lbm.bgk_lbm import BGKLBM, tau_from_viscosity
from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional, NonDimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class Couette(Scenario[BGKLBM]):
	def define_scenario(self, lbm: BGKLBM) -> None:
		lbm.width = lbm.us.quantity(50, "m")
		lbm.height = lbm.us.quantity(20, "m")

		dx = lbm.us.quantity(1, "m")
		dt = lbm.us.quantity(1, "s")
		lbm.lattice = D2Q9(dx, dt)

		lbm.density[:, :] = lbm.us.quantity(1, "kg/m**3")
		lbm.tau = tau_from_viscosity(lbm.us.quantity(0.5, "m**2/s"), lbm.lattice)

		lbm.stream = lbm.stream_periodic
		lbm.boundary = ZouHe(lbm)

		max_speed = lbm.lattice.q
		for x in range(lbm.x):
			lbm.boundary.velocity_profile[0, x] = max_speed * np.array([1, 0])
			lbm.boundary.geometry[-1, x] = 1

	def post_run(self, lbm: BGKLBM) -> None:
		y_ind = np.arange(lbm.y)
		analytical = lbm.lattice.q * (1 - 1 / (lbm.y - 1) * y_ind)
		data = lbm.u[:, lbm.x//2, 0]

		plt.plot(y_ind, data)
		plt.plot(y_ind, analytical, "x")
		plt.show()

if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(1, "m"), (1, "s"), (2, "kg/m**3")]

	sim = Couette(BGKLBM, characteristic_quantities, conversion_mode=Dimensional)
	sim.run(500, dump_period=50, dump_dir=Path("test/couette"))

