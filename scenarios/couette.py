from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dimensional_lbm.bgk_lbm import BGKLBM, from_viscosity
from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.lattices.d2q9 import D2Q9
from scenarios.scenario import Scenario


class Couette(Scenario[BGKLBM]):
	def define(self, lbm: BGKLBM) -> None:
		lbm.width = lbm.us.quantity(50, "m")
		lbm.height = lbm.us.quantity(20, "m")

		dx = lbm.us.quantity(1, "m")
		dt = lbm.us.quantity(1, "s")
		lbm.lattice = D2Q9(dx, dt)

		viscosity = lbm.us.quantity(0.5, "m**2/s")
		lbm.tau = from_viscosity(viscosity, lbm.lattice)
		lbm.density[:, :] = lbm.us.quantity(1, "kg/m**3")

		lbm.stream = lbm.stream_periodic
		zou_he = ZouHe(lbm)

		self.max_speed = lbm.us.quantity(0.2, "m/s")
		for x in range(lbm.x):
			zou_he.velocity_profile[0, x] = self.max_speed * np.array([1, 0])
			zou_he.geometry[-1, x] = 1

		lbm.boundaries += zou_he

	def post_run(self, lbm: BGKLBM) -> None:
		y_ind = np.arange(lbm.y)
		analytical = self.max_speed * (1 - 1 / (lbm.y - 1) * y_ind)
		data = lbm.u[:, lbm.x // 2, 0]

		plt.plot(y_ind, data)
		plt.plot(y_ind, analytical, "x")
		plt.show()


if __name__ == "__main__":
	sim = Couette(BGKLBM)
	sim.run(500, dump_period=50, dump_dir=Path("couette"))
