import math
from pathlib import Path

import numpy as np

from dimensional_lbm.bgk_lbm import BGKLBM, from_viscosity
from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class LidDrivenCavity(Scenario[BGKLBM]):
	def define(self, lbm: BGKLBM) -> None:
		lbm.width = lbm.us.quantity(128, "m")
		lbm.height = lbm.us.quantity(128, "m")

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
			zou_he.velocity_profile[0, x] = lambda step: self.max_speed * (1 - math.exp(-step**2 / t_ramp**2)) * np.array([1, 0])

		lbm.boundaries += zou_he


# def post_run(self, lbm: LBM) -> None:
# ux = lbm.u[:, lbm.x//2, 0] / self.max_speed
# # In the Ghia paper y grows from bot to top, i.e., our y velocity is inverted with respect to theirs
# uy = -lbm.u[lbm.y//2, :, 1] / self.max_speed
#
# # Need to convert coordinates to the same orientation as in the Ghia paper (their y grows upwards)
# xs = np.arange(lbm.x) / lbm.x
# ys = (lbm.y - np.arange(lbm.y)) / lbm.y
#
# with Path("ldc.csv").open("w") as f:
# f.write("y,ux,x,uy\n")
# f.writelines(f"{ys[i]},{ux[i]},{xs[i]},{uy[i]}\n" for i in range(len(ux)))

if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(1, "m"), (1, "s"), (1, "kg/m**3")]

	sim = LidDrivenCavity(BGKLBM, characteristic_quantities, conversion_mode=Dimensional)
	sim.run(2000, 100, Path("test/ldc"))
