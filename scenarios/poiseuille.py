import pathlib

import numpy as np

from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class Poiseuille(Scenario):

	def setup(self, lbm: LBM) -> None:
		lbm.dx = lbm.us.quantity(1, "m")
		lbm.dt = lbm.us.quantity(1, "s")

		lbm.width = lbm.us.quantity(50, "m")
		lbm.height = lbm.us.quantity(10, "m")

		lbm.lattice = D2Q9(lbm.dx, lbm.dt)
		lbm.bgk_tau = lbm.viscosity_to_bgk_tau(lbm.us.quantity(0.01, "m**2/s"))

	def define_scenario(self, lbm: LBM) -> None:
		lbm.solid = load_geometry("obstacles.png")

		initial_density = lbm.us.quantity(1, "kg/m**3")
		lbm.density[:, :] = initial_density

		lbm.boundary = ZouHe(lbm)

		# Poiseuille source flow velocity profile
		max_speed = lbm.us.quantity(0.4, "ms")
		for y in range(lbm.y):
			lbm.boundary._velocity_profile[0, y] = max_speed / (lbm.y - 1)**2 * y * (lbm.y-1-y) * np.array([1, 0])

		# Poiseuille sink density profile
		lbm.boundary.density_profile[-1, :] = initial_density


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(1, "m"), (1, "s"), (1, "kg/m**3")]

	Poiseuille(characteristic_quantities, conversion_mode=Dimensional).run(100, 5, pathlib.Path("test/poiseuille"))
