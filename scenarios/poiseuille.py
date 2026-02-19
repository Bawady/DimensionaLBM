import pathlib

import matplotlib.pyplot as plt
import numpy as np

from dimensional_lbm.boundaries.boundary import load_geometry
from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional, NonDimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class Poiseuille(Scenario):

	def setup(self, lbm: LBM) -> None:
		lbm.dx = lbm.us.quantity(1, "m")
		lbm.dt = lbm.us.quantity(1, "s")

		lbm.width = lbm.us.quantity(80, "m")
		lbm.height = lbm.us.quantity(20, "m")

		lbm.lattice = D2Q9(lbm.dx, lbm.dt)
		lbm.bgk_tau = lbm.viscosity_to_bgk_tau(lbm.us.quantity(0.026, "m**2/s"))

	def define_scenario(self, lbm: LBM) -> None:

		initial_density = lbm.us.quantity(1, "kg/m**3")
		lbm.density[:, :] = initial_density

		lbm.boundary = ZouHe(lbm)
		lbm.boundary.geometry = load_geometry("pipe.png")

		max_speed = lbm.us.quantity(0.8, "m/s")
		for y in range(lbm.y-1):
			# Poiseuille source velocity profile
			lbm.boundary.velocity_profile[y, 0] = max_speed / (lbm.y - 1)**2 * y * (lbm.y-1-y) * np.array([1, 0])
			# Poiseuille sink density profile
			lbm.boundary.density_profile[y, -1] = initial_density

	def post_run (self, lbm: LBM) -> None:
		y_ind = np.arange(lbm.y)
		ref = 4 / (lbm.y-1)**2*y_ind * (lbm.y-1-y_ind)
		data = lbm.u[:, lbm.x//2, 0] / lbm.us.quantity(0.2, "m/s")

		plt.plot(ref, y_ind, label="ref")
		plt.plot(data, y_ind, label="lbm", marker="o")
		plt.show()


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(1, "m"), (1, "s"), (1, "kg/m**3")]

	sim = Poiseuille(characteristic_quantities, conversion_mode=NonDimensional)
	sim.run(5000, 100, pathlib.Path("test/poiseuille"))
