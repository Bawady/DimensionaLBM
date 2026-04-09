import math
import pathlib

import numpy as np

from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional, NonDimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.trt_lbm import TRTLBM, tau_minus_from_magic_param, tau_plus_from_viscosity
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class Turek(Scenario[TRTLBM]):
	def define(self, lbm: TRTLBM) -> None:
		lbm.width = lbm.us.quantity(2.2, "m")
		lbm.height = lbm.us.quantity(41, "cm")

		viscosity = lbm.us.quantity(1e-3, "m**2/s")
		dx = lbm.us.quantity(1, "cm")
		dt = lbm.us.quantity(3, "ms")
		lbm.lattice = D2Q9(dx, dt)

		initial_density = lbm.us.quantity(1, "kg/m**3")
		lbm.density[:, :] = initial_density
		lbm.tau_plus = tau_plus_from_viscosity(viscosity, lbm.lattice)
		lbm.tau_minus = tau_minus_from_magic_param(lbm.tau_plus, lbm.lattice)

		lbm.boundary = ZouHe(lbm)

		# Physical-space coordinate grids.
		y_phys = (lbm.y - 1 - np.arange(lbm.y)) * dx

		lbm.boundary.geometry[0, :] = 1   # top
		lbm.boundary.geometry[-1, :] = 1  # bottom

		cyl_x = lbm.us.magnitude(lbm.us.quantity(0.2, "m") / dx)
		cyl_y = lbm.us.magnitude(lbm.us.quantity(0.2, "m") / dx)
		cyl_radius = lbm.us.magnitude(lbm.us.quantity(5, "cm") / dx)

		# Cyclinder / Circle-shaped obstacle
		xs, ys = np.meshgrid(np.arange(lbm.x), np.arange(lbm.y))
		lbm.boundary.geometry[(xs - int(cyl_x)) ** 2 + (ys - int(cyl_y)) ** 2 < cyl_radius**2] = 1

		# Inlet: velocity profile with ramp-up
		u_max = lbm.us.quantity(0.3, "m/s")
		t_ramp = lbm.us.quantity(0.1, "s")
		for y_idx in range(1, lbm.y - 1):
			y_p = y_phys[y_idx]
			u_in = u_max * (4.0 * y_p * (lbm.height - y_p) / lbm.height**2)
			lbm.boundary.velocity_profile[y_idx, 0] = (
				lambda step, s=u_in, t=t_ramp: s * (1 - math.exp(-float(step / t) ** 2)) * np.array([1, 0])
			)

		# Outlet: constant-pressure outflow
		for y_idx in range(1, lbm.y - 1):
			lbm.boundary.density_profile[y_idx, -1] = initial_density


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [
		(1, "cm"), (3, "ms"), (1, "kg/m**3")
	]

	sim = Turek(TRTLBM, characteristic_quantities)
	sim.run(2000, dump_period=50, dump_dir=pathlib.Path("test/turek"))
