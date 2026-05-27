import math
import os
import pathlib

import matplotlib as mpl
import matplotlib.image as plt_img
import numpy as np

from dimensional_lbm.boundaries.zero_gradient import ZeroGradient
from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional
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
		dt = lbm.us.quantity(.25, "ms")
		lbm.lattice = D2Q9(dx, dt)

		initial_density = lbm.us.quantity(1, "kg/m**3")
		lbm.density[:, :] = initial_density
		lbm.tau_plus = tau_plus_from_viscosity(viscosity, lbm.lattice)
		lbm.tau_minus = tau_minus_from_magic_param(lbm.tau_plus, lbm.lattice)

		# Physical-space coordinate grids.
		y_phys = (lbm.y - 1 - np.arange(lbm.y)) * dx

		# Inlet and bounce-back boundaries using Zou He
		zou_he = ZouHe(lbm)
		zou_he.geometry[0, :] = 1  # top
		zou_he.geometry[-1, :] = 1  # bottom

		cyl_x = lbm.us.magnitude(lbm.us.quantity(0.2, "m") / dx)
		cyl_y = lbm.us.magnitude(lbm.us.quantity(0.2, "m") / dx)
		cyl_radius = lbm.us.magnitude(lbm.us.quantity(5, "cm") / dx)

		# Cylinder / Circle-shaped obstacle
		xs, ys = np.meshgrid(np.arange(lbm.x), np.arange(lbm.y))
		zou_he.geometry[(xs - int(cyl_x)) ** 2 + (ys - int(cyl_y)) ** 2 < cyl_radius**2] = 1

		# Inlet: velocity profile with ramp-up (Turek case 2D-3)
		u_max = lbm.us.quantity(1.5, "m/s")
		t_ramp = lbm.us.quantity(8, "s")
		for y_idx in range(1, lbm.y - 1):
			y_p = y_phys[y_idx]
			u_in = u_max * (4.0 * y_p * (lbm.height - y_p) / lbm.height**2)
			# 2D-2 requires a velocity ramp as well - the sudden velocity causes pressure / acoustic wave
			zou_he.velocity_profile[y_idx, 0] = lambda step, u=u_in, t=t_ramp: u * (1 - math.exp(-((step / t) ** 2))) * np.array([1, 0])
			# 2D-3
#			zou_he.velocity_profile[y_idx, 0] = lambda step, u=u_in_si, t=t_ramp: u * np.sin(step * np.pi / t) * np.array([1, 0])

		# Outlet: constant-pressure outflow
		zero_grad = ZeroGradient(lbm)

		for y_idx in range(1, lbm.y - 1):
			zero_grad.zero_gradient[y_idx, -2] = ZeroGradient.GradientDirection.RIGHT

		lbm.boundaries += zou_he
		lbm.boundaries += zero_grad

		print(f"Reynolds {lbm.us.magnitude((2 * u_max / 3 * 2 * lbm.us.quantity(5, "cm")) / viscosity)}")
		print(f"Mach {lbm.us.magnitude(u_max / lbm.lattice.cs)}")

	def dump(self, lbm: TRTLBM, dump_dir: os.PathLike) -> None:
		# Dump density and velocity fields
		super().dump(lbm, dump_dir)

		ux = lbm.u[:, :, 0]
		uy = lbm.u[:, :, 1]
		# d u_y / dx - d u_x / dy
		vorticity = (np.gradient(uy, axis=1) - np.gradient(ux, axis=0)) / lbm.lattice.dx
		vorticity_mag = lbm.us.magnitude(vorticity)

		vorticity_scaled = 0.5 + vorticity_mag / max(np.max(np.abs(vorticity_mag)), 1e-12)

		cmap = mpl.colormaps["RdBu"]
		vor_rgba = cmap(vorticity_scaled)
		rgba_masked = np.where((lbm.boundary_geometry() > 0)[..., np.newaxis], (0, 0, 0, 1), vor_rgba)

		plt_img.imsave(pathlib.Path(dump_dir) / f"vorticity_{lbm._runs}.png", rgba_masked, dpi=600)


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(1, "cm"), (.25, "ms"), (1, "kg/m**3")]

	sim = Turek(TRTLBM, characteristic_quantities)
	sim.run(100000, dump_period=250, dump_dir=pathlib.Path("test/turek"))
