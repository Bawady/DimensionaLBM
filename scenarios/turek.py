import math
import pathlib

import numpy as np

from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.conversion_mode import Dimensional
from dimensional_lbm.lattices.d2q9 import D2Q9
from dimensional_lbm.trt_lbm import TRTLBM, tau_minus_from_magic_param, tau_plus_from_viscosity
from scenarios.scenario import Scenario

# Schäfer–Turek benchmark geometry (all values in SI)
_L = 2.2    # channel length [m]
_H = 0.41   # channel height [m]
_CX = 0.2   # cylinder centre x from inlet [m]
_CY = 0.2   # cylinder centre y from bottom [m]
_R = 0.05   # cylinder radius [m]
_NU = 1e-3  # kinematic viscosity [m^2/s]
_RHO0 = 1.0 # reference density [kg/m^3]
_U_MAX = 0.3  # peak inlet velocity [m/s]  →  Re = (2/3·U_MAX)·D/nu ≈ 20
_DX = 0.01  # lattice spacing [m]  →  cylinder spans ~10 cells


class Turek(Scenario[TRTLBM]):
	def define(self, lbm: TRTLBM) -> None:
		lbm.width = lbm.us.quantity(_L, "m")
		lbm.height = lbm.us.quantity(_H, "m")

		dx = lbm.us.quantity(_DX, "m")
		# dt chosen so tau_plus (lattice) = 0.6: dt = 0.1·dx²/(3·nu)
		dt = lbm.us.quantity(0.1 * _DX**2 / (3.0 * _NU), "s")
		lbm.lattice = D2Q9(dx, dt)

		rho_0 = lbm.us.quantity(_RHO0, "kg/m**3")
		lbm.density[:, :] = rho_0

		nu = lbm.us.quantity(_NU, "m**2/s")
		lbm.tau_plus = tau_plus_from_viscosity(nu, lbm.lattice)
		lbm.tau_minus = tau_minus_from_magic_param(lbm.tau_plus, lbm.lattice)

		lbm.boundary = ZouHe(lbm)

		# Physical-space coordinate grids.
		# Array convention: y_idx=0 is top of channel (physical y = H).
		x_phys = np.arange(lbm.x) * _DX
		y_phys = (lbm.y - 1 - np.arange(lbm.y)) * _DX

		# ── Channel walls (no-slip) ───────────────────────────────────────────
		lbm.boundary.geometry[0, :] = 1   # top
		lbm.boundary.geometry[-1, :] = 1  # bottom

		# ── Circular cylinder (no-slip) ───────────────────────────────────────
		X, Y = np.meshgrid(x_phys, y_phys)
		lbm.boundary.geometry[(X - _CX) ** 2 + (Y - _CY) ** 2 <= _R**2] = 1

		# ── Inlet: parabolic profile with smooth temporal ramp-up ─────────────
		U_max = lbm.us.quantity(_U_MAX, "m/s")
		T_ramp = lbm.us.quantity(1.0, "s")
		for y_idx in range(1, lbm.y - 1):
			y_p = y_phys[y_idx]
			u_in = U_max * (4.0 * y_p * (_H - y_p) / _H**2)
			lbm.boundary.velocity_profile[y_idx, 0] = (
				lambda step, s=u_in, T=T_ramp: s * (1 - math.exp(-float(step / T) ** 2)) * np.array([1, 0])
			)

		# ── Outlet: constant-pressure outflow ─────────────────────────────────
		for y_idx in range(1, lbm.y - 1):
			lbm.boundary.density_profile[y_idx, -1] = rho_0


if __name__ == "__main__":
	characteristic_quantities = [(_DX, "m"), (0.1 * _DX**2 / (3.0 * _NU), "s"), (_RHO0, "kg/m**3")]
	sim = Turek(TRTLBM, characteristic_quantities)
	sim.run(2000, dump_period=100, dump_dir=pathlib.Path("test/turek"))
