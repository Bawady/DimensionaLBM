from typing import Generic, cast

import numpy as np

from dimensional_lbm.conversion_mode import ModeT
from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarT, VectorT


def tau_from_viscosity(viscosity: ScalarT, lattice: DdQqLattice) -> ScalarT:
	return lattice.dt / 2 + viscosity * lattice.cs_n2

class BGKLBM(LBM[ModeT, ScalarT, VectorT]):
	_tau: ScalarT
	_omega: ScalarT

	@property
	def tau(self) -> ScalarT:
		return self._tau

	@tau.setter
	def tau(self, tau: ScalarT) -> None:
		self._tau = tau
		self._omega = 1 / tau

	def equilibrium(self) -> None:
		cs_n2 = self.lattice.cs_n2
		vel_x = self.lattice.dir_x * self.lattice.q
		vel_y = self.lattice.dir_y * self.lattice.q
		ws = self.lattice.weights
		u_sq = cs_n2 / 2.0 * (self.u[:, :, 0] ** 2 + self.u[:, :, 1] ** 2)
		for i in range(self.lattice.Q):
			lin_term = cs_n2 * (vel_x[i] * self.u[:, :, 0] + vel_y[i] * self.u[:, :, 1])
			self.feq[i] = ws[i] * self.density * (1 + lin_term + lin_term**2 / 2.0 - u_sq)

	def collide(self) -> None:
		relax_factor = self.dt * self._omega
		self.fcoll = cast("VectorT", (1 - relax_factor) * self.f + relax_factor * self.feq)
