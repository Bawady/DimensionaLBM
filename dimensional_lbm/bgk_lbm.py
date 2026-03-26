from typing import Generic, cast

import numpy as np

from dimensional_lbm.conversion_mode import ModeT
from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarT, VectorT


def tau_from_viscosity(viscosity: ScalarT, lattice: DdQqLattice) -> ScalarT:
	return lattice.dt / 2 + viscosity * lattice.cs_n2

class BGKLBM(LBM[ModeT, ScalarT, VectorT]):
	f: VectorT
	fcoll: VectorT
	feq: VectorT
	u: VectorT
	density: VectorT

	_tau: ScalarT
	_omega: ScalarT

	@property
	def tau(self) -> ScalarT:
		"""BGK relaxation time, related to the fluid viscosity."""
		return self._tau

	@tau.setter
	def tau(self, tau: ScalarT) -> None:
		self._tau = tau
		self._omega = 1 / tau

	def initialize_populations(self) -> None:
		"""Initialize distribution functions to equilibrium.

		Computes the equilibrium distribution based on the current macroscopic
		fields (density, u) and sets f = f_eq as the initial condition.
		"""
		self.equilibrium()
		self.f = cast("VectorT", self.feq.copy())

	def equilibrium(self) -> None:
		"""Compute equilibrium distribution functions.

		Calculates the Maxwell-Boltzmann equilibrium distribution expanded
		to second order in velocity:
			f_eq_i = w_i * density * (1 + (c_i . u)/cs^2 + (c_i . u)^2/(2*cs^4) - u^2/(2*cs^2))

		where w_i are the lattice weights and cs is the speed of sound.
		"""
		cs_n2 = self.lattice.cs_n2
		vel_x = self.lattice.dir_x * self.lattice.q
		vel_y = self.lattice.dir_y * self.lattice.q
		ws = self.lattice.weights
		u_sq = cs_n2 / 2.0 * (self.u[:, :, 0] ** 2 + self.u[:, :, 1] ** 2)
		for i in range(self.lattice.Q):
			lin_term = cs_n2 * (vel_x[i] * self.u[:, :, 0] + vel_y[i] * self.u[:, :, 1])
			self.feq[i] = ws[i] * self.density * (1 + lin_term + lin_term**2 / 2.0 - u_sq)

	def collide(self) -> None:
		"""Perform BGK collision step.

		Applies the Bhatnagar-Gross-Krook (BGK) single relaxation time collision:
			f_coll = (1 - omega*dt) * f + omega*dt * f_eq

		where omega = 1/tau is the relaxation frequency.
		"""
		relax_factor = self.dt * self._omega
		self.fcoll = cast("VectorT", (1 - relax_factor) * self.f + relax_factor * self.feq)
