from typing import cast

import numpy as np

from dimensional_lbm.conversion_mode import ModeT
from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarT, VectorT


def tau_plus_from_viscosity(viscosity: ScalarT, lattice: DdQqLattice) -> ScalarT:
	return lattice.dt / 2 + viscosity * lattice.cs_n2


def tau_minus_from_magic_param(tau_plus: ScalarT, lattice: DdQqLattice, magic: float = 3 / 16) -> ScalarT:
	"""Compute tau_minus from the magic parameter Λ = (τ+/dt − ½)(τ−/dt − ½)."""
	return lattice.dt / 2 + magic * lattice.dt**2 / (tau_plus - lattice.dt / 2)


def _compute_opposite_lattice_indices(lattice: DdQqLattice) -> np.ndarray:
	Q = lattice.Q
	dir_x = lattice.dir_x
	dir_y = lattice.dir_y
	opposite = np.zeros(Q, dtype=int)
	for i in range(Q):
		for j in range(Q):
			if dir_x[j] == -dir_x[i] and dir_y[j] == -dir_y[i]:
				opposite[i] = j
				break
	return opposite

class TRTLBM(LBM[ModeT, ScalarT, VectorT]):
	tau_plus: ScalarT
	tau_minus: ScalarT
	_op_ind: np.ndarray

	def initialize_fields(self) -> None:
		super().initialize_fields()
		self._op_ind = _compute_opposite_lattice_indices(self.lattice)

	def collide(self) -> None:
		f_plus = 0.5 * cast("VectorT", (self.f + self.f[self._op_ind]))
		f_minus = 0.5 * (self.f - self.f[self._op_ind])
		feq_plus = 0.5 * cast("VectorT", (self.feq + self.feq[self._op_ind]))
		feq_minus = 0.5 * (self.feq - self.feq[self._op_ind])
		relax_plus = self.dt / self.tau_plus
		relax_minus = self.dt / self.tau_minus
		self.fcoll = cast("VectorT", self.f - relax_plus * (f_plus - feq_plus) - relax_minus * (f_minus - feq_minus))
