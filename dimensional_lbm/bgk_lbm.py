from typing import TYPE_CHECKING, cast

from dimensional_lbm.conversion_mode import ModeT
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarT, VectorT

if TYPE_CHECKING:
	from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice


def from_viscosity(viscosity: ScalarT, lattice: DdQqLattice[ScalarT, VectorT]) -> ScalarT:
	"""Compute the BGK relaxation time from the kinematic viscosity."""
	# Pint's scalar operator stubs widen the result type, so restore the known ScalarT.
	return cast("ScalarT", lattice.dt / 2 + viscosity * lattice.cs_n2)

class BGKLBM(LBM[ModeT, ScalarT, VectorT]):
	tau: ScalarT

	def collide(self) -> None:
		relax_factor = self.dt / self.tau
		self.fcoll = cast("VectorT", (1 - relax_factor) * self.f + relax_factor * self.feq)
