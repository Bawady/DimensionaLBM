from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np

from dimensional_lbm.conversion_mode import ModeT
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarT, VectorT

if TYPE_CHECKING:
	from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice


def from_diffusivity(diffusivity: ScalarT, lattice: DdQqLattice[ScalarT, VectorT]) -> ScalarT:
	"""Compute the diffusive BGK relaxation time from the diffusivity."""
	# Pint's scalar operator stubs widen the result type, so restore the known ScalarT.
	return cast("ScalarT", 2 / 3 * diffusivity * lattice.cs_n2 + lattice.dt / 2)


@dataclass
class AdrSpecies[ScalarT, VectorT]:
	"""Scalar species for advection-diffusion-reaction simulations."""
	density: VectorT
	f: VectorT
	feq: VectorT
	fcoll: VectorT
	tau: ScalarT


class AdrLBM(LBM[ModeT, ScalarT, VectorT]):
	"""Multi-species LBM for advection-diffusion-reaction systems.

	Each species undergoes pure diffusion (feq_i = w_i * density, no bulk flow).
	Reactions are applied after collision by setting lbm.react to a callback.
	Uses periodic streaming — no boundary conditions are required.
	"""

	nutrients: AdrSpecies[ScalarT, VectorT]
	bacteria:  AdrSpecies[ScalarT, VectorT]
	inactive:  VectorT

	_species: list[AdrSpecies[ScalarT, VectorT]]

	def __init__(self) -> None:
		super().__init__()
		self._species = []

	def add_species(self, tau: ScalarT, unit: str) -> AdrSpecies[ScalarT, VectorT]:
		"""Add a scalar species with the given BGK relaxation time."""
		arr = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x)), unit)
		# us.quantity cannot infer the unbound VectorT and resolves it to its default, so re-tie
		# the species to this LBM's VectorT to match the _species list.
		species = cast("AdrSpecies[ScalarT, VectorT]", AdrSpecies(
			density=self.us.quantity(np.zeros((self._y, self._x)), unit),
			f=arr, feq=arr.copy(), fcoll=arr.copy(),
			tau=tau,
		))
		self._species.append(species)
		return species

	def initialize_populations(self) -> None:
		for s in self._species:
			for i, w in enumerate(self._lattice.weights):
				s.f[i] = s.feq[i] = w * s.density

	def collide(self) -> None:
		for s in self._species:
			relax = float(self.us.magnitude(self._lattice.dt / s.tau))
			# Pint's float * VectorT operator stubs widen the result; restore the known VectorT.
			s.fcoll = cast("VectorT", (1 - relax) * s.f + relax * s.feq)

	def react(self) -> None:
		"""Apply reaction source terms to fcoll. Assign a closure to lbm.react to override."""

	def single_step(self) -> None:
		for s in self._species:
			s.density = s.f.sum(axis=0)
		for s in self._species:
			for i, w in enumerate(self._lattice.weights):
				s.feq[i] = w * s.density
		self.collide()
		self.react()
		for s in self._species:
			self._lattice.stream_periodic(s.f, s.fcoll)
		self._runs += 1
