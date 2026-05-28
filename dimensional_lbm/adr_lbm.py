from dataclasses import dataclass

import numpy as np

from dimensional_lbm.boundaries.boundary import Boundary
from dimensional_lbm.conversion_mode import ModeT
from dimensional_lbm.lbm import LBM
from dimensional_lbm.unit_system_if import ScalarT, VectorT


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

	substrate: AdrSpecies
	bacteria: AdrSpecies

	_species: list[AdrSpecies]

	def __init__(self) -> None:
		super().__init__()
		self._species = []

	def initialize_fields(self) -> None:
		# Reinitialize existing species when grid dimensions change.
		for s in self._species:
			s.density = np.zeros((self._y, self._x))
			arr = np.zeros((self._lattice.Q, self._y, self._x))
			s.f, s.feq, s.fcoll = arr, arr.copy(), arr.copy()

	def add_species(self, tau: ScalarT, unit: str) -> AdrSpecies:
		"""Add a scalar species with the given BGK relaxation time."""
		arr = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x)), unit)
		species = AdrSpecies(
			density=self.us.quantity(np.zeros((self._y, self._x)), unit),
			f=arr, feq=arr.copy(), fcoll=arr.copy(),
			tau=tau,
		)
		self._species.append(species)
		return species

	def initialize_populations(self) -> None:
		for s in self._species:
			for i, w in enumerate(self._lattice.weights):
				s.f[i] = s.feq[i] = w * s.density

	def collide(self) -> None:
		for s in self._species:
			relax = float(self.us.magnitude(self._lattice.dt / s.tau))
			s.fcoll = (1 - relax) * s.f + relax * s.feq

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
