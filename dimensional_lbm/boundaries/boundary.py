from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
from dimensional_lbm.unit_system_if import ScalarT, VectorT


class Boundary(ABC, Generic[ScalarT, VectorT]):
	_lattice: DdQqLattice

	def __init__(self, lattice: DdQqLattice) -> None:
		self._lattice = lattice

	@abstractmethod
	def setup(self, geometry: np.ndarray) -> None:
		pass

	@abstractmethod
	def apply_boundaries(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		pass
