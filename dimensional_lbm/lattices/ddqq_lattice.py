from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic

import numpy as np

from dimensional_lbm.unit_system_if import ScalarT, VectorT

# For circular dependence due to type hints
if TYPE_CHECKING:
	from .ddqq_lattice import DdQqLattice


class DdQqLattice(ABC, Generic[ScalarT, VectorT]):
	D: ClassVar[int]
	Q: ClassVar[int]
	weights: ClassVar[np.ndarray]

	# lattice velocity direction vector (entries only 0,1)
	dir_x: ClassVar[np.ndarray]
	dir_y: ClassVar[np.ndarray]

	dx: ScalarT
	dt: ScalarT

	q: ScalarT

	cs: ScalarT
	cs_n2: ScalarT
	cs_n4: ScalarT

	def __init__(self, dx: ScalarT, dt: ScalarT) -> None:
		self.dx = dx
		self.dt = dt

		self.q = dx / dt

	@classmethod
	def __init_subclass(cls, **kwargs) -> None:  # noqa: ANN003 (PEP 487 doesn't provide )
		super().__init_subclass__(**kwargs)

		if not hasattr(cls, "D") or cls.D is None:
			msg: str = f"{cls.__name__} must define a value for class variable 'D'"
			raise TypeError(msg)

		if not hasattr(cls, "Q") or cls.Q is None:
			msg: str = f"{cls.__name__} must define a value for class variable 'Q'"
			raise TypeError(msg)

		if not hasattr(cls, "weights") or cls.weights is None or not (cls.weights.ndim == 1 and cls.weights.size == cls.Q):
			msg: str = f"{cls.__name__} must set class variable 'weights' to one dimensional array of length {cls.Q}"
			raise TypeError(msg)

		if not hasattr(cls, "dir_x") or cls.dir_x is None or not (cls.dir_x.ndim == 1 and cls.dir_x.size == cls.Q):
			msg: str = f"{cls.__name__} must set class variable 'dir_x' to one dimensional array of length {cls.Q}"
			raise TypeError(msg)

		if not hasattr(cls, "dir_y") or cls.dir_y is None or not (cls.dir_y.ndim == 1 and cls.dir_y.size == cls.Q):
			msg: str = f"{cls.__name__} must set class variable 'dir_y' to one dimensional array of length {cls.Q}"
			raise TypeError(msg)

	@abstractmethod
	def stream_periodic(self, f_new: VectorT, f_old: VectorT) -> None:
		pass

	@abstractmethod
	def stream(self, f_new: VectorT, f_old: VectorT) -> None:
		pass

	@abstractmethod
	def equilibrium(self, density: VectorT, velocity: VectorT, eq: VectorT) -> None:
		pass
