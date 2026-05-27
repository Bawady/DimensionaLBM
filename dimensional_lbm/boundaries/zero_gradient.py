from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic

import numpy as np

from dimensional_lbm.boundaries.boundary import Boundary
from dimensional_lbm.boundaries.wall_detector import WallDetector
from dimensional_lbm.unit_system_if import ScalarT, VectorT

if TYPE_CHECKING:
	from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
	from dimensional_lbm.lbm import LBM


class _GradientArray(np.ndarray):
	"""ndarray subclass that silently converts Enum values to their integer value on item assignment."""

	def __setitem__(self, key: Any, value: Any) -> None:
		if isinstance(value, Enum):
			value = value.value
		super().__setitem__(key, value)


class _ZeroGradientBoundary(ABC, Generic[ScalarT, VectorT]):
	"""Abstract base class for individual zero-gradient boundary segments."""

	@abstractmethod
	def apply(self, f: VectorT, rho: VectorT, u: VectorT, time: ScalarT) -> None:
		pass


class _LeftGradient(_ZeroGradientBoundary[ScalarT, VectorT]):
	"""Left boundary (fluid to the right): copies populations from x+1."""

	def __init__(self, x: int, y0: int, y1: int) -> None:
		self._x = x
		self._y0 = y0
		self._y1 = y1

	def apply(self, f: VectorT, rho: VectorT, u: VectorT, time: ScalarT) -> None:  # noqa: ARG002
		x, y0, y1 = self._x, self._y0, self._y1
		f[:, y0:y1, x] = f[:, y0:y1, x + 1]
		rho[y0:y1, x] = rho[y0:y1, x + 1]
		u[y0:y1, x] = u[y0:y1, x + 1]


class _RightGradient(_ZeroGradientBoundary[ScalarT, VectorT]):
	"""Right boundary (fluid to the left): copies populations from x-1."""

	def __init__(self, x: int, y0: int, y1: int) -> None:
		self._x = x
		self._y0 = y0
		self._y1 = y1

	def apply(self, f: VectorT, rho: VectorT, u: VectorT, time: ScalarT) -> None:  # noqa: ARG002
		x, y0, y1 = self._x, self._y0, self._y1
		f[:, y0:y1, x] = f[:, y0:y1, x - 1]
		rho[y0:y1, x] = rho[y0:y1, x - 1]
		u[y0:y1, x] = u[y0:y1, x - 1]


class _TopGradient(_ZeroGradientBoundary[ScalarT, VectorT]):
	"""Top boundary (fluid below): copies populations from y+1."""

	def __init__(self, y: int, x0: int, x1: int) -> None:
		self._y = y
		self._x0 = x0
		self._x1 = x1

	def apply(self, f: VectorT, rho: VectorT, u: VectorT, time: ScalarT) -> None:  # noqa: ARG002
		y, x0, x1 = self._y, self._x0, self._x1
		f[:, y, x0:x1] = f[:, y + 1, x0:x1]
		rho[y, x0:x1] = rho[y + 1, x0:x1]
		u[y, x0:x1] = u[y + 1, x0:x1]


class _BottomGradient(_ZeroGradientBoundary[ScalarT, VectorT]):
	"""Bottom boundary (fluid above): copies populations from y-1."""

	def __init__(self, y: int, x0: int, x1: int) -> None:
		self._y = y
		self._x0 = x0
		self._x1 = x1

	def apply(self, f: VectorT, rho: VectorT, u: VectorT, time: ScalarT) -> None:  # noqa: ARG002
		y, x0, x1 = self._y, self._x0, self._x1
		f[:, y, x0:x1] = f[:, y - 1, x0:x1]
		rho[y, x0:x1] = rho[y - 1, x0:x1]
		u[y, x0:x1] = u[y - 1, x0:x1]


class ZeroGradient(Boundary[ScalarT, VectorT]):
	_boundaries: list[_ZeroGradientBoundary]
	_lattice: DdQqLattice
	_lbm: LBM
	zero_gradient: _GradientArray

	class GradientDirection(Enum):
		LEFT  = 0
		RIGHT = 1
		UP    = 2
		DOWN  = 3

	def __init__(self, lbm: LBM) -> None:
		self._lattice = lbm.lattice
		self._lbm = lbm

		self._boundaries = []
		self.zero_gradient = np.zeros((lbm.y, lbm.x)).view(_GradientArray)

	def get_geometry(self) -> np.ndarray:
		return np.zeros((self._lbm.y, self._lbm.x))

	def setup(self) -> None:
		detector = WallDetector()
		detector.detect(self.zero_gradient)

		for x, y_start, y_end in detector.left_walls:
			self._boundaries.append(_LeftGradient(x, y_start, y_end + 1))
		for x, y_start, y_end in detector.right_walls:
			self._boundaries.append(_RightGradient(x, y_start, y_end + 1))
		for y, x_start, x_end in detector.top_walls:
			self._boundaries.append(_TopGradient(y, x_start, x_end + 1))
		for y, x_start, x_end in detector.bot_walls:
			self._boundaries.append(_BottomGradient(y, x_start, x_end + 1))

	def apply_boundaries(self, f: VectorT, rho: VectorT, u: VectorT, time: ScalarT) -> None:
		for boundary in self._boundaries:
			boundary.apply(f, rho, u, time)
