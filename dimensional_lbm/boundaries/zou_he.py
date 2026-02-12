from abc import ABC, abstractmethod
from enum import Enum, auto

import numpy as np
from numpy import ndarray

from dimensional_lbm.boundaries.boundary import Boundary
from dimensional_lbm.boundaries.wall_detector import WallDetector
from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
from dimensional_lbm.unit_system_if import ScalarT, VectorT


class _CornerOrientation(Enum):
	BOT_LEFT = auto()
	BOT_RIGHT = auto()
	TOP_LEFT = auto()
	TOP_RIGHT = auto()


class _ZouHeBoundary(ABC):
	"""Abstract base class for individual Zou-He boundary segments."""

	@abstractmethod
	def apply(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		pass

	@abstractmethod
	def defined_by(self, coordinates: tuple[int]) -> bool:
		"""Returns true if a boundary's position is defined by the given coordinates."""
		pass


class _BottomWall(_ZouHeBoundary):
	def __init__(self, q: ScalarT, y: int, x0: int, x1: int, velocity_profile: VectorT | None = None, density_profile: VectorT | None = None) -> None:
		self._q = q
		self._y = y
		self._x0 = x0
		self._x1 = x1
		self.velocity_profile = velocity_profile
		self.density_profile = density_profile

	def apply(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		y, x0, x1, q = self._y, self._x0, self._x1, self._q
		velocity_profile = self.velocity_profile
		density_profile = self.density_profile

		if density_profile is None:
			scale = q / (q + u[1, y, x0:x1])
			rho[y, x0:x1] = scale * (f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] + 2 * (f[4, y, x0:x1] + f[5, y, x0:x1] + f[8, y, x0:x1]))
			if velocity_profile is not None:
				u[:, y, x0:x1] = velocity_profile
			else:
				u[:, y, x0:x1] *= 0
		elif velocity_profile is None:
			rho[y, x0:x1] = density_profile
			u[0, y, x0:x1] *= 0
			u[1, y, x0:x1] = q * (
				(f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] + 2 * (f[4, y, x0:x1] + f[5, y, x0:x1] + f[8, y, x0:x1])) / density_profile - 1
			)
		else:
			msg: str = "Neither velocity nor density profile given"
			raise ValueError(msg)

		wall_dens = rho[y, x0:x1]
		vel_x = u[0, y, x0:x1]
		vel_y = u[1, y, x0:x1]
		# up
		f[3, y, x0:x1] = f[4, y, x0:x1] - 2 * wall_dens * vel_y / (3 * q)
		# up-left
		f[6, y, x0:x1] = f[5, y, x0:x1] + 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1]) + wall_dens / q * (-0.5 * vel_x - 1 / 6.0 * vel_y)
		# up-right
		f[7, y, x0:x1] = f[8, y, x0:x1] - 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1]) + wall_dens / q * (0.5 * vel_x - 1 / 6.0 * vel_y)

	def defined_by(self, coordinates: tuple[int]) -> bool:
		return len(coordinates) == 3 and self._y == coordinates[0] and self._x0 == coordinates[1] and self._x1 == coordinates[2]


class _TopWall(_ZouHeBoundary):
	def __init__(self, q: ScalarT, y: int, x0: int, x1: int, velocity_profile: VectorT | None = None, density_profile: VectorT | None = None) -> None:
		self._q = q
		self._y = y
		self._x0 = x0
		self._x1 = x1
		self.velocity_profile = velocity_profile
		self.density_profile = density_profile

	def apply(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		y, x0, x1, q = self._y, self._x0, self._x1, self._q
		velocity_profile = self.velocity_profile
		density_profile = self.density_profile

		if density_profile is None:
			scale = q / (q - u[1, y, x0:x1])
			rho[y, x0:x1] = scale * (f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] + 2 * (f[3, y, x0:x1] + f[6, y, x0:x1] + f[7, y, x0:x1]))
			if velocity_profile is not None:
				u[:, y, x0:x1] = velocity_profile
			else:
				u[:, y, x0:x1] *= 0
		elif velocity_profile is None:
			rho[y, x0:x1] = density_profile
			u[0, y, x0:x1] = 0
			u[1, y, x0:x1] = q * (
				1 - (f[0, y, x0:x1] + f[1, y, x0:x1] + f[2, y, x0:x1] + 2 * (f[3, y, x0:x1] + f[6, y, x0:x1] + f[7, y, x0:x1])) / density_profile
			)
		else:
			msg: str = "Neither velocity nor density profile given"
			raise ValueError(msg)

		wall_dens = rho[y, x0:x1]
		vel_x = u[0, y, x0:x1]
		vel_y = u[1, y, x0:x1]
		# down
		f[4, y, x0:x1] = f[3, y, x0:x1] + 2 * wall_dens * vel_y / (3 * q)
		# down-right
		f[5, y, x0:x1] = f[6, y, x0:x1] - 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1]) + wall_dens / q * (0.5 * vel_x + 1 / 6.0 * vel_y)
		# down-left
		f[8, y, x0:x1] = f[7, y, x0:x1] + 0.5 * (f[1, y, x0:x1] - f[2, y, x0:x1]) + wall_dens / q * (-0.5 * vel_x + 1 / 6.0 * vel_y)

	def defined_by(self, coordinates: tuple[int]) -> bool:
		return len(coordinates) == 3 and self._y == coordinates[0] and self._x0 == coordinates[1] and self._x1 == coordinates[2]


class _LeftWall(_ZouHeBoundary):
	def __init__(self, q: ScalarT, x: int, y0: int, y1: int, velocity_profile: VectorT | None = None, density_profile: VectorT | None = None) -> None:
		self._q = q
		self._x = x
		self._y0 = y0
		self._y1 = y1
		self.velocity_profile = velocity_profile
		self.density_profile = density_profile

	def apply(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		x, y0, y1, q = self._x, self._y0, self._y1, self._q
		velocity_profile = self.velocity_profile
		density_profile = self.density_profile

		if density_profile is None:
			scale = q / (q - u[0, y0:y1, x])
			rho[y0:y1, x] = scale * (f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] + 2 * (f[2, y0:y1, x] + f[6, y0:y1, x] + f[8, y0:y1, x]))
			if velocity_profile is not None:
				u[:, y0:y1, x] = velocity_profile
			else:
				u[:, y0:y1, x] *= 0
		elif velocity_profile is None:
			rho[y0:y1, x] = density_profile
			u[1, y0:y1, x] = 0
			u[0, y0:y1, x] = q * (
				1 - (f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] + 2 * (f[2, y0:y1, x] + f[6, y0:y1, x] + f[8, y0:y1, x])) / density_profile
			)
		else:
			raise ValueError("Neither velocity nor density profile given")

		wall_dens = rho[y0:y1, x]
		vel_x = u[0, y0:y1, x]
		vel_y = u[1, y0:y1, x]
		# right
		f[1, y0:y1, x] = f[2, y0:y1, x] + 2 * wall_dens * vel_x / (3 * q)
		# down-right
		f[5, y0:y1, x] = f[6, y0:y1, x] - 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x]) + wall_dens / q * (1 / 6.0 * vel_x + 0.5 * vel_y)
		# up-right
		f[7, y0:y1, x] = f[8, y0:y1, x] + 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x]) + wall_dens / q * (1 / 6.0 * vel_x - 0.5 * vel_y)

	def defined_by(self, coordinates: tuple[int]) -> bool:
		return len(coordinates) == 3 and self._x == coordinates[0] and self._y0 == coordinates[1] and self._y1 == coordinates[2]


class _RightWall(_ZouHeBoundary):
	def __init__(self, q: ScalarT, x: int, y0: int, y1: int, velocity_profile: VectorT | None = None, density_profile: VectorT | None = None) -> None:
		self._q = q
		self._x = x
		self._y0 = y0
		self._y1 = y1
		self.velocity_profile = velocity_profile
		self.density_profile = density_profile

	def apply(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		x, y0, y1, q = self._x, self._y0, self._y1, self._q
		velocity_profile = self.velocity_profile
		density_profile = self.density_profile

		if density_profile is None:
			scale = q / (q + u[0, y0:y1, x])
			rho[y0:y1, x] = scale * (f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] + 2 * (f[1, y0:y1, x] + f[5, y0:y1, x] + f[7, y0:y1, x]))
			if velocity_profile is not None:
				u[:, y0:y1, x] = velocity_profile
			else:
				u[:, y0:y1, x] *= 0
		elif velocity_profile is None:
			rho[y0:y1, x] = density_profile
			u[1, y0:y1, x] *= 0
			u[0, y0:y1, x] = q * (
				(f[0, y0:y1, x] + f[3, y0:y1, x] + f[4, y0:y1, x] + 2 * (f[1, y0:y1, x] + f[5, y0:y1, x] + f[7, y0:y1, x])) / density_profile - 1
			)
		else:
			raise ValueError("Neither velocity nor density profile given")

		wall_dens = rho[y0:y1, x]
		vel_x = u[0, y0:y1, x]
		vel_y = u[1, y0:y1, x]
		# down
		f[2, y0:y1, x] = f[1, y0:y1, x] - 2 * wall_dens * vel_x / (3 * q)
		# down-right
		f[6, y0:y1, x] = f[5, y0:y1, x] + 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x]) - wall_dens / q * (1 / 6.0 * vel_x + 0.5 * vel_y)
		# down-left
		f[8, y0:y1, x] = f[7, y0:y1, x] - 0.5 * (f[4, y0:y1, x] - f[3, y0:y1, x]) + wall_dens / q * (-1 / 6.0 * vel_x + 0.5 * vel_y)

	def defined_by(self, coordinates: tuple[int]) -> bool:
		return len(coordinates) == 3 and self._x == coordinates[0] and self._y0 == coordinates[1] and self._y1 == coordinates[2]


_CONVEX_MAP: dict[_CornerOrientation, tuple[int, int]] = {
	_CornerOrientation.TOP_RIGHT: (7, 8),
	_CornerOrientation.BOT_RIGHT: (5, 6),
	_CornerOrientation.TOP_LEFT: (6, 5),
	_CornerOrientation.BOT_LEFT: (8, 7),
}


class _ConvexCorner(_ZouHeBoundary):
	def __init__(self, ys: ndarray, xs: ndarray, orientation: _CornerOrientation) -> None:
		self._ys = ys
		self._xs = xs
		dst, src = _CONVEX_MAP[orientation]
		self._dst = dst
		self._src = src

	def apply(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		f[self._dst, self._ys, self._xs] = f[self._src, self._ys, self._xs]

	def defined_by(self, coordinates: tuple[int]) -> bool:
		return False


class _ConcaveCorner(_ZouHeBoundary):
	def __init__(self, q: ScalarT, ys: ndarray, xs: ndarray, orientation: _CornerOrientation) -> None:
		self._q = q
		self._ys = ys
		self._xs = xs
		self._apply_single = {
			_CornerOrientation.BOT_LEFT: self._apply_bot_left,
			_CornerOrientation.BOT_RIGHT: self._apply_bot_right,
			_CornerOrientation.TOP_LEFT: self._apply_top_left,
			_CornerOrientation.TOP_RIGHT: self._apply_top_right,
		}[orientation]

	def defined_by(self, coordinates: tuple[int]) -> bool:
		return False

	def apply(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		for i in range(len(self._ys)):
			self._apply_single(f, rho, u, self._ys[i], self._xs[i])

	def _apply_bot_left(self, f: VectorT, rho: VectorT, u: VectorT, y: int, x: int) -> None:
		return
		q = self._q
		u[0, y, x] = u[0, y, x + 1]
		u[1, y, x] = u[1, y - 1, x]
		rho[y, x] = rho[y, x + 1]

		f[1, y, x] = f[2, y, x] + 2.0 * rho[y, x] * u[0, y, x] / (3 * q)
		f[3, y, x] = f[4, y, x] - 2.0 * rho[y, x] * u[1, y, x] / (3 * q)
		f[7, y, x] = f[8, y, x] + rho[y, x] * (u[0, y, x] - u[1, y, x]) / (6 * q)
		f[5, y, x] *= 0
		f[6, y, x] *= 0
		f[0, y, x] = rho[y, x] - np.sum(f[1:9, y, x])

	def _apply_bot_right(self, f: VectorT, rho: VectorT, u: VectorT, y: int, x: int) -> None:
		return
		q = self._q
		u[0, y, x] = u[0, y, x - 1]
		u[1, y, x] = u[1, y - 1, x]
		rho[y, x] = rho[y, x - 1]

		f[2, y, x] = f[1, y, x] - 2.0 * rho[y, x] * u[0, y, x] / (3 * q)
		f[3, y, x] = f[4, y, x] - 2.0 * rho[y, x] * u[1, y, x] / (3 * q)
		f[6, y, x] = f[5, y, x] - rho[y, x] * (u[0, y, x] + u[1, y, x]) / (6 * q)
		f[7, y, x] *= 0
		f[8, y, x] *= 0
		f[0, y, x] = rho[y, x] - np.sum(f[1:9, y, x])

	def _apply_top_left(self, f: VectorT, rho: VectorT, u: VectorT, y: int, x: int) -> None:
		return
		q = self._q
		u[0, y, x] = u[0, y, x + 1]
		u[1, y, x] = u[1, y + 1, y]
		rho[y, x] = rho[y, x + 1]

		f[1, y, x] = f[2, y, x] + 2.0 * rho[y, x] * u[0, y, x] / (3 * q)
		f[4, y, x] = f[3, y, x] + 2.0 * rho[y, x] * u[1, y, x] / (3 * q)
		f[5, y, x] = f[6, y, x] + rho[y, x] * (u[0, y, x] + u[1, y, x]) / (6 * q)
		f[7, y, x] *= 0
		f[8, y, x] *= 0
		f[0, y, x] = rho[y, x] - np.sum(f[1:9, y, x])

	def _apply_top_right(self, f: VectorT, rho: VectorT, u: VectorT, y: int, x: int) -> None:
		return
		q = self._q
		u[0, y, x] = u[0, y, x - 1]
		u[1, y, x] = u[1, y + 1, x]
		rho[y, x] = rho[y, x - 1]

		f[2, y, x] = f[1, y, x] - 2.0 * rho[y, x] * u[0, y, x] / (3 * q)
		f[4, y, x] = f[3, y, x] + 2.0 * rho[y, x] * u[1, y, x] / (3 * q)
		f[8, y, x] = f[7, y, x] + rho[y, x] * (-u[0, y, x] + u[1, y, x]) / (6 * q)
		f[5, y, x] *= 0
		f[6, y, x] *= 0
		f[0, y, x] = rho[y, x] - np.sum(f[1:9, y, x])


class ZouHe(Boundary):
	_boundaries: list[_ZouHeBoundary]

	def __init__(self, lattice: DdQqLattice) -> None:
		super().__init__(lattice)
		self._boundaries = []

	def setup(self, geometry: ndarray) -> None:
		detector = WallDetector()
		detector.detect(geometry)

		self._boundaries = []

		# Wall segments (detector end coordinates are inclusive, slicing needs exclusive)
		for y, x_start, x_end in detector.bot_walls:
			self._boundaries.append(_BottomWall(self._lattice.q, y, x_start, x_end + 1))
		for y, x_start, x_end in detector.top_walls:
			self._boundaries.append(_TopWall(self._lattice.q, y, x_start, x_end + 1))
		for x, y_start, y_end in detector.left_walls:
			self._boundaries.append(_LeftWall(self._lattice.q, x, y_start, y_end + 1))
		for x, y_start, y_end in detector.right_walls:
			self._boundaries.append(_RightWall(self._lattice.q, x, y_start, y_end + 1))

		# Convex corners
		for orientation, corners in [
			(_CornerOrientation.TOP_LEFT, detector.conv_tl),
			(_CornerOrientation.TOP_RIGHT, detector.conv_tr),
			(_CornerOrientation.BOT_LEFT, detector.conv_bl),
			(_CornerOrientation.BOT_RIGHT, detector.conv_br),
		]:
			if corners.size > 0:
				self._boundaries.append(_ConvexCorner(corners[0], corners[1], orientation))

		# Concave corners
		for orientation, corners in [
			(_CornerOrientation.TOP_LEFT, detector.conc_tl),
			(_CornerOrientation.TOP_RIGHT, detector.conc_tr),
			(_CornerOrientation.BOT_LEFT, detector.conc_bl),
			(_CornerOrientation.BOT_RIGHT, detector.conc_br),
		]:
			if corners.size > 0:
				self._boundaries.append(_ConcaveCorner(self._lattice.q, corners[0], corners[1], orientation))

	def add_horizontal_velocity_profile(self, y: int, x0: int, x1: int, profile: VectorT) -> None:
		for boundary in self._boundaries:
			if isinstance(boundary, (_BottomWall, _TopWall)) and boundary.defined_by((y, x0, x1)):
				boundary.velocity_profile = profile
				return

	def add_horizontal_density_profile(self, x: int, y0: int, y1: int, profile: VectorT) -> None:
		for boundary in self._boundaries:
			if isinstance(boundary, (_BottomWall, _TopWall)) and boundary.defined_by((y, x0, x1)):
				boundary.velocity_profile = profile
				return

	def add_vertical_velocity_profile(self, x: int, y0: int, y1: int, profile: VectorT) -> None:
		for boundary in self._boundaries:
			if isinstance(boundary, (_LeftWall, _RightWall)) and boundary.defined_by((x, y0, y1)):
				boundary.velocity_profile = profile
				return

	def add_vertical_density_profile(self, x: int, y0: int, y1: int, profile: VectorT) -> None:
		for boundary in self._boundaries:
			if isinstance(boundary, (_LeftWall, _RightWall)) and boundary.defined_by((x, y0, y1)):
				boundary.density_profile = profile
				return

	def apply_boundaries(self, f: VectorT, rho: VectorT, u: VectorT) -> None:
		for boundary in self._boundaries:
			boundary.apply(f, rho, u)
