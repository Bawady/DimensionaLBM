from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
import pint
import unit_jit
from PIL import Image

from dimensional_lbm.conversion_mode import Dimensional
from dimensional_lbm.unit_system_if import ScalarT, VectorT

if TYPE_CHECKING:
	from collections.abc import Iterator

	from dimensional_lbm.lbm import LBM


class TimeCallbackList:
	"""Ordered list of ``(key, callable)`` pairs for time-dependent boundary callbacks.

	Each callable must accept a single argument — the current simulation time — and
	return a field value with a known physical dimension.  The expected dimensions are
	declared at construction via *return_unit* (a unit string such as ``"m/s"``).

	In Dimensional mode, every registered callable is automatically passed through
	``unit_jit.jit_closure``: its body is abstract-interpreted once at registration
	time, raising ``TypeError`` immediately if a dimensional inconsistency is found
	(e.g. dividing by a length instead of a time).  Inside the unit-jit fast zone
	the callable then runs on plain SI floats.  In all other modes the callable is
	stored and called unchanged.

	Usage in a boundary class::

		self._velocity_callbacks = TimeCallbackList(lbm, "m/s")
		self._velocity_callbacks.append(key, lambda step, u=u0, t=t_ramp: ...)
		for key, cb in self._velocity_callbacks:
			field[key] = cb(time)
	"""

	def __init__(self, lbm: LBM, return_unit: str) -> None:
		self._lbm = lbm
		self._return_unit = return_unit
		self._items: list[tuple[Any, Any]] = []

	def append(self, key: Any, callback: Any) -> None:
		self._items.append((key, callback))

	def __iter__(self) -> Iterator[tuple[Any, Any]]:
		return iter(self._items)

	def __len__(self) -> int:
		return len(self._items)

	def __bool__(self) -> bool:
		return bool(self._items)


class BoundaryCollection(list["Boundary"]):
	"""List of boundaries that supports ``lbm.boundaries += boundary`` without explicit keys."""

	def __add__(self, boundary: Boundary) -> BoundaryCollection:  # type: ignore [override]
		self.append(boundary)
		return self

	def __iadd__(self, boundary: Boundary) -> BoundaryCollection:  # type: ignore[override]
		self.append(boundary)
		return self

	def __iter__(self) -> Iterator[Boundary]:
		return super().__iter__()


class Boundary(ABC, Generic[ScalarT, VectorT]):

	@abstractmethod
	def __init__(self, lbm : LBM) -> None:
		pass

	@abstractmethod
	def setup(self) -> None:
		pass

	@abstractmethod
	def apply_boundaries(self, f: VectorT, rho: VectorT, u: VectorT, time: ScalarT) -> None:
		pass

	@abstractmethod
	def get_geometry(self) -> np.ndarray:
		pass


def load_geometry(geometry_img: str) -> np.ndarray:
	"""Load B/W image and convert to solid array (black=1, white=0)."""
	img = Image.open(geometry_img).convert("L")
	img_array = np.array(img)
	return np.where(img_array == 0, 1, 0)
