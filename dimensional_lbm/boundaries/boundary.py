from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, Self

import numpy as np
from PIL import Image

from dimensional_lbm.unit_system_if import ScalarT, VectorT

if TYPE_CHECKING:
	from collections.abc import Callable, Iterator

	from dimensional_lbm._typing import NDIndex
	from dimensional_lbm.lbm import LBM


class TimeCallbackList:
	"""Ordered list of ``(key, callable)`` pairs for time-dependent boundary callbacks.

	Each callable accepts a single argument — the current simulation time — and returns
	the field value for the boundary cells selected by *key*.

	Usage in a boundary class::

		self._velocity_callbacks = TimeCallbackList()
		self._velocity_callbacks.append(key, lambda time, u=u0, t=t_ramp: ...)
		for key, cb in self._velocity_callbacks:
			field[key] = cb(time)
	"""

	def __init__(self) -> None:
		self._items: list[tuple[Any, Any]] = []

	def append(self, key: NDIndex, callback: Callable[..., object]) -> None:
		self._items.append((key, callback))

	def __iter__(self) -> Iterator[tuple[Any, Any]]:
		"""Iterate over the registered ``(key, callable)`` pairs."""
		return iter(self._items)

	def __len__(self) -> int:
		"""Return the number of registered callbacks."""
		return len(self._items)

	def __bool__(self) -> bool:
		"""Return whether any callbacks are registered."""
		return bool(self._items)


class BoundaryCollection(list["Boundary[ScalarT, VectorT]"], Generic[ScalarT, VectorT]):
	"""List of boundaries that supports ``lbm.boundaries += boundary`` without explicit keys."""

	def __add__(self, boundary: Boundary[ScalarT, VectorT]) -> BoundaryCollection[ScalarT, VectorT]:  # type: ignore[override]
		"""Append ``boundary`` and return the collection."""
		self.append(boundary)
		return self

	def __iadd__(self, boundary: Boundary[ScalarT, VectorT]) -> Self:  # type: ignore[override]
		"""Append ``boundary`` in place and return the collection."""
		self.append(boundary)
		return self

	def __iter__(self) -> Iterator[Boundary[ScalarT, VectorT]]:
		"""Iterate over the contained boundaries."""
		return super().__iter__()


class Boundary(ABC, Generic[ScalarT, VectorT]):

	@abstractmethod
	def __init__(self, lbm: LBM[Any, ScalarT, VectorT]) -> None:
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
