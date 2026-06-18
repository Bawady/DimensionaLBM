from typing import Protocol, TypeVar

from dimensional_lbm.unit_system_if import (
	PintQuantityScalar,
	PintQuantityVector,
	ScalarMagnitude,
	ScalarT,
	VectorMagnitude,
)

_ScalarT_contra = TypeVar("_ScalarT_contra", PintQuantityScalar, ScalarMagnitude, contravariant=True)
_VectorT_co = TypeVar("_VectorT_co", PintQuantityVector, VectorMagnitude, covariant=True)


class VectorCallback(Protocol[_ScalarT_contra, _VectorT_co]):
	"""Callable producing a time-dependent vector field value."""

	def __call__(self, time: _ScalarT_contra) -> _VectorT_co: ...


class ScalarCallback(Protocol[ScalarT]):
	"""Callable producing a time-dependent scalar field value."""

	def __call__(self, time: ScalarT) -> ScalarT: ...
