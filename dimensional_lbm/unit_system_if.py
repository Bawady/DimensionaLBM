"""Provide the abstract interface of a unit system."""

from typing import TypeVar, cast, overload

import numpy as np
import pint
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity

from dimensional_lbm.conversion_mode import ConversionMode, Dimensional, MagnitudeOnly, NonDimensional

type PintQuantityScalar = PlainQuantity[float] | PlainQuantity[int]
type PintQuantityVector = NumpyQuantity
type PintQuantity = PintQuantityScalar | PintQuantityVector

type ScalarMagnitude = float | int
type VectorMagnitude = np.ndarray

type Magnitude = ScalarMagnitude | VectorMagnitude

type QuantityScalar = PintQuantityScalar | ScalarMagnitude
type QuantityVector = PintQuantityVector | VectorMagnitude
type Quantity = Magnitude | PintQuantity

type ScalarQuantityDefinition = tuple[ScalarMagnitude, str]

ScalarT = TypeVar("ScalarT", PintQuantityScalar, ScalarMagnitude)
VectorT = TypeVar("VectorT", PintQuantityVector, VectorMagnitude)


class UnitSystem[Mode: (Dimensional, NonDimensional, MagnitudeOnly)]:
	"""A generic, fully typed, unit system for different conversion modes."""

	__characteristic_quantities: list[PintQuantityScalar]
	__ureg: pint.UnitRegistry
	__mode: ConversionMode

	@property
	def mode(self) -> ConversionMode:
		"""Get the conversion mode of this unit system."""
		return self.__mode

	def __init__(self, _mode: ConversionMode | None = None) -> None:
		"""Initialize an empty unit system."""
		self.__characteristic_quantities = []
		self.__ureg = cast("pint.UnitRegistry", pint.get_application_registry())
		self.__mode = _mode if _mode else Dimensional()

	@overload
	def quantity(self: "UnitSystem[Dimensional]", value: float, unit: str) -> PlainQuantity[float]: ...

	@overload
	def quantity(self: "UnitSystem[Dimensional]", value: np.ndarray, unit: str) -> NumpyQuantity: ...

	@overload
	def quantity(self: "UnitSystem[NonDimensional]", value: float, unit: str) -> float: ...

	@overload
	def quantity(self: "UnitSystem[NonDimensional]", value: np.ndarray, unit: str) -> np.ndarray: ...

	@overload
	def quantity(self: "UnitSystem[MagnitudeOnly]", value: float, unit: str) -> float: ...

	@overload
	def quantity(self: "UnitSystem[MagnitudeOnly]", value: np.ndarray, unit: str) -> np.ndarray: ...

	@overload
	def quantity(self, value: ScalarMagnitude, unit: str) -> ScalarT: ...

	@overload
	def quantity(self, value: VectorMagnitude, unit: str) -> VectorT: ...

	def quantity(self, value: float | np.ndarray, unit: str) -> Quantity:
		"""Return the specified unit afflicted quantity converted to the the unit system's conversion mode."""
		if isinstance(self.__mode, MagnitudeOnly):
			return value

		q = self.__ureg.Quantity(value, unit)
		if isinstance(self.__mode, Dimensional):
			return q
		if isinstance(self.__mode, NonDimensional):
			return self.__non_dimensionalize(q)

		msg: str = f"Unknown conversion mode {self.__mode}"
		raise ValueError(msg)

	def __set_characteristic_quantities(self, quantities: list[PintQuantityScalar]) -> None:
		# Must only be called with valid quantities (as guaranteed within with_characteristic_quantities.)
		self.__characteristic_quantities = quantities

	def with_characteristic_quantities(self, definitions: list[ScalarQuantityDefinition]) -> "UnitSystem[NonDimensional]":
		"""Create a new unit system out of this one using the given characteristic quantities to non dimensional future quantities.

		There must be at least 3, independent, scalar quantities. Otherwise a ValueError will be raised.
		"""
		least_nr_of_charas: int = 3

		quantities: list[PintQuantityScalar] = [self.__ureg.Quantity(q[0], q[1]) for q in definitions]

		# We require at least 3 quantities
		if len(quantities) < least_nr_of_charas:
			msg = f"At least three characteristic quantities must be specified, but only {len(quantities)} where: {quantities}"
			raise ValueError(msg)

		# They are all required to be scalars
		non_scalar_idxs: list[int] = [i for i, q in enumerate(quantities) if not isinstance(q.magnitude, (int, float))]
		if len(non_scalar_idxs) > 0:
			msg = f"Characteristic quantities must be scalar. However, the quantities at the following indices are not scalar: {non_scalar_idxs}"
			raise ValueError(msg)

		qs: dict[str, PintQuantityScalar] = {}
		for i in range(len(quantities)):
			qs[chr(ord("a") + i)] = quantities[i]
		coeffs: list[dict[str, float]] = self.__pi_theorem_typed(qs)

		if len(coeffs) > 0:
			msg = "The specified characteristic quantities are not independent!"
			raise ValueError(msg)

		non_dim_us = UnitSystem(_mode=NonDimensional())
		non_dim_us.__set_characteristic_quantities(quantities)
		non_dim_us.__ureg = self.__ureg

		return non_dim_us

	def with_magnitude_only(self) -> "UnitSystem[MagnitudeOnly]":
		"""Create a new unit system out of this one configured to only provide quantity magnitudes from now on."""
		return UnitSystem(_mode=MagnitudeOnly())

	def __pi_theorem_typed(self, quantities: dict[str, PintQuantityScalar]) -> list[dict[str, float]]:
		"""Typed wrapper around Pint's Pi theorem to facilitate static type checking."""
		return cast("list[dict[str, float]]", self.__ureg.pi_theorem(quantities))

	@overload
	def magnitude(self, q: float) -> float: ...

	@overload
	def magnitude(self, q: np.ndarray) -> np.ndarray: ...

	@overload
	def magnitude(self, q: PlainQuantity[float]) -> float: ...

	@overload
	def magnitude(self, q: PlainQuantity[int]) -> int: ...

	@overload
	def magnitude(self, q: ScalarT) -> ScalarMagnitude: ...

	@overload
	def magnitude(self, q: VectorT) -> VectorMagnitude: ...

	def magnitude(self, q: Quantity) -> Magnitude:
		"""Get the magnitude of a Quantity."""
		if isinstance(q, (int, float, np.ndarray)):
			return q
		return q.magnitude

	@overload
	def __non_dimensionalize(self, q: float) -> float: ...

	@overload
	def __non_dimensionalize(self, q: np.ndarray) -> np.ndarray: ...

	@overload
	def __non_dimensionalize(self, q: PlainQuantity[float]) -> float: ...

	@overload
	def __non_dimensionalize(self, q: NumpyQuantity) -> np.ndarray: ...

	def __non_dimensionalize(self, q: Quantity) -> float | np.ndarray:
		if isinstance(q, (int, float, np.ndarray)):
			return q

		q_dimensionless: PintQuantity = self.__apply_pi_theorem(q, q)
		assert len(q_dimensionless.dimensionality) == 0  # noqa: S101
		return q_dimensionless.magnitude

	def __apply_pi_theorem(self, x: PintQuantity, q: PintQuantity) -> PintQuantity:
		if len(self.__characteristic_quantities) == 0:
			msg: str = "Applying the pi theorem requires characteristic quantities."
			raise ValueError(msg)

		qs: dict[str, PintQuantityScalar] = {}
		for i, characteristic in enumerate(self.__characteristic_quantities):
			# Characteristice are enforced to be PQ[float] when being defined -> helping the type checker here completely alright.
			qs[chr(ord("a") + i)] = characteristic
		qs["q"] = cast("PlainQuantity[float]", self.__ureg.Quantity(1, q.units))
		# The pi theorem does not come with complete type hints. Refer to doc for more details.
		coeffs: list[dict[str, float]] = self.__pi_theorem_typed(qs)

		scale = 1
		for key in coeffs[0]:
			if key != "q" and key in qs:
				p: int = int(coeffs[0][key])
				scale *= qs[key] ** p
		d = 1 / coeffs[0]["q"]
		return (x * scale**d).to_base_units()

	@overload
	def to_unit(self, x: float, unit: str) -> float: ...

	@overload
	def to_unit(self, x: np.ndarray, unit: str) -> np.ndarray: ...

	@overload
	def to_unit(self, x: PlainQuantity[float], unit: str) ->PlainQuantity[float]: ...

	@overload
	def to_unit(self, x: PlainQuantity[int], unit: str) ->PlainQuantity[int]: ...

	@overload
	def to_unit(self, x: NumpyQuantity, unit: str) -> NumpyQuantity: ...

	def to_unit(self, x: Quantity, unit: str) -> Quantity:
		"""Return x converted to the given unit (must have same dimensionality)."""
		# TODO: Either handle MangitudeOnly here (by passiong an optional unit for x), or remove this mode altogether
		if isinstance(x, (int, float, np.ndarray)):
			return x
		return x.to(unit)
