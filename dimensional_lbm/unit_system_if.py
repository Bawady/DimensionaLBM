"""Provide the abstract interface of a unit system."""

from typing import TypeVar, cast, overload

import numpy as np
import pint
from pint.facets.numpy.quantity import NumpyQuantity
from pint.facets.plain import PlainQuantity

from dimensional_lbm.conversion_mode import ConversionMode, Dimensional, NonDimensional

type PintQuantityScalar = PlainQuantity[float] | PlainQuantity[int]
type PintQuantityVector = NumpyQuantity[np.ndarray]
type PintQuantity = PintQuantityScalar | PintQuantityVector

type ScalarMagnitude = float | int
type VectorMagnitude = np.ndarray

type Magnitude = ScalarMagnitude | VectorMagnitude

type QuantityScalar = PintQuantityScalar | ScalarMagnitude
type QuantityVector = PintQuantityVector | VectorMagnitude
type Quantity = Magnitude | PintQuantity

type ScalarQuantityDefinition = tuple[ScalarMagnitude, str]

ScalarT = TypeVar("ScalarT", PintQuantityScalar, ScalarMagnitude, default=PintQuantityScalar)
VectorT = TypeVar("VectorT", PintQuantityVector, VectorMagnitude, default=PintQuantityVector)


def as_magnitude_array(x: PintQuantityVector | VectorMagnitude) -> np.ndarray:
	"""Return the underlying ndarray of a (possibly unit-carrying) vector field.

	The result is a view into the original buffer, so in-place updates propagate back.
	"""
	if isinstance(x, np.ndarray):
		return x
	return cast("PlainQuantity[np.ndarray]", x).magnitude


class UnitSystem[Mode: (Dimensional, NonDimensional)]:
	"""A generic, fully typed, unit system for different conversion modes."""

	__characteristic_quantities: list[PintQuantityScalar]
	__ureg: pint.UnitRegistry[float]
	__mode: ConversionMode

	@property
	def mode(self) -> ConversionMode:
		"""Get the conversion mode of this unit system."""
		return self.__mode

	def __init__(self, _mode: ConversionMode | None = None) -> None:
		"""Initialize an empty unit system."""
		self.__characteristic_quantities = []
		self.__ureg = cast("pint.UnitRegistry[float]", pint.get_application_registry())
		self.__ureg.setup_matplotlib(enable=True)
		self.__mode = _mode or Dimensional()

	def define_unit(self, unit_str: str) -> None:
		self.__ureg.define(unit_str)

	@overload
	def quantity(self: UnitSystem[Dimensional], value: float, unit: str) -> PlainQuantity[float]: ...

	@overload
	def quantity(self: UnitSystem[Dimensional], value: np.ndarray, unit: str) -> PintQuantityVector: ...

	@overload
	def quantity(self: UnitSystem[NonDimensional], value: float, unit: str) -> float: ...

	@overload
	def quantity(self: UnitSystem[NonDimensional], value: np.ndarray, unit: str) -> np.ndarray: ...

	@overload
	def quantity(self, value: ScalarMagnitude, unit: str) -> ScalarT: ...

	@overload
	def quantity(self, value: VectorMagnitude, unit: str) -> VectorT: ...

	def quantity(self, value: float | np.ndarray, unit: str) -> Quantity:
		"""Return the specified unit carrying quantity converted to the the unit system's conversion mode."""
		# Pint's `Quantity(...)` constructor overloads do not propagate the magnitude type
		# (they resolve to PlainQuantity[Any]). Given `value: float | np.ndarray`, the result is
		# a float scalar or an array quantity, so we restore that known type here.
		q = cast("PlainQuantity[float] | PintQuantityVector", self.__ureg.Quantity(value, unit))

		if isinstance(self.__mode, Dimensional):
			return q

		return self._non_dimensionalize(q)

	def __set_characteristic_quantities(self, quantities: list[PintQuantityScalar]) -> None:
		# Must only be called with valid quantities (as guaranteed within with_characteristic_quantities.)
		self.__characteristic_quantities = quantities

	def with_characteristic_quantities(self, definitions: list[ScalarQuantityDefinition]) -> UnitSystem[NonDimensional]:
		"""Create a new unit system out of this one using the given characteristic quantities to non dimensional future quantities.

		There must be at least 3, independent, scalar quantities. Otherwise a ValueError will be raised.
		"""
		least_nr_of_charas: int = 3

		quantities: list[PintQuantityScalar] = [self.__ureg.Quantity(q[0], q[1]) for q in definitions]

		# We require at least 3 quantities
		if len(quantities) < least_nr_of_charas:
			msg = f"At least three characteristic quantities must be specified, but only {len(quantities)} where: {quantities}"
			raise ValueError(msg)

		qs: dict[str, PintQuantityScalar] = {}
		for i in range(len(quantities)):
			qs[chr(ord("a") + i)] = quantities[i]
		coeffs: list[dict[str, float]] = self.__pi_theorem_typed(qs)

		if len(coeffs) > 0:
			msg = "The specified characteristic quantities are not independent!"
			raise ValueError(msg)

		non_dim_us: UnitSystem[NonDimensional] = UnitSystem(_mode=NonDimensional())
		non_dim_us.__set_characteristic_quantities(quantities)
		non_dim_us.__ureg = self.__ureg

		return non_dim_us

	def __pi_theorem_typed(self, quantities: dict[str, PintQuantityScalar]) -> list[dict[str, float]]:
		"""Typed wrapper around Pint's Pi theorem to facilitate static type checking."""
		# Pint ships `pi_theorem` without type annotations, so the member access is untyped.
		return cast("list[dict[str, float]]", self.__ureg.pi_theorem(quantities))  # pyright: ignore[reportUnknownMemberType]

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
		return q.to_base_units().magnitude

	@overload
	def _non_dimensionalize(self, q: float) -> float: ...

	@overload
	def _non_dimensionalize(self, q: np.ndarray) -> np.ndarray: ...

	@overload
	def _non_dimensionalize(self, q: PlainQuantity[float]) -> float: ...

	@overload
	def _non_dimensionalize(self, q: PintQuantityVector) -> np.ndarray: ...

	def _non_dimensionalize(self, q: Quantity) -> float | np.ndarray:
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
				# Pint's scalar `**`/`*` operator stubs drop the magnitude type; the runtime
				# result is a Pint scalar, so we restore the known type.
				scale = cast("PintQuantityScalar", scale * qs[key] ** p)
		d = 1 / coeffs[0]["q"]
		# Pint operators on a NumpyQuantity are stubbed to return PlainQuantity rather than
		# NumpyQuantity, so the result is not seen as PintQuantity even though it is one at
		# runtime; the casts restore the declared return type.
		scaled = cast("PintQuantity", x * scale**d)
		return cast("PintQuantity", scaled.to_base_units())

	@overload
	def to_unit(self, x: float, unit: str) -> float: ...

	@overload
	def to_unit(self, x: np.ndarray, unit: str) -> np.ndarray: ...

	@overload
	def to_unit(self, x: PlainQuantity[float], unit: str) ->PlainQuantity[float]: ...

	@overload
	def to_unit(self, x: PlainQuantity[int], unit: str) ->PlainQuantity[int]: ...

	@overload
	def to_unit(self, x: PintQuantityVector, unit: str) -> PintQuantityVector: ...

	def to_unit(self, x: Quantity, unit: str) -> Quantity:
		"""Return x converted to the given unit (must have same dimensionality)."""
		if isinstance(x, (int, float, np.ndarray)):
			return x
		# Pint's `to` has unannotated *contexts/**kwargs, so the member type is partially unknown.
		return x.to(unit)  # pyright: ignore[reportUnknownMemberType]

	@overload
	def dim(self, x: float, unit: str) -> PlainQuantity[float]: ...

	@overload
	def dim(self, x: np.ndarray, unit: str) -> PintQuantityVector: ...

	@overload
	def dim(self, x: PlainQuantity[float], unit: str) ->PlainQuantity[float]: ...

	@overload
	def dim(self, x: PlainQuantity[int], unit: str) ->PlainQuantity[int]: ...

	@overload
	def dim(self, x: PintQuantityVector, unit: str) -> PintQuantityVector: ...

	def dim(self, x: Quantity, unit: str) -> Quantity:
		"""Dimensionalize x to the given unit (already dimensional quantities are checked to have the given unit's dimensionality)."""
		if isinstance(x, (int, float, np.ndarray)):
			target = cast("PintQuantityScalar", self.__ureg.Quantity(1, unit))
			x_dimensionless = cast("PintQuantity", self.__ureg.Quantity(x, "1"))
			return self.__apply_pi_theorem(x_dimensionless, 1 / target)

		if x.dimensionality != self.__ureg.Quantity(1, unit).dimensionality:
			extra_msg = "Attempted to dimensionalize already dimensional quantity to different dimensionality."
			raise pint.DimensionalityError(x.units, unit, extra_msg=extra_msg)
		return x

