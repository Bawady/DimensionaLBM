from typing import TypeVar

class Dimensional:
	pass

class NonDimensional:
	pass

class MagnitudeOnly:
	pass

type ConversionMode = Dimensional | NonDimensional | MagnitudeOnly

ModeT = TypeVar("ModeT", Dimensional, NonDimensional, MagnitudeOnly)
