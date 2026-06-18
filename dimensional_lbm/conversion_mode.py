from typing import TypeVar


class Dimensional:
	pass

class NonDimensional:
	pass

type ConversionMode = Dimensional | NonDimensional

ModeT = TypeVar("ModeT", Dimensional, NonDimensional, default=Dimensional)
