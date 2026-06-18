"""Shared type aliases for the dimensional LBM framework."""

from types import EllipsisType

import numpy as np

# A NumPy basic or advanced index: a single index element or a tuple of them. NumPy exposes
# no public index type, so this approximates the index forms used throughout this codebase.
type _NDIndexElem = int | slice | EllipsisType | None | np.ndarray
type NDIndex = _NDIndexElem | tuple[_NDIndexElem, ...]
