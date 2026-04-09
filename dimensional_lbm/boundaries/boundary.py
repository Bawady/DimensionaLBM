from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

import numpy as np
from PIL import Image

from dimensional_lbm.unit_system_if import ScalarT, VectorT

if TYPE_CHECKING:
	from dimensional_lbm.lbm import LBM


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
