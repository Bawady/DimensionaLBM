from typing import ClassVar, cast

import numpy as np
import pint

from dimensional_lbm.unit_system_if import ScalarT, VectorT

from .ddqq_lattice import DdQqLattice


class D2Q9(DdQqLattice):
	D: ClassVar[int] = 2
	Q: ClassVar[int] = 9

	# lattice weights
	__base_weight = 1 / 9
	__w_rest = 4 * __base_weight
	__w_straight = (1 - __w_rest) / 5
	__w_diag = (1 - __w_rest - 4 * __w_straight) / 4

	assert __w_rest + 4 * (__w_straight + __w_diag) == 1.0
	assert __w_diag > 0

	weights = np.array([__w_rest, __w_straight, __w_straight, __w_straight, __w_straight, __w_diag, __w_diag, __w_diag, __w_diag], dtype=float)

	dir_x: ClassVar[np.ndarray] = np.array([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=int)
	dir_y: ClassVar[np.ndarray] = np.array([0, 0, 0, -1, 1, 1, -1, -1, 1], dtype=int)

	def __init__(self, dx: ScalarT, dt: ScalarT):
		self.dx = dx
		self.dt = dt

		self.q = dx / dt

		self.cs = self.dx / (self.dt * np.sqrt(3))
		self.cs_n2 = self.dt**2 * 3 / self.dx**2
		self.cs_n4 = self.cs_n2**2

	#######################################
	#####          Streaming          #####
	#######################################

	def stream_periodic(self, f_new: VectorT, f_old: VectorT) -> None:
		mag_new = cast("np.ndarray", f_new.magnitude if isinstance(f_new, pint.Quantity) else f_new)
		mag_old = cast("np.ndarray", f_old.magnitude if isinstance(f_old, pint.Quantity) else f_old)

		for i in range(self.Q):
			mag_new[i] = np.roll(mag_old[i], (self.dir_y[i], self.dir_x[i]), axis=(0, 1))

	def stream(self, f_new: VectorT, f_old: VectorT) -> None:
		mag_new = cast("np.ndarray", f_new.magnitude if isinstance(f_new, pint.Quantity) else f_new)
		mag_old = cast("np.ndarray", f_old.magnitude if isinstance(f_old, pint.Quantity) else f_old)

		height, width = mag_new.shape[1], mag_new.shape[2]

		mag_new[0] = mag_old[0]

		mag_new[1, :, 1:width] = mag_old[1, :, 0 : width - 1].copy()
		mag_new[2, :, 0 : width - 1] = mag_old[2, :, 1:width].copy()
		mag_new[3, 0 : height - 1, :] = mag_old[3, 1:height, :].copy()
		mag_new[4, 1:height, :] = mag_old[4, 0 : height - 1, :].copy()

		mag_new[5, 1:height, 1:width] = mag_old[5, 0 : height - 1, 0 : width - 1].copy()
		mag_new[6, 0 : height - 1, 0 : width - 1] = mag_old[6, 1:height, 1:width].copy()
		mag_new[7, 0 : height - 1, 1:width] = mag_old[7, 1:height, 0 : width - 1].copy()
		mag_new[8, 1:height, 0 : width - 1] = mag_old[8, 0 : height - 1, 1:width].copy()
