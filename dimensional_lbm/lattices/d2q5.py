from typing import ClassVar

import numpy as np

from dimensional_lbm.unit_system_if import ScalarT, VectorT, as_magnitude_array

from .ddqq_lattice import DdQqLattice


class D2Q5(DdQqLattice[ScalarT, VectorT]):
	D: ClassVar[int] = 2
	Q: ClassVar[int] = 5

	# lattice weights
	__w_rest = 1 / 3
	__w_straight = (1 - __w_rest) / 4

	weights = np.array([__w_rest, __w_straight, __w_straight, __w_straight, __w_straight], dtype=float)

	dir_x: ClassVar[np.ndarray] = np.array([0, 1, -1, 0, 0], dtype=int)
	dir_y: ClassVar[np.ndarray] = np.array([0, 0, 0, -1, 1], dtype=int)

	def __init__(self, dx: ScalarT, dt: ScalarT) -> None:
		super().__init__(dx, dt)

		self.cs = self.dx / (self.dt * np.sqrt(3))
		# Pint's `Quantity * int` overload widens the type, `/` does not.
		self.cs_n2 = self.dt**2 / (self.dx**2 / 3)
		self.cs_n4 = self.dt**4 / (self.dx**4 / 9)

	#######################################
	#####          Streaming          #####
	#######################################

	def stream_periodic(self, f_new: VectorT, f_old: VectorT) -> None:
		mag_new = as_magnitude_array(f_new)
		mag_old = as_magnitude_array(f_old)

		for i in range(self.Q):
			mag_new[i] = np.roll(mag_old[i], (self.dir_y[i], self.dir_x[i]), axis=(0, 1))

	def stream(self, f_new: VectorT, f_old: VectorT) -> None:
		mag_new = as_magnitude_array(f_new)
		mag_old = as_magnitude_array(f_old)

		height, width = mag_new.shape[1], mag_new.shape[2]

		mag_new[0] = mag_old[0]

		mag_new[1, :, 1:width] = mag_old[1, :, 0 : width - 1].copy()
		mag_new[2, :, 0 : width - 1] = mag_old[2, :, 1:width].copy()
		mag_new[3, 0 : height - 1, :] = mag_old[3, 1:height, :].copy()
		mag_new[4, 1:height, :] = mag_old[4, 0 : height - 1, :].copy()

	def equilibrium(self, density: VectorT, velocity: VectorT, eq: VectorT) -> None:
		vel_x = self.dir_x * self.q
		vel_y = self.dir_y * self.q
		# Pint's array-operator stubs yield a spurious `datetime` union -> Pyright can't infer expressions with arithmetic on indexed quantity arrays
		u_sq = self.cs_n2 / 2.0 * (velocity[:, :, 0]**2 + velocity[:, :, 1]**2)  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
		for i in range(self.Q):
			lin_term = self.cs_n2 * (vel_x[i] * velocity[:, :, 0] + vel_y[i] * velocity[:, :, 1])
			eq[i] = self.weights[i] * density * (1 + lin_term + lin_term**2 / 2.0 - u_sq)
