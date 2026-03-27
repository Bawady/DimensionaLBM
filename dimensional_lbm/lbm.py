"""Lattice Boltzmann Method (LBM) simulation framework.

This module provides the core LBM simulation class that implements the fundamental
algorithms for fluid dynamics simulation using the Lattice Boltzmann approach.
The implementation supports dimensional and non-dimensional modes through a
generic unit system interface.

The LBM algorithm follows the standard workflow:
    1. Compute macroscopic moments (density, velocity) from distribution functions
    2. Calculate equilibrium distribution functions
    3. Perform collision step (BGK single relaxation time model)
    4. Stream distribution functions to neighboring lattice nodes
    5. Apply boundary conditions
"""

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, cast

import numpy as np

from dimensional_lbm.boundaries.boundary import Boundary
from dimensional_lbm.conversion_mode import ModeT
from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition, ScalarT, UnitSystem, VectorT


class LBM(ABC, Generic[ModeT, ScalarT, VectorT]):
	"""Base class for Lattice Boltzmann Method simulations.

	This class provides the fundamental framework for LBM simulations, implementing
	the core collision, streaming, and moment calculation routines. It is designed
	to be subclassed for specific simulation scenarios (e.g., Couette flow, lid-driven
	cavity, etc.).

	The class is generic over:
		- ModeT: The conversion mode (Dimensional or NonDimensional)
		- ScalarT: The scalar quantity type (with or without units)
		- VectorT: The vector/array quantity type (with or without units)

	Attributes:
		f (VectorT): Distribution functions at each lattice node, shape (Q, y, x).
		fcoll (VectorT): Post-collision distribution functions, shape (Q, y, x).
		feq (VectorT): Equilibrium distribution functions, shape (Q, y, x).
		u (VectorT): Macroscopic velocity field, shape (D, y, x).
		density (VectorT): Macroscopic density field, shape (y, x).
		us (UnitSystem[ModeT]): Unit system for handling dimensional quantities.
		solid (np.ndarray): Boolean mask for solid nodes (obstacles), shape (y, x).
	"""

	f: VectorT
	fcoll: VectorT
	feq: VectorT
	u: VectorT
	density: VectorT

	us: UnitSystem[ModeT]

	_width: ScalarT
	_height: ScalarT
	_x: int
	_y: int
	_runs: int
	_lattice: DdQqLattice
	_boundary: Boundary
	_dt: ScalarT

	@property
	def x(self) -> int:
		"""The amount of cells of the lattice in x direction (i.e., the lattice width)."""
		return self._x

	@property
	def y(self) -> int:
		"""The amount of cells of the lattice in y direction (i.e., the lattice height)."""
		return self._y

	@property
	def lattice(self) -> DdQqLattice:
		"""The lattice structure used for the simulation (e.g., D2Q5, D2Q9)."""
		return self._lattice

	@lattice.setter
	def lattice(self, lattice: DdQqLattice) -> None:
		self._lattice = lattice

		if hasattr(self ,"_width"):
			self._set_lattice_width()

		if hasattr(self ,"_height"):
			self._set_lattice_height()

		if self._x > 0 and self._y > 0:
			self.initialize_fields()

	@property
	def boundary(self) -> Boundary:
		"""Boundary conditions used for the simulation."""
		return self._boundary

	@boundary.setter
	def boundary(self, boundary: Boundary) -> None:
		if not self.lattice:
			msg = "Cannot specify the boundary conditions without the lattice being set prior to it."
			raise ValueError(msg)
		self._boundary = boundary

	@property
	def dx(self) -> ScalarT:
		"""Lattice spacing (spatial discretization step)."""
		return self._lattice.dx

	@property
	def dt(self) -> ScalarT:
		"""Time step (temporal discretization step)."""
		return self._lattice.dt

	@property
	def width(self) -> ScalarT:
		"""Width of the simulation lattice."""
		return self._width

	@width.setter
	def width(self, width: ScalarT) -> None:
		self._width = width

		if hasattr(self, "_lattice"):
			self._set_lattice_width()

			if hasattr(self, "height"):
				self.initialize_fields()

	@property
	def height(self) -> ScalarT:
		"""Height of the simulation lattice."""
		return self._height

	@height.setter
	def height(self, height: ScalarT) -> None:
		if self.us.magnitude(height) <= 0:
			msg = f"The 'height' of the simulation domain must be greater 0 but {height} was provided."
			raise ValueError(msg)

		self._height = height

		if hasattr(self, "_lattice"):
			self._set_lattice_height()

			if hasattr(self, "width"):
				self.initialize_fields()

	def __init__(self) -> None:
		"""Initialize the LBM simulation with default values."""
		self._runs = 0
		self._x = 0
		self._y = 0

	def single_step(self) -> None:
		"""Execute a single LBM time step.

		Performs the complete LBM cycle:
			1. Update macroscopic moments (density, u) from distribution functions
			2. Compute equilibrium distribution functions
			4. Stream distribution functions to neighbors
			5. Apply boundary conditions
			6. Increment run counter
		"""
		self.moments()
		self.lattice.equilibrium(self.density, self.u, self.feq)
		self.collide()
		self.stream()
		self._boundary.apply_boundaries(self.f, self.density, self.u, self._runs * self.dt)
		self._runs += 1

	def initialize_populations(self) -> None:
		"""Initialize distribution functions to equilibrium.

		Computes the equilibrium distribution based on the current macroscopic
		fields (density, u) and sets f = f_eq as the initial condition.
		"""
		self.lattice.equilibrium(self.density, self.u, self.feq)
		self.f = cast("VectorT", self.feq.copy())

	def check_parameters_set(self) -> None:
		"""Initialize simulation parameters and allocate arrays.

		Validates that all required parameters (dx, dt, lattice, width, height)
		have been set in the setup() method, then:
			- Computes grid dimensions from physical dimensions and spacing
			- Allocates the solid mask array
			- Allocates distribution function arrays (f, feq, fcoll)
			- Allocates macroscopic field arrays (u, density)

		Raises:
			ValueError: If any required parameter is not set.
		"""
		required_properties = ["_lattice", "_boundary", "_width", "_height"]

		for req_prop in required_properties:
			if not getattr(self, req_prop):
				msg: str = f"The property {req_prop[1:]} must be set within the scenario's `define_scenario` method"
				raise ValueError(msg)

	def moments(self) -> None:
		"""Compute macroscopic moments from distribution functions.

		Calculates the density (density) and velocity (u) fields from the
		distribution functions using:
			density = sum_i(f_i)
			u = (1/density) * sum_i(f_i * c_i)

		where c_i are the lattice velocity vectors.
		"""
		self.density = self.f.sum(axis=0)

		# TODO: make work for arbitrary lattice dimensions
		self.u[:, :, 0] = self.lattice.q * np.sum(self.lattice.dir_x[:, None, None] * self.f, axis=0) / self.density
		self.u[:, :, 1] = self.lattice.q * np.sum(self.lattice.dir_y[:, None, None] * self.f, axis=0) / self.density

	@abstractmethod
	def collide(self) -> None:
		pass

	def stream(self) -> None:
		"""Stream distribution functions to neighboring nodes.

		Propagates the post-collision distribution functions (fcoll) along
		their respective lattice velocity directions using periodic boundary
		conditions. The result is stored in f.
		"""
		self.lattice.stream(self.f, self.fcoll)

	def stream_periodic(self) -> None:
		self.lattice.stream_periodic(self.f, self.fcoll)

	def _set_unit_system(self, us: UnitSystem[ModeT]) -> None:
		"""Set the unit system for the simulation.

		Args:
			us: The unit system to use (Dimensional or NonDimensional mode).
				This determines how quantities are created and converted.
		"""
		self.us = us

	def initialize_fields(self) -> None:
		self.f = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x), dtype=np.float64), "kg/m**3")
		self.feq = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x), dtype=np.float64), "kg/m**3")
		self.fcoll = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x), dtype=np.float64), "kg/m**3")
		self.u = self.us.quantity(np.zeros((self._y, self._x, self._lattice.D), dtype=np.float64), "m/s")
		self.density = self.us.quantity(np.zeros((self._y, self._x), dtype=np.float64), "kg/m**3")

	def _set_lattice_width(self) -> None:
		x = int(self.us.magnitude(self._width / self._lattice.dx))
		if x <= 0:
			msg = f"'width' of the lattice must be greater 0 but is {x} for the provided domain width of '{self._width}' and dx '{self._lattice.dx}'."
			raise ValueError(msg)
		self._x = x

	def _set_lattice_height(self) -> None:
		y = int(self.us.magnitude(self._height / self._lattice.dx))
		if y <= 0:
			msg = f"'height' of the lattice must be greater 0 but is {y} for the provided domain width of '{self._height}' and dx '{self._lattice.dx}'."
			raise ValueError(msg)
		self._y = y
