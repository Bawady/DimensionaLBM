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

import os
from pathlib import Path
from typing import ClassVar, Generic

import matplotlib.image as plt_img
import numpy as np
from matplotlib import cm

from dimensional_lbm.boundaries.boundary import Boundary
from dimensional_lbm.conversion_mode import Dimensional, ModeT
from dimensional_lbm.lattices.ddqq_lattice import DdQqLattice
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition, ScalarT, UnitSystem, VectorT


class LBM(Generic[ModeT, ScalarT, VectorT]):
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
		rho (VectorT): Macroscopic density field, shape (y, x).
		us (UnitSystem[ModeT]): Unit system for handling dimensional quantities.
		characteristic_quantities (ClassVar): List of characteristic quantities for
			non-dimensionalization.
		solid (np.ndarray): Boolean mask for solid nodes (obstacles), shape (y, x).
		bgk_tau (ScalarT): BGK relaxation time parameter.

	Example:
		>>> class MyCouetteFlow(LBM):
		...     def setup(self):
		...         self.lattice = D2Q9()
		...         self.dx = self.us.quantity(0.01, "m")
		...         self.dt = self.us.quantity(0.001, "s")
		...         # ... set other parameters
		...
		...     def boundaries(self):
		...         # Apply boundary conditions
		...         pass
	"""

	f: VectorT
	fcoll: VectorT
	feq: VectorT
	u: VectorT
	rho: VectorT

	us: UnitSystem[ModeT]

	characteristic_quantities: ClassVar[list[ScalarQuantityDefinition]] = []
	solid: np.ndarray

	_width: ScalarT
	_height: ScalarT
	_x: int
	_y: int
	_runs: int
	_lattice: DdQqLattice
	_boundary: Boundary
	_dx: ScalarT
	_dt: ScalarT

	_bgk_tau: ScalarT
	_bgk_omega: ScalarT

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

	def set_boundary(self, boundary: Boundary) -> None:
		if not self.lattice:
			msg = "Cannot specify the boundary conditions without the lattice being set prior."
			raise ValueError(msg)
		self._boundary = boundary

	@property
	def dx(self) -> ScalarT:
		"""Lattice spacing (spatial discretization step)."""
		return self._dx

	@dx.setter
	def dx(self, dx: ScalarT) -> None:
		self._dx = dx

	@property
	def dt(self) -> ScalarT:
		"""Time step (temporal discretization step)."""
		return self._dt

	@dt.setter
	def dt(self, dt: ScalarT) -> None:
		self._dt = dt

	@property
	def width(self) -> ScalarT:
		"""Width of the simulation lattice."""
		return self._width

	@width.setter
	def width(self, width: ScalarT) -> None:
		self._width = width

	@property
	def height(self) -> ScalarT:
		"""Height of the simulation lattice."""
		return self._height

	@height.setter
	def height(self, height: ScalarT) -> None:
		self._height = height

	@property
	def bgk_tau(self) -> ScalarT:
		"""BGK relaxation time, related to the fluid viscosity."""
		return self._bgk_tau

	@bgk_tau.setter
	def bgk_tau(self, tau: ScalarT) -> None:
		self._bgk_tau = tau
		self._bgk_omega = 1 / tau

	def __init__(self):
		"""Initialize the LBM simulation with default values."""
		self._runs = 0
		self._x = 0
		self._y = 0

	def setup(self) -> None:
		"""Configure simulation parameters before running.

		Override this method in subclasses to set up:
			- Lattice type (e.g., D2Q9)
			- Grid spacing (dx) and time step (dt)
			- Domain dimensions (width, height)
			- BGK relaxation time (bgk_tau)

		This method is called automatically before before the simulation starts with a call to `run`.
		"""
		pass

	def define_scenario(self) -> None:
		"""Define the initial conditions and scenario-specific setup.

		Override this method in subclasses to set up:
			- Initial velocity and density fields
			- Solid geometry (obstacles)
			- Any scenario-specific parameters

		This method is called after setup() and __init_sim_params(), so all arrays
		(f, feq, u, rho, solid) are already allocated and available for initialization.
		"""
		pass

	def viscosity_to_bgk_tau(self, viscosity: ScalarT) -> ScalarT:
		"""Convert kinematic viscosity to BGK relaxation time.

		Computes the BGK relaxation parameter tau from the kinematic viscosity
		using the relation:
			tau = dt/2 + nu * cs^{-2}

		where cs is the lattice speed of sound.

		Args:
			viscosity: Kinematic viscosity (nu) of the fluid.

		Returns:
			The BGK relaxation time tau.

		Raises:
			ValueError: If the lattice has not been set (required for speed of sound).
		"""
		if not self.lattice:
			msg: str = "Cannot compute the BGK relaxation time tau without the lattice having been declared prior - the speed of sound must be known."
			raise ValueError(msg)
		return self.dt / 2 + viscosity * self.lattice.cs_n2

	def _sim_step(self) -> None:
		"""Execute a single LBM time step.

		Performs the complete LBM cycle:
			1. Update macroscopic moments (rho, u) from distribution functions
			2. Compute equilibrium distribution functions
			3. Perform BGK collision
			4. Stream distribution functions to neighbors
			5. Apply boundary conditions
			6. Increment run counter
		"""
		self.update_moments()
		self.update_feq()
		self.collide()
		self.stream()
		self._boundary.apply_boundaries(self.f, self.rho, self.u)
		self._runs += 1

	def initialize_distribution_function(self) -> None:
		"""Initialize distribution functions to equilibrium.

		Computes the equilibrium distribution based on the current macroscopic
		fields (rho, u) and sets f = f_eq as the initial condition.
		"""
		self.update_feq()
		self.f = self.feq.copy()

	def run(self, runs: int, dump_period: int, out_dir_p: Path) -> None:
		"""Run the LBM simulation for a specified number of time steps.

		Executes the complete simulation workflow:
			1. Call setup() to configure parameters
			2. Initialize simulation parameters and allocate arrays
			3. Call define_scenario() for initial conditions
			4. Initialize distribution functions to equilibrium
			5. Run the main simulation loop with periodic output dumps

		Args:
			runs: Total number of time steps to simulate.
			dump_period: Interval between output dumps (in time steps).
			out_dir_p: Output directory path for simulation snapshots.
		"""
		for i in range(runs):
			if i % dump_period == 0:
				self.dump(out_dir_p)
			self._sim_step()

	def init_sim_params(self) -> None:
		"""Initialize simulation parameters and allocate arrays.

		Validates that all required parameters (dx, dt, lattice, width, height)
		have been set in the setup() method, then:
			- Computes grid dimensions from physical dimensions and spacing
			- Allocates the solid mask array
			- Allocates distribution function arrays (f, feq, fcoll)
			- Allocates macroscopic field arrays (u, rho)

		Raises:
			ValueError: If any required parameter is not set.
		"""
		required_properties = ["_dx", "_dt", "_lattice", "_width", "_height"]

		for req_prop in required_properties:
			if not getattr(self, req_prop):
				msg: str = f"The property {req_prop[1:]} must be set within the scenario's setup method"
				raise ValueError(msg)

		self._x = int(self.us.magnitude(self._width / self._dx))
		self._y = int(self.us.magnitude(self._height / self._dx))

		self.solid = np.zeros((self._y, self._x), dtype=np.int32)

		self.f = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x), dtype=np.float64), "kg/m**3")
		self.feq = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x), dtype=np.float64), "kg/m**3")
		self.fcoll = self.us.quantity(np.zeros((self._lattice.Q, self._y, self._x), dtype=np.float64), "kg/m**3")
		self.u = self.us.quantity(np.zeros((self._lattice.D, self._y, self._x), dtype=np.float64), "m/s")
		self.rho = self.us.quantity(np.zeros((self._y, self._x), dtype=np.float64), "1")

	def update_moments(self) -> None:
		"""Compute macroscopic moments from distribution functions.

		Calculates the density (rho) and velocity (u) fields from the
		distribution functions using:
			rho = sum_i(f_i)
			u = (1/rho) * sum_i(f_i * c_i)

		where c_i are the lattice velocity vectors.
		"""
		self.rho = self.f.sum(axis=0)

		# TODO: make work for arbitrary lattice dimensions
		self.us.magnitude(self.u)[0] = self.lattice.q * np.sum(self.lattice.dir_x[:, None, None] * self.f, axis=0) / self.rho
		self.us.magnitude(self.u)[1] = self.lattice.q * np.sum(self.lattice.dir_y[:, None, None] * self.f, axis=0) / self.rho

	def update_feq(self) -> None:
		"""Compute equilibrium distribution functions.

		Calculates the Maxwell-Boltzmann equilibrium distribution expanded
		to second order in velocity:
			f_eq_i = w_i * rho * (1 + (c_i . u)/cs^2 + (c_i . u)^2/(2*cs^4) - u^2/(2*cs^2))

		where w_i are the lattice weights and cs is the speed of sound.
		"""
		cs_n2 = self.lattice.cs_n2
		vel_x = self.lattice.dir_x * self.lattice.q
		vel_y = self.lattice.dir_y * self.lattice.q
		ws = self.lattice.weights
		u_sq = cs_n2 / 2.0 * (self.u[0] ** 2 + self.u[1] ** 2)
		for i in range(self.lattice.Q):
			lin_term = cs_n2 * (vel_x[i] * self.u[0] + vel_y[i] * self.u[1])
			self.feq[i] = ws[i] * self.rho * (1 + lin_term + lin_term**2 / 2.0 - u_sq)

	def stream(self) -> None:
		"""Stream distribution functions to neighboring nodes.

		Propagates the post-collision distribution functions (fcoll) along
		their respective lattice velocity directions using periodic boundary
		conditions. The result is stored in f.
		"""
		self.lattice.stream_periodic(self.f, self.fcoll)

	def collide(self) -> None:
		"""Perform BGK collision step.

		Applies the Bhatnagar-Gross-Krook (BGK) single relaxation time collision:
			f_coll = (1 - omega*dt) * f + omega*dt * f_eq

		where omega = 1/tau is the relaxation frequency.
		"""
		relax_factor = self.dt * self._bgk_omega
		self.fcoll = (1 - relax_factor) * self.f + relax_factor * self.feq

	def boundaries(self) -> None:
		"""Apply boundary conditions.

		Override this method in subclasses to implement specific boundary
		conditions such as:
			- No-slip walls (bounce-back)
			- Moving walls (Zou-He or other velocity BCs)
			- Pressure/density boundaries
			- Periodic boundaries (handled separately in stream)

		Called at the end of each simulation step after streaming.
		"""
		pass

	def dump(self, dump_dir_p: os.PathLike) -> None:
		"""Save simulation snapshots to disk.

		Outputs density and velocity magnitude fields as PNG images. Solid
		regions are overlaid with a red tint for visualization. In dimensional
		mode, also performs unit consistency assertions.

		Args:
			dump_dir_p: Directory path where output files will be saved.
				The directry is created if it does not yet exist.
				Files are named "density_{step}.png" and "fluid_velocity_{step}.png".
		"""
		dump_dir_p.mkdir(exist_ok=True)

		cmap = cm.get_cmap("viridis")
		density_rgba = cmap(self.rho)

		fluid_vel_abs = np.sqrt(self.u[0] ** 2 + self.u[1] ** 2)
		fluid_vel_rgba = cmap(fluid_vel_abs)

		overlay = np.array([1, 0, 0, 0.5])
		mask = self.solid == 1

		dump_data = {"density": density_rgba, "fluid_velocity": fluid_vel_rgba}

		for name, rgba in dump_data.items():
			for i in range(3):
				rgba = dump_data[name]
				rgba[mask, i] = overlay[3] * overlay[i] + (1 - overlay[3]) * rgba[mask, i]
			plt_img.imsave(dump_dir_p / f"{name}_{self._runs}.png", rgba, dpi=600)

		if isinstance(self.us.mode, Dimensional):
			assert self.rho.dimensionality == self.us.quantity(1, "kg/m**3").dimensionality
			assert self.u.dimensionality == self.us.quantity(1, "m/s").dimensionality

	def _set_unit_system(self, us: UnitSystem[ModeT]) -> None:
		"""Set the unit system for the simulation.

		Args:
			us: The unit system to use (Dimensional or NonDimensional mode).
				This determines how quantities are created and converted.
		"""
		self.us = us
