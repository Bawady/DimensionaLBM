"""Bacterial colony growth patterns (De Rosis, Harish & Wang 2024).

Reproduces the patterns from "Lattice Boltzmann modelling of bacterial colony patterns"
"""

import pathlib
from typing import TYPE_CHECKING

import matplotlib.image as plt_img
import numpy as np

from dimensional_lbm.adr_lbm import AdrLBM, from_diffusivity
from dimensional_lbm.conversion_mode import NonDimensional
from dimensional_lbm.lattices.d2q5 import D2Q5
from scenarios.scenario import Scenario

if TYPE_CHECKING:
	import os

	from dimensional_lbm.unit_system_if import ScalarQuantityDefinition


class DHW24(Scenario[AdrLBM]):
	def define(self, lbm: AdrLBM) -> None:
		lbm.width = lbm.us.quantity(5, "cm")
		lbm.height = lbm.us.quantity(5, "cm")
		dx = lbm.us.quantity(50, "um")
		dt = lbm.us.quantity(0.47, "s")
		lbm.lattice = D2Q5(dx, dt)

		tau_sub = 2.5 * dt
		lbm.nutrients = lbm.add_species(tau=tau_sub, unit="g/ml")
		# [0.07, 0.02, 0.025, 0.028] g / ml
		lbm.nutrients.density[:, :] = lbm.us.quantity(0.028, "g/ml")

		# [1.33e-5, 6.65e-6, 2.66e-6] cm**2/s
		diffusion_bac = lbm.us.quantity(2.66e-6, "cm**2/s") # bacteria diffusivity in 2-4e-6 acc to Murray (Intro to Bio Mod 2)
		tau_bac = from_diffusivity(diffusion_bac, lbm.lattice)
		lbm.bacteria = lbm.add_species(tau=tau_bac, unit="cfu/ml")
		lbm.bacteria.density[lbm.y // 2, lbm.x // 2] = lbm.us.quantity(1e9, "cfu/ml")

		lbm.inactive = lbm.us.quantity(np.zeros((lbm.y, lbm.x)), "cfu/ml")

		alpha1 = lbm.us.quantity(416.7, "cfu / ul")
		alpha2 = lbm.us.quantity(2.33, "kg/m**3")

		k1 = lbm.us.quantity(7.6, "ml/(g * s)")
		k2 = lbm.us.quantity(2.13e-9, "ml/(cfu * s)")
		k3 = lbm.us.quantity(2.12, "1/s")

		def react() -> None:
			sub, bac = lbm.nutrients, lbm.bacteria
			growth = sub.density * np.maximum(bac.density, 0)
			# Pint's NumPy operator stubs are incomplete, so Pyright mis-types this dimensionally
			# correct arithmetic on quantity arrays (spurious datetime union); runtime is unaffected.
			death = bac.density / ((1 + sub.density / alpha2) * (1 + bac.density / alpha1))  # pyright: ignore[reportOperatorIssue]
			lbm.inactive += death  # pyright: ignore[reportAttributeAccessIssue]
			for i, w in enumerate(lbm.lattice.weights):
				bac.fcoll[i] += w * lbm.dt * (k1 * growth - k3 * death)
				sub.fcoll[i] -= w * lbm.dt * k2 * growth

		lbm.react = react
		lbm.stream = lbm.stream_periodic

	def dump(self, lbm: AdrLBM, dump_dir: os.PathLike) -> None:
		dump_dir = pathlib.Path(dump_dir)
		dump_dir.mkdir(exist_ok=True)

		# RGB: red=nutrients, green=bacteria, blue=inactive
		rgb = np.zeros((lbm.y, lbm.x, 3), dtype=np.uint8)
		sub_mag = lbm.us.magnitude(lbm.nutrients.density)
		rgb[:, :, 0] = np.clip(sub_mag / max(1e-12, np.max(sub_mag)) * 255, 0, 255).astype(np.uint8)
		bac_mag = lbm.us.magnitude(lbm.bacteria.density)
		rgb[:, :, 1] = np.clip(bac_mag / max(1e-12, np.max(bac_mag)) * 255, 0, 255).astype(np.uint8)
		inactive_mag = lbm.us.magnitude(lbm.inactive)
		rgb[:, :, 2] = np.clip(inactive_mag  / max(1e-12, np.max(inactive_mag)) * 255, 0, 255).astype(np.uint8)

		plt_img.imsave(dump_dir / f"colony_{lbm.runs:06d}.png", rgb, dpi=300)


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(50, "um"), (.47, "s"), (0.035, "microgram"), (125, "cfu")]
	sim = DHW24(AdrLBM, characteristic_quantities, conversion_mode=NonDimensional)
	sim.run(5000, dump_period=250, dump_dir=pathlib.Path("test/dhw24_exp"))
