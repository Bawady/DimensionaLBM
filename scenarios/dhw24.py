"""Bacterial colony growth patterns (De Rosis, Harish & Wang 2024).

Reproduces the patterns from "Lattice Boltzmann modelling of bacterial colony patterns"
using two coupled D2Q5 diffusion LBMs for substrate and bacteria concentrations.
"""

import os
import pathlib

import matplotlib.image as plt_img
import numpy as np

from dimensional_lbm.adr_lbm import AdrLBM
from dimensional_lbm.conversion_mode import Dimensional, NonDimensional
from dimensional_lbm.lattices.d2q5 import D2Q5
from dimensional_lbm.unit_system_if import ScalarQuantityDefinition
from scenarios.scenario import Scenario


class DHW24(Scenario[AdrLBM]):
	"""Bacterial colony growth on a periodic domain.

	Three coupled fields are evolved:
	- substrate: nutrient that diffuses and is consumed by bacteria
	- bacteria:  motile cells that grow by consuming substrate and die due to stress
	- dead:      accumulated dead cells (no diffusion)
	"""

	def define(self, lbm: AdrLBM) -> None:
		lbm.width = lbm.us.quantity(10, "cm")
		lbm.height = lbm.us.quantity(10, "cm")
		lbm.lattice = D2Q5(lbm.us.quantity(0.2, "mm"), lbm.us.quantity(1, "s"))

		tau_sub = lbm.us.quantity(2.5, "s")
		lbm.substrate = lbm.add_species(tau=tau_sub, unit="g/ml")
		lbm.substrate.density[:, :] = lbm.us.quantity(0.72, "g/ml")

		# tau_bac from relative diffusivity: D_bac = diff_bac * D_sub,
		# D = (tau - dt/2) * cs^2  =>  tau_bac = dt/2 + diff_bac * (tau_sub - dt/2)
		diffusion_bac = lbm.us.quantity(1.33e-5, "cm**2/s") #0.05 * (tau_sub - 0.5) * lbm.lattice.cs**2 # bacteria diffusivity relative to substrate (Agar; from Murray)
		tau_bac = 1.67 * (diffusion_bac * lbm.lattice.cs_n2 + lbm.lattice.dt / 2)
		lbm.bacteria = lbm.add_species(tau=tau_bac, unit="cfu/ml")
		lbm.bacteria.density[lbm.y // 2, lbm.x // 2] = lbm.us.quantity(8e3, "cfu/ml") # should be in non_dim, inspired by 1e9 cfu/ml found somewhere

		lbm.dead = lbm.us.quantity(np.zeros((lbm.y, lbm.x)), "cfu/ml")

#		alpha1 = 2400 #lbm.us.quantity(41.67e3, "ml/mol")
#		alpha2 = 120 #lbm.us.quantity(1.67e-5, "l/mol")
		alpha1 = lbm.us.quantity(96, "ml/cfu")
		alpha2 = lbm.us.quantity(4.8, "ml/g")

		k1 = lbm.us.quantity(0.04, "ml/cfu")
		k2 = lbm.us.quantity(0.04, "ml/g")

		print("Diff bacteria", diffusion_bac)
		print("tau bacteria", tau_bac)
		print("tau substrate", tau_sub)
		print("alpha1", alpha1)
		print("alpha2", alpha2)

		def react() -> None:
			sub, bac = lbm.substrate, lbm.bacteria
			growth = sub.density * np.maximum(bac.density, 0)
			death = bac.density / (
				(1 + sub.density * alpha2) * (1 + bac.density * alpha1)
			)
			lbm.dead += death
			for i, w in enumerate(lbm.lattice.weights):
				sub.fcoll[i] -= w * k1 * growth
				bac.fcoll[i] += w * (k2 * growth - death)

		lbm.react = react
		lbm.stream = lbm.stream_periodic

	def dump(self, lbm: AdrLBM, dump_dir: os.PathLike) -> None:
		dump_dir = pathlib.Path(dump_dir)
		dump_dir.mkdir(exist_ok=True)

		# RGB: red=substrate, green=bacteria, blue=dead
		rgb = np.zeros((lbm.y, lbm.x, 3), dtype=np.uint8)
		sub_mag = lbm.us.magnitude(lbm.substrate.density)
		rgb[:, :, 0] = np.clip((sub_mag - np.min(sub_mag)) / max(1e-12, np.max(sub_mag)) * 255, 0, 255).astype(np.uint8)
		bac_mag = lbm.us.magnitude(lbm.bacteria.density)
		rgb[:, :, 1] = np.clip((bac_mag - np.min(bac_mag)) / max(1e-12, np.max(bac_mag)) * 255, 0, 255).astype(np.uint8)
		dead_mag = lbm.us.magnitude(lbm.dead)
		rgb[:, :, 2] = np.clip((dead_mag- np.min(dead_mag)) / max(1e-12, np.max(dead_mag)) * 255, 0, 255).astype(np.uint8)

#		rgb[:, :, 1] = np.clip(lbm.us.magnitude(lbm.bacteria.density) * 8 * 255, 0, 255).astype(np.uint8)
#		rgb[:, :, 2] = np.clip(lbm.us.magnitude(lbm.dead) * 8 * 255, 0, 255).astype(np.uint8)

		plt_img.imsave(dump_dir / f"colony_{lbm._runs:06d}.png", rgb, dpi=300)


if __name__ == "__main__":
	characteristic_quantities: list[ScalarQuantityDefinition] = [(0.2, "mm"), (1, "s"), (2.8, "g/ml"), (8e3, "cfu")]
	sim = DHW24(AdrLBM, characteristic_quantities, conversion_mode=NonDimensional)
	sim.run(2500, dump_period=100, dump_dir=pathlib.Path("test/dhw24_exp"))
