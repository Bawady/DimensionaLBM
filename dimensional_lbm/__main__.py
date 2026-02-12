from pathlib import Path

import scenarios
from dimensional_lbm.conversion_mode import Dimensional, MagnitudeOnly, NonDimensional  # noqa: F401

if __name__ == "__main__":
	# Set desired test here
	runs = 100
	dump_period = 2

	store_at_p = Path("test")
	if not store_at_p.exists():
		store_at_p.mkdir(exist_ok=True, parents=True)

	sim = scenarios.factory.create("Couette", conversion_mode=NonDimensional)

	sim.run(runs, dump_period, store_at_p)
