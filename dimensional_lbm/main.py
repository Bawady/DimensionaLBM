from pathlib import Path

import scenarios
from dimensional_lbm.unit_system_if import Dimensional, MagnitudeOnly, NonDimensional  # noqa: F401

if __name__ == "__main__":
	# Set desired test here
	runs = 100
	dump_period = 10

	store_at_p = Path("test")
	if not store_at_p.exists():
		store_at_p.mkdir(exist_ok=True, parents=True)

	sim = scenarios.factory.create("Couette", conversion_mode=Dimensional)
	sim.run(runs, dump_period, store_at_p)
