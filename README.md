# DimensionaLBM

A lattice Boltzmann (LBM) framework for rapid, safe prototyping of LBM models for non-traditional fluid flows. Every quantity
carries its physical units (via [pint](https://pint.readthedocs.io)), so scenarios are written in
real-world units and the unit system catches dimensional mistakes before a simulation ever runs.
The same scenario code can be executed either dimensionally or in an automatically non-dimensionalised
mode. The framework is fully type-annotated and passes `pyright` and `ruff` checks.

## Repository layout

```
dimensional_lbm/              Core framework
├── lbm.py                    Abstract LBM corse class: moments, equilibrium, collision, streaming, time loop
├── bgk_lbm.py                BGK single-relaxation-time model
├── trt_lbm.py                TRT two-relaxation-time model
├── adr_lbm.py                Multi-species advection–diffusion–reaction LBM
├── unit_system_if.py         Unit system: builds/converts unit-carrying quantities (optionally pintrs-backed)
├── conversion_mode.py        Dimensional / NonDimensional mode tags
├── _typing.py                Shared type aliases
├── lattices/
│   ├── ddqq_lattice.py        Abstract DdQq lattice base (weights, directions, streaming)
│   ├── d2q9.py                D2Q9 stencil (fluid flow)
│   └── d2q5.py                D2Q5 stencil (scalar transport)
└── boundaries/
    ├── boundary.py            Boundary base class + BoundaryCollection
    ├── zou_he.py              Zou–He velocity/pressure boundaries (incl. corner handling)
    ├── zero_gradient.py       Zero-gradient (outflow) boundaries
    ├── wall_detector.py       Detects walls/corners from a solid/fluid geometry mask
    └── callbacks.py           Protocols for time-dependent boundary callbacks

scenarios/                    Ready-to-run example simulations
├── scenario.py               Scenario base class (setup -> run -> dump)
├── couette.py                Couette flow (vs. analytical profile)
├── poiseuille.py             Poiseuille channel flow
├── ldc.py                    Lid-driven cavity (Ghia et al. validation; writes ldc.csv)
├── schaefer_turek.py         Schäfer–Turek 2D cylinder benchmark (laminar flow around a cylinder)
└── dhw24.py                  Bacterial colony growth (De Rosis, Harish & Wang 2024)

geometries/                   Geometry mask images (pipe.png)
pyproject.toml                Project metadata and dependencies
pyrightconfig.json, .ruff.toml  Type-checking and linting configuration
LICENSE                       MIT
```

## Installation

Requires **Python ≥ 3.14.4**. Using [uv](https://docs.astral.sh/uv/):

```sh
uv venv                       # create .venv (Python >= 3.14.4)
uv pip install -e .           # install DimensionaLBM and its dependencies
source .venv/bin/activate
```

(Equivalently, in any Python ≥ 3.14.4 virtual environment: `pip install -e .`.)

## Running the scenarios

Each scenario is runnable as a module from the repository root. Image/CSV output is written to the
`<scenario>/` directory configured in the scenario's `__main__` block.

```sh
python -m scenarios.couette          # Couette flow
python -m scenarios.poiseuille       # Poiseuille channel flow
python -m scenarios.ldc              # Lid-driven cavity (also writes ldc.csv)
python -m scenarios.schaefer_turek   # Schäfer–Turek cylinder benchmark
python -m scenarios.dhw24            # Bacterial colony growth
```

## Defining a new scenario

A scenario subclasses `Scenario[<LBM type>]` and implements `define(self, lbm)`, where the geometry,
lattice, fluid properties and boundary conditions are set using `lbm.us.quantity(value, unit)` to
create unit-carrying quantities. The collision model is chosen by the LBM class passed at
construction (e.g., `BGKLBM`, `TRTLBM`, or `AdrLBM`).

```python
from pathlib import Path

from dimensional_lbm.bgk_lbm import BGKLBM, from_viscosity
from dimensional_lbm.boundaries.zou_he import ZouHe
from dimensional_lbm.lattices.d2q9 import D2Q9
from scenarios.scenario import Scenario


class MyFlow(Scenario[BGKLBM]):
    # Optional: register domain-specific Pint units this scenario uses (omit if none).
    custom_units = ["cfu = [population]"]

    def define(self, lbm: BGKLBM) -> None:
        # Domain and discretisation (all in physical units)
        lbm.width = lbm.us.quantity(1.0, "m")
        lbm.height = lbm.us.quantity(0.2, "m")
        lbm.lattice = D2Q9(lbm.us.quantity(5, "mm"), lbm.us.quantity(1, "ms"))

        # Fluid: relaxation time from kinematic viscosity, uniform initial density
        lbm.tau = from_viscosity(lbm.us.quantity(1e-6, "m**2/s"), lbm.lattice)
        lbm.density[:, :] = lbm.us.quantity(1000, "kg/m**3")

        # Boundaries: mark cells in the geometry mask, then register the boundary
        bc = ZouHe(lbm)
        bc.geometry[0, :] = 1            # e.g. a wall along the top row
        lbm.boundaries += bc


if __name__ == "__main__":
    MyFlow(BGKLBM).run(2000, dump_period=100, dump_dir=Path("myflow"))
```

Notes:

- **Output** — override `dump(self, lbm, dump_dir)` to control what is written each `dump_period`
  steps, or `post_run(self, lbm)` for end-of-run analysis (see `couette.py` / `ldc.py`).
- **Non-dimensional mode** — pass characteristic quantities to run the *same* `define` code:
  `MyFlow(BGKLBM, characteristic_quantities=[(1, "m"), (1, "s"), (1, "kg/m**3")], conversion_mode=NonDimensional)`.
- **Time-dependent boundaries** — assign a callback to a boundary profile (e.g., for Zou-He), e.g.
  `zou_he.velocity_profile[0, x] = lambda time, u=u0: u * np.array([1, 0])`.
- **Custom units** — declare domain-specific units on the scenario class via the `custom_units`
  attribute, e.g. `custom_units = ["cfu = [population]"]`; they are registered before `define` runs
  (see `dhw24.py`).

## License

Released under the MIT License (see `LICENSE`).
