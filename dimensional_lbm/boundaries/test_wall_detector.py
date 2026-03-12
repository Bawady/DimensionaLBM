from dataclasses import dataclass, field

import numpy as np

from dimensional_lbm.boundaries.wall_detector import WallDetector


def _corners_to_tuples(corners: np.ndarray) -> list[tuple[int, int]]:
	"""Convert (2, N) corner array to sorted list of (y, x) tuples."""
	if corners.size == 0:
		return []
	return sorted(zip(corners[0].tolist(), corners[1].tolist(), strict=True))


@dataclass
class ExpectedDetection:
	left_walls: list[tuple[int, int, int]] = field(default_factory=list)
	right_walls: list[tuple[int, int, int]] = field(default_factory=list)
	top_walls: list[tuple[int, int, int]] = field(default_factory=list)
	bot_walls: list[tuple[int, int, int]] = field(default_factory=list)
	conc_tl: list[tuple[int, int]] = field(default_factory=list)
	conc_tr: list[tuple[int, int]] = field(default_factory=list)
	conc_bl: list[tuple[int, int]] = field(default_factory=list)
	conc_br: list[tuple[int, int]] = field(default_factory=list)
	conv_tl: list[tuple[int, int]] = field(default_factory=list)
	conv_tr: list[tuple[int, int]] = field(default_factory=list)
	conv_bl: list[tuple[int, int]] = field(default_factory=list)
	conv_br: list[tuple[int, int]] = field(default_factory=list)


def assert_detection(detector: WallDetector, expected: ExpectedDetection, name: str) -> None:
	errors = []

	for attr in ["left_walls", "right_walls", "top_walls", "bot_walls"]:
		actual = sorted(getattr(detector, attr))
		exp = sorted(getattr(expected, attr))
		if actual != exp:
			errors.append(f"  {attr}: expected {exp}, got {actual}")

	for attr in ["conc_tl", "conc_tr", "conc_bl", "conc_br",
	             "conv_tl", "conv_tr", "conv_bl", "conv_br"]:  # noqa: E101
		actual = _corners_to_tuples(getattr(detector, attr))
		exp = sorted(getattr(expected, attr))
		if actual != exp:
			errors.append(f"  {attr}: expected {exp}, got {actual}")

	if errors:
		msg = f"FAIL [{name}]:\n" + "\n".join(errors)
		raise AssertionError(msg)


def test_poiseuille() -> None:
	"""Test WallDetector with Poiseuille pipe geometry.

	Geometry: 80x20 pipe with solid top (y=0) and bottom (y=19) rows.
	Velocity boundary at x=0, y=1..18 (LeftWall with velocity profile).
	Density boundary at x=79, y=1..18 (RightWall with density profile).

	Expected boundaries (matching poiseuille.py manual setup):
	- LeftWall at x=0, y=1..18 with velocity_profile (not None)
	- RightWall at x=79, y=1..18 with density_profile (not None)
	- TopWall at y=0, x=1..78 (corners excluded)
	- BottomWall at y=19, x=1..78 (corners excluded)
	- 4 ConcaveCorners at (0,0) TL, (0,79) TR, (19,0) BL, (19,79) BR
	"""
	height, width = 20, 80

	# Pipe geometry: top and bottom rows solid
	geometry = np.zeros((height, width), dtype=int)
	geometry[0, :] = 1
	geometry[-1, :] = 1

	# Velocity boundary at x=0, y=1..18
	velocity_geometry = np.zeros((height, width), dtype=int)
	velocity_geometry[1:height - 1, 0] = 1

	# Density boundary at x=79, y=1..18
	density_geometry = np.zeros((height, width), dtype=int)
	density_geometry[1:height - 1, -1] = 1

	# Plain geometry: solid geometry minus velocity/density cells
	plain_geometry = geometry & ~velocity_geometry & ~density_geometry

	# --- Velocity detector ---
	vel_detector = WallDetector()
	vel_detector.detect(velocity_geometry, other_boundaries=plain_geometry | density_geometry)

	assert_detection(vel_detector, ExpectedDetection(
		left_walls=[(0, 1, 18)],
	), "velocity detector")

	# Velocity walls should carry a velocity profile (not None) in ZouHe.setup
	assert len(vel_detector.left_walls) > 0, "velocity detector must find left wall"
	assert len(vel_detector.right_walls) == 0, "velocity detector must not find phantom right wall"

	# --- Density detector ---
	dens_detector = WallDetector()
	dens_detector.detect(density_geometry, other_boundaries=plain_geometry | velocity_geometry)

	assert_detection(dens_detector, ExpectedDetection(
		right_walls=[(79, 1, 18)],
	), "density detector")

	# Density walls should carry a density profile (not None) in ZouHe.setup
	assert len(dens_detector.right_walls) > 0, "density detector must find right wall"
	assert len(dens_detector.left_walls) == 0, "density detector must not find phantom left wall"

	# --- Plain detector ---
	plain_detector = WallDetector()
	plain_detector.detect(plain_geometry, other_boundaries=velocity_geometry | density_geometry)

	assert_detection(plain_detector, ExpectedDetection(
		top_walls=[(0, 1, 78)],
		bot_walls=[(19, 1, 78)],
		conc_tl=[(0, 0)],
		conc_tr=[(0, 79)],
		conc_bl=[(19, 0)],
		conc_br=[(19, 79)],
	), "plain detector")

	# Plain walls should have no profiles (both None) in ZouHe.setup
	assert len(plain_detector.left_walls) == 0, "plain detector must not find left walls"
	assert len(plain_detector.right_walls) == 0, "plain detector must not find right walls"


if __name__ == "__main__":
	test_poiseuille()
