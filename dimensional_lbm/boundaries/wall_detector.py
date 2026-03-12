from pathlib import Path

import numpy as np
import pint
from PIL import Image


class WallDetector:
	"""Detects boundary types (walls, corners) in a solid/fluid array.

	Solid cells are 1, fluid cells are 0. Boundaries are solid cells adjacent to fluid.
	Uses periodic boundary conditions for walls intersecting image edges, except
	at image corners where both horizontal and vertical walls meet.

	Walls and convex corners are named by position: wall_l / left_walls are walls
	on the LEFT side (fluid to the right), conv_tl is a convex corner at the
	top-left (fluid to the bottom-right), etc. Concave corners are also named by
	position (the direction where the two walls meet).
	"""

	def __init__(self) -> None:
		self._height = 0
		self._width = 0
		# Corner positions as (2, N) arrays: [y_coords; x_coords]
		self.conc_bl = np.empty((2, 0), dtype=int)
		self.conc_br = np.empty((2, 0), dtype=int)
		self.conc_tl = np.empty((2, 0), dtype=int)
		self.conc_tr = np.empty((2, 0), dtype=int)
		self.conv_bl = np.empty((2, 0), dtype=int)
		self.conv_br = np.empty((2, 0), dtype=int)
		self.conv_tl = np.empty((2, 0), dtype=int)
		self.conv_tr = np.empty((2, 0), dtype=int)
		# Wall positions
		self.wall_l = np.empty((2, 0), dtype=int)
		self.wall_r = np.empty((2, 0), dtype=int)
		self.wall_t = np.empty((2, 0), dtype=int)
		self.wall_b = np.empty((2, 0), dtype=int)
		# Segmented walls: list of (static_coord, start, end)
		self.left_walls: list[tuple[int, int, int]] = []
		self.right_walls: list[tuple[int, int, int]] = []
		self.top_walls: list[tuple[int, int, int]] = []
		self.bot_walls: list[tuple[int, int, int]] = []

	def _get_neighbors(self, solid: np.ndarray) -> dict:
		"""Get shifted arrays for each direction.

		Cardinal directions: periodic (wrap around) so image edges don't create
		phantom boundaries - only real solid/fluid interfaces are detected.
		Diagonals: pad with 1 (solid) so corners at edges are NOT detected
		(they should be straight walls due to periodic assumption).
		"""
		# Cardinal: periodic wrapping
		left = np.roll(solid, 1, axis=1)
		right = np.roll(solid, -1, axis=1)
		top = np.roll(solid, 1, axis=0)
		bot = np.roll(solid, -1, axis=0)

		# Diagonals: pad with 1 (solid) to suppress corner detection at image edges
		tl = np.ones_like(solid)
		tl[1:, 1:] = solid[:-1, :-1]

		tr = np.ones_like(solid)
		tr[1:, :-1] = solid[:-1, 1:]

		bl = np.ones_like(solid)
		bl[:-1, 1:] = solid[1:, :-1]

		br = np.ones_like(solid)
		br[:-1, :-1] = solid[1:, 1:]

		return {"l": left, "r": right, "t": top, "b": bot, "tl": tl, "tr": tr, "bl": bl, "br": br}

	def detect(self, solid: np.ndarray, other_boundaries: np.ndarray | None = None) -> None:
		"""Detect walls and corners in a solid/fluid array.

		Args:
			solid: The boundary cells to detect (1=boundary, 0=fluid).
			other_boundaries: Optional mask of cells belonging to other boundary
				types. These are treated as solid for neighbor lookups (preventing
				false corners at junctions between boundary types) but are not
				themselves detected as walls or corners.
		"""
		if isinstance(solid, pint.Quantity):
			solid = solid.magnitude

		solid = solid.astype(np.int32)
		self._height, self._width = solid.shape

		if other_boundaries is not None:
			if isinstance(other_boundaries, pint.Quantity):
				other_boundaries = other_boundaries.magnitude
			other_boundaries = other_boundaries.astype(np.int32)
			combined = solid | other_boundaries
		else:
			combined = solid

		n = self._get_neighbors(combined)

		# Fluid neighbors (where there's fluid adjacent to this cell)
		fluid_l = solid & ~n["l"]  # solid here, fluid to left
		fluid_r = solid & ~n["r"]
		fluid_t = solid & ~n["t"]
		fluid_b = solid & ~n["b"]

		# Convex corners: solid with fluid in two adjacent cardinal directions
		# and fluid diagonally between them
		_conv_tl = solid & ~n["t"] & ~n["l"] & ~n["tl"]
		_conv_tr = solid & ~n["t"] & ~n["r"] & ~n["tr"]
		_conv_bl = solid & ~n["b"] & ~n["l"] & ~n["bl"]
		_conv_br = solid & ~n["b"] & ~n["r"] & ~n["br"]

		# Concave corners: solid with solid in two adjacent cardinal directions
		# but fluid diagonally between them
		_conc_tl = solid & n["t"] & n["l"] & ~n["tl"]
		_conc_tr = solid & n["t"] & n["r"] & ~n["tr"]
		_conc_bl = solid & n["b"] & n["l"] & ~n["bl"]
		_conc_br = solid & n["b"] & n["r"] & ~n["br"]

		# Store corner positions
		# Convex corners named by position (opposite of fluid direction):
		# conv_tl = corner at top-left (fluid is to bottom-right)
		self.conv_tl = np.vstack(np.where(_conv_br))
		self.conv_tr = np.vstack(np.where(_conv_bl))
		self.conv_bl = np.vstack(np.where(_conv_tr))
		self.conv_br = np.vstack(np.where(_conv_tl))
		# Concave corners named by position (opposite of fluid direction):
		# conc_tl = corner at top-left (fluid pocket at bottom-right diagonal)
		self.conc_tl = np.vstack(np.where(_conc_br))
		self.conc_tr = np.vstack(np.where(_conc_bl))
		self.conc_bl = np.vstack(np.where(_conc_tr))
		self.conc_br = np.vstack(np.where(_conc_tl))

		# All corners mask
		corners = _conv_tl | _conv_tr | _conv_bl | _conv_br
		corners |= _conc_tl | _conc_tr | _conc_bl | _conc_br

		# Straight walls: fluid in one direction, not a corner
		_fluid_to_l = fluid_l & ~corners
		_fluid_to_r = fluid_r & ~corners
		_fluid_to_t = fluid_t & ~corners
		_fluid_to_b = fluid_b & ~corners

		# Walls named by position: wall_l = wall on the left side (fluid to right)
		self.wall_l = np.vstack(np.where(_fluid_to_r))
		self.wall_r = np.vstack(np.where(_fluid_to_l))
		self.wall_t = np.vstack(np.where(_fluid_to_b))
		self.wall_b = np.vstack(np.where(_fluid_to_t))

		# Segment walls with periodic handling
		self.left_walls = self._segment_walls_periodic(self.wall_l, "y", self._height)
		self.right_walls = self._segment_walls_periodic(self.wall_r, "y", self._height)
		self.top_walls = self._segment_walls_periodic(self.wall_t, "x", self._width)
		self.bot_walls = self._segment_walls_periodic(self.wall_b, "x", self._width)

	def _segment_walls_periodic(self, wall: np.ndarray, direction: str, max_coord: int) -> list[tuple[int, int, int]]:
		"""Group wall pixels into contiguous segments with periodic wrapping."""
		if wall.size == 0:
			return []

		if direction == "x":
			static_ax, move_ax = 0, 1
		else:
			static_ax, move_ax = 1, 0

		# Sort by static axis, then by moving axis
		order = np.lexsort((wall[move_ax], wall[static_ax]))
		sw = wall[:, order]

		segments = []
		curr = 0
		n = sw.shape[1]

		for i in range(n - 1):
			same_line = sw[static_ax, i] == sw[static_ax, i + 1]
			contiguous = abs(sw[move_ax, i] - sw[move_ax, i + 1]) <= 1

			if not (same_line and contiguous):
				segments.append((int(sw[static_ax, curr]), int(sw[move_ax, curr]), int(sw[move_ax, i])))
				curr = i + 1

		# Add final segment
		segments.append((int(sw[static_ax, curr]), int(sw[move_ax, curr]), int(sw[move_ax, n - 1])))

		# Merge periodic segments (walls that wrap around)
		return self._merge_periodic_segments(segments, max_coord, direction)

	def _merge_periodic_segments(self, segments: list[tuple[int, int, int]], max_coord: int, direction: str) -> list[tuple[int, int, int]]:
		"""Merge segments that touch opposite edges (periodic boundary)."""
		if len(segments) < 2:
			return segments

		# Group by static coordinate
		by_static = {}
		for seg in segments:
			static, start, end = seg
			if static not in by_static:
				by_static[static] = []
			by_static[static].append((start, end))

		# Check for corners at image edges that should prevent merging
		corner_positions = self._get_edge_corner_positions(direction)

		merged = []
		for static, segs in by_static.items():
			if len(segs) >= 2:
				# Check if there's a segment at start (0) and end (max_coord-1)
				at_start = [s for s in segs if s[0] == 0]
				at_end = [s for s in segs if s[1] == max_coord - 1]

				if at_start and at_end and not self._has_corner_at_edge(static, corner_positions):
					# Merge: segment from at_end[0] wraps to at_start[0]
					start_seg = at_start[0]
					end_seg = at_end[0]

					# Remove both from segs
					segs = [s for s in segs if s not in (start_seg, end_seg)]
					# Add merged segment (end_seg.start to start_seg.end, wrapping)
					segs.append((end_seg[0], start_seg[1]))

			for start, end in segs:
				merged.append((static, start, end))

		return merged

	def _get_edge_corner_positions(self, direction: str) -> set:
		"""Get static coordinates where corners exist at image edges."""
		positions = set()
		h, w = self._height, self._width

		if direction == "x":  # horizontal walls, static is y
			# Check for corners at left (x=0) or right (x=w-1) edges
			for corners in [self.conv_tl, self.conv_bl, self.conc_tl, self.conc_bl]:
				if corners.size > 0:
					mask = corners[1] == 0  # x == 0
					positions.update(corners[0, mask].tolist())
			for corners in [self.conv_tr, self.conv_br, self.conc_tr, self.conc_br]:
				if corners.size > 0:
					mask = corners[1] == w - 1  # x == w-1
					positions.update(corners[0, mask].tolist())
		else:  # vertical walls, static is x
			# Check for corners at top (y=0) or bottom (y=h-1) edges
			for corners in [self.conv_tl, self.conv_tr, self.conc_tl, self.conc_tr]:
				if corners.size > 0:
					mask = corners[0] == 0  # y == 0
					positions.update(corners[1, mask].tolist())
			for corners in [self.conv_bl, self.conv_br, self.conc_bl, self.conc_br]:
				if corners.size > 0:
					mask = corners[0] == h - 1  # y == h-1
					positions.update(corners[1, mask].tolist())

		return positions

	def _has_corner_at_edge(self, static_coord: int, corner_positions: set) -> bool:
		"""Check if there's a corner at the edge for this wall line."""
		return static_coord in corner_positions

	def plot(self, output_path: Path) -> None:
		"""Save visualization with different colors for each boundary type."""
		rgb = np.zeros((self._height, self._width, 3), dtype=np.uint8)

		colors = {
			"wall_l": (255, 0, 0),  # red
			"wall_r": (0, 255, 0),  # green
			"wall_t": (0, 0, 255),  # blue
			"wall_b": (255, 255, 0),  # yellow
			"conc_bl": (255, 0, 255),  # magenta
			"conc_br": (255, 128, 0),  # orange
			"conc_tl": (0, 255, 255),  # cyan
			"conc_tr": (128, 0, 255),  # purple
			"conv_bl": (255, 128, 128),  # light red
			"conv_br": (128, 255, 128),  # light green
			"conv_tl": (128, 128, 255),  # light blue
			"conv_tr": (255, 255, 128),  # light yellow
		}

		for name, color in colors.items():
			arr = getattr(self, name)
			if arr is not None and arr.size > 0:
				rgb[arr[0], arr[1]] = color

		img = Image.fromarray(rgb)
		img.save(output_path)


def img_to_solid(image_path: Path) -> np.ndarray:
	"""Load B/W image and convert to solid array (black=1, white=0)."""
	img = Image.open(image_path).convert("L")
	img_array = np.array(img)
	return np.where(img_array == 0, 1, 0)
