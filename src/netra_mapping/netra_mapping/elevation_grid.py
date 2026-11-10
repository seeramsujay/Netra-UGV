"""
NETRA-UGV 2.5D Rolling Elevation Grid
========================================
Vehicle-centric rolling 2.5D elevation grid that stores per-cell:
  - Mean elevation Z
  - Height variance σ²_Z (roughness indicator)
  - Semantic terrain class
  - Void flag (Bayesian-confirmed negative obstacle)

Grid scrolls with vehicle motion using odometry updates.

Reference: TECHNICAL_ARCHITECTURE.md §5.4
  10m × 10m area, 0.1m cell resolution → 100×100 grid
  Update rate: 10 Hz
"""

from typing import Optional, Tuple

import numpy as np


class ElevationGrid:
    """
    Rolling 2.5D elevation grid centered on the vehicle.
    """

    def __init__(
        self,
        grid_size: int = 100,
        resolution: float = 0.1,
    ):
        """
        Args:
            grid_size:  Number of cells per dimension (NxN).
            resolution: Cell size in meters.
        """
        self.N = grid_size
        self.res = resolution
        self.half_extent = (grid_size * resolution) / 2.0

        # Grid data layers
        self.elevation = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.variance = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.semantic = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.void_flag = np.zeros((grid_size, grid_size), dtype=bool)
        self.hit_count = np.zeros((grid_size, grid_size), dtype=np.int32)

        # Grid origin in world frame (bottom-left corner)
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Last vehicle position for scroll computation
        self._last_vx = 0.0
        self._last_vy = 0.0

    def world_to_grid(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to grid indices."""
        col = int((wx - self.origin_x) / self.res)
        row = int((wy - self.origin_y) / self.res)
        if 0 <= row < self.N and 0 <= col < self.N:
            return (row, col)
        return None

    def grid_to_world(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        wx = self.origin_x + (col + 0.5) * self.res
        wy = self.origin_y + (row + 0.5) * self.res
        return (wx, wy)

    def update_vehicle_pose(self, vx: float, vy: float):
        """
        Scroll the grid to keep the vehicle centered.

        Shifts grid data when the vehicle moves more than one cell
        from the center.
        """
        # Desired origin (vehicle at center)
        desired_ox = vx - self.half_extent
        desired_oy = vy - self.half_extent

        # Compute shift in cells
        shift_x = int(round((desired_ox - self.origin_x) / self.res))
        shift_y = int(round((desired_oy - self.origin_y) / self.res))

        if abs(shift_x) >= 1 or abs(shift_y) >= 1:
            self._scroll_grid(shift_x, shift_y)
            self.origin_x += shift_x * self.res
            self.origin_y += shift_y * self.res

        self._last_vx = vx
        self._last_vy = vy

    def _scroll_grid(self, dx: int, dy: int):
        """Scroll all grid layers by (dx, dy) cells, clearing exposed regions."""
        for layer in [self.elevation, self.variance, self.hit_count]:
            layer[:] = np.roll(np.roll(layer, -dy, axis=0), -dx, axis=1)

        self.semantic[:] = np.roll(np.roll(self.semantic, -dy, axis=0), -dx, axis=1)
        self.void_flag[:] = np.roll(np.roll(self.void_flag, -dy, axis=0), -dx, axis=1)

        # Clear newly exposed cells
        if dx > 0:
            self.elevation[:, -dx:] = 0
            self.variance[:, -dx:] = 0
            self.semantic[:, -dx:] = 0
            self.void_flag[:, -dx:] = False
            self.hit_count[:, -dx:] = 0
        elif dx < 0:
            self.elevation[:, :(-dx)] = 0
            self.variance[:, :(-dx)] = 0
            self.semantic[:, :(-dx)] = 0
            self.void_flag[:, :(-dx)] = False
            self.hit_count[:, :(-dx)] = 0

        if dy > 0:
            self.elevation[-dy:, :] = 0
            self.variance[-dy:, :] = 0
            self.semantic[-dy:, :] = 0
            self.void_flag[-dy:, :] = False
            self.hit_count[-dy:, :] = 0
        elif dy < 0:
            self.elevation[:(-dy), :] = 0
            self.variance[:(-dy), :] = 0
            self.semantic[:(-dy), :] = 0
            self.void_flag[:(-dy), :] = False
            self.hit_count[:(-dy), :] = 0

    def update_cell(
        self,
        wx: float,
        wy: float,
        z_value: float,
        semantic_class: int = 0,
    ):
        """
        Update a single cell with a new elevation observation.

        Uses incremental mean and variance computation.
        """
        cell = self.world_to_grid(wx, wy)
        if cell is None:
            return

        r, c = cell
        n = self.hit_count[r, c]

        if n == 0:
            self.elevation[r, c] = z_value
            self.variance[r, c] = 0.0
        else:
            # Welford's online algorithm for mean and variance
            old_mean = self.elevation[r, c]
            new_mean = old_mean + (z_value - old_mean) / (n + 1)
            self.variance[r, c] += (z_value - old_mean) * (z_value - new_mean)
            self.elevation[r, c] = new_mean

        self.hit_count[r, c] = n + 1
        self.semantic[r, c] = semantic_class

    def set_void(self, wx: float, wy: float, is_void: bool = True):
        """Mark a cell as a confirmed negative obstacle (void)."""
        cell = self.world_to_grid(wx, wy)
        if cell is not None:
            self.void_flag[cell] = is_void

    def get_variance_grid(self) -> np.ndarray:
        """Return normalized variance (roughness) grid."""
        counts = np.maximum(self.hit_count, 1).astype(np.float32)
        return self.variance / counts
