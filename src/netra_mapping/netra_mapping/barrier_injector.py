"""
NETRA-UGV Virtual Barrier Injector
=====================================
Projects Bayesian-confirmed negative obstacle (ditch) lip coordinates
from camera frame into the costmap grid and sets corresponding cells
to lethal cost (255) with a configurable inflation radius.

Reference: MASTER_PROJECT_REPORT.md §3.2
  Virtual barrier committed when L_t(x,y) >= 3.5
  Barrier placed at ditch front lip coordinates
"""

from typing import List

import numpy as np

from netra_mapping.elevation_grid import ElevationGrid


class BarrierInjector:
    """
    Injects lethal virtual barriers at confirmed negative obstacle locations.
    """

    def __init__(
        self,
        inflation_radius_m: float = 0.3,
        resolution: float = 0.1,
    ):
        """
        Args:
            inflation_radius_m: Radius around the ditch lip to mark as lethal (m).
            resolution:         Costmap cell resolution (m).
        """
        self.inflation_cells = max(1, int(inflation_radius_m / resolution))
        self.resolution = resolution

    def inject_barriers(
        self,
        grid: ElevationGrid,
        obstacle_points: List[dict],
    ):
        """
        Mark costmap cells as voids at confirmed negative obstacle locations.

        Args:
            grid:             The ElevationGrid to modify.
            obstacle_points:  List of dicts with 'world_x' and 'world_y' keys
                              (from NegativeObstacleVoid messages).
        """
        for obs in obstacle_points:
            cx = obs.get("world_x", obs.get("x", 0.0))
            cy = obs.get("world_y", obs.get("y", 0.0))

            # Set void flag on the center cell
            grid.set_void(cx, cy, True)

            # Inflate around the lip coordinates
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                    if dx * dx + dy * dy <= self.inflation_cells * self.inflation_cells:
                        wx = cx + dx * self.resolution
                        wy = cy + dy * self.resolution
                        grid.set_void(wx, wy, True)

    def inject_from_msg(
        self,
        grid: ElevationGrid,
        edge_start_x: float,
        edge_start_y: float,
        edge_end_x: float,
        edge_end_y: float,
    ):
        """
        Inject a barrier line segment between two ditch lip endpoints.

        Interpolates points along the segment and marks all cells as lethal.

        Args:
            grid:          ElevationGrid instance.
            edge_start_*:  Start point of ditch lip (meters, base_link frame).
            edge_end_*:    End point of ditch lip (meters, base_link frame).
        """
        length = np.sqrt((edge_end_x - edge_start_x) ** 2 + (edge_end_y - edge_start_y) ** 2)
        n_steps = max(2, int(length / self.resolution) + 1)

        for t in np.linspace(0.0, 1.0, n_steps):
            wx = edge_start_x + t * (edge_end_x - edge_start_x)
            wy = edge_start_y + t * (edge_end_y - edge_start_y)
            grid.set_void(wx, wy, True)

            # Inflate
            for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                    if dx * dx + dy * dy <= self.inflation_cells * self.inflation_cells:
                        grid.set_void(
                            wx + dx * self.resolution,
                            wy + dy * self.resolution,
                            True,
                        )
