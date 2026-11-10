"""
NETRA-UGV Risk-Traversability Cost Function
=============================================
Computes the navigation cost C(x,y) for each cell on the 2.5D elevation grid.

Formula (TECHNICAL_ARCHITECTURE.md §5.4):
  C(x,y) = clamp(w1*‖∇Z(x,y)‖ + w2*σ²_Z(x,y) + w3*C_semantic(x,y) + w4*H_void(x,y), 0, 255)

Where:
  w1: Slope gradient weight     — prevents vehicle rollover on steep inclines
  w2: Roughness (variance) wt   — penalizes jagged rocky terrain
  w3: Semantic class cost        — SOLID: 0, PLIANT: 35, MUD: 75, RIGID: 255
  w4: Void (ditch) cost          — 255 (lethal) if Bayesian-confirmed
"""

from typing import Dict

import numpy as np

# Semantic class cost mapping (matches TerrainClassification.msg constants)
SEMANTIC_COSTS: Dict[int, int] = {
    0: 0,  # SOLID_GROUND:      Free passage
    1: 35,  # PLIANT_VEGETATION: Speed-governed traversal
    2: 75,  # MUD_HAZARD:        High traction risk
    3: 255,  # RIGID_OBSTACLE:    Lethal barrier
}


class CostFunction:
    """
    Multi-factor traversability cost computation for the elevation costmap.
    """

    def __init__(
        self,
        w_slope: float = 80.0,
        w_roughness: float = 60.0,
        w_semantic: float = 1.0,
        w_void: float = 255.0,
        resolution: float = 0.1,
    ):
        """
        Args:
            w_slope:     Weight for terrain slope gradient.
            w_roughness: Weight for surface roughness (height variance).
            w_semantic:  Weight for semantic class cost.
            w_void:      Weight for confirmed negative obstacle (void).
            resolution:  Grid cell resolution in meters.
        """
        self.w1 = w_slope
        self.w2 = w_roughness
        self.w3 = w_semantic
        self.w4 = w_void
        self.res = resolution

    def compute(
        self,
        elevation: np.ndarray,
        variance: np.ndarray,
        semantic: np.ndarray,
        void_flag: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the full costmap from elevation grid layers.

        Args:
            elevation: 2D array of mean cell heights (float32).
            variance:  2D array of height variance per cell (float32).
            semantic:  2D array of terrain class indices (uint8, 0-3).
            void_flag: 2D boolean array of confirmed negative obstacles.

        Returns:
            2D uint8 costmap of shape (N, N), values in [0, 255].
        """

        # --- Component 1: Slope gradient ‖∇Z(x,y)‖ ---
        # Sobel gradients for dZ/dx and dZ/dy
        dz_dx = np.zeros_like(elevation)
        dz_dy = np.zeros_like(elevation)

        dz_dx[:, 1:-1] = (elevation[:, 2:] - elevation[:, :-2]) / (2.0 * self.res)
        dz_dy[1:-1, :] = (elevation[2:, :] - elevation[:-2, :]) / (2.0 * self.res)

        slope_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)

        # --- Component 2: Roughness (local height variance) ---
        # Use the Welford variance already computed in elevation_grid
        roughness = variance

        # --- Component 3: Semantic class cost ---
        semantic_cost = np.zeros_like(elevation, dtype=np.float32)
        for cls, cost in SEMANTIC_COSTS.items():
            semantic_cost[semantic == cls] = cost

        # --- Component 4: Void (confirmed negative obstacle) ---
        void_cost = void_flag.astype(np.float32) * 255.0

        # --- Combined cost ---
        raw_cost = (
            self.w1 * slope_magnitude
            + self.w2 * roughness
            + self.w3 * semantic_cost
            + self.w4 * (void_cost / 255.0)  # Normalize to w4 scale
        )

        # Clamp to [0, 255] and convert to uint8
        costmap = np.clip(raw_cost, 0, 255).astype(np.uint8)
        return costmap
