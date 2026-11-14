"""
NETRA-UGV Bayesian Temporal Confirmation Filter
=================================================
Suppresses false positive negative obstacle detections caused by
camera vibration, gravel kick-up, or momentary disparity noise.

A virtual barrier is committed to the costmap ONLY when a depression
is geometrically verified across ≥3 consecutive frames at 50 Hz
(≈60 ms temporal persistence).

Reference: MASTER_PROJECT_REPORT.md §3.2, TECHNICAL_ARCHITECTURE.md §5.3

Log-odds update rule:
  L_t(x,y) = L_{t-1}(x,y) + ln[P(Void|D_t) / (1 - P(Void|D_t))]
                            - ln[P(Void) / (1 - P(Void))]

Barrier committed when: L_t(x,y) >= L_threshold = 3.5
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BayesianVoidFilter:
    """
    Temporal Bayesian log-odds accumulator for negative obstacle confirmation.

    Maintains a 2D grid of log-odds values representing the probability
    that each cell contains a ditch/void. Only cells exceeding the
    commitment threshold are injected as lethal barriers into the costmap.
    """

    # Default configuration
    DEFAULT_GRID_SIZE = 100  # 100x100 cells
    DEFAULT_RESOLUTION = 0.1  # 0.1 m per cell → 10m x 10m coverage
    DEFAULT_THRESHOLD = 3.5  # Log-odds threshold for commitment
    DEFAULT_PRIOR = 0.01  # Prior probability of void P(Void)
    DEFAULT_DETECTION_PROB = 0.85  # P(Void | D_t) when detector fires
    DEFAULT_FALSE_ALARM_PROB = 0.05  # P(Void | D_t) when detector does NOT fire
    DEFAULT_DECAY_RATE = 0.1  # Log-odds decay per frame (forgetting factor)
    MIN_CONSECUTIVE_HITS = 3  # Minimum consecutive frame detections

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        resolution: float = DEFAULT_RESOLUTION,
        log_odds_threshold: float = DEFAULT_THRESHOLD,
        prior_void_prob: float = DEFAULT_PRIOR,
        detection_prob: float = DEFAULT_DETECTION_PROB,
        false_alarm_prob: float = DEFAULT_FALSE_ALARM_PROB,
        decay_rate: float = DEFAULT_DECAY_RATE,
    ):
        """
        Args:
            grid_size:          Number of cells per dimension (NxN grid).
            resolution:         Cell size in meters.
            log_odds_threshold: Log-odds value to commit a barrier (L_threshold).
            prior_void_prob:    Prior probability P(Void) of any cell being a void.
            detection_prob:     P(Void | D_t) when detector reports a void.
            false_alarm_prob:   P(Void | D_t) when detector does NOT report void.
            decay_rate:         Log-odds decay per frame for unobserved cells.
        """
        self.grid_size = grid_size
        self.resolution = resolution
        self.threshold = log_odds_threshold
        self.decay_rate = decay_rate

        # Pre-compute log-odds constants
        self.l_prior = self._prob_to_log_odds(prior_void_prob)
        self.l_detection = self._prob_to_log_odds(detection_prob)
        self.l_free = self._prob_to_log_odds(false_alarm_prob)

        # Log-odds grid (initialized to prior)
        self.log_odds_grid = np.full((grid_size, grid_size), self.l_prior, dtype=np.float32)

        # Consecutive hit counter per cell
        self.consecutive_hits = np.zeros((grid_size, grid_size), dtype=np.int32)

        # Grid origin in world frame (updated by odometry)
        self.origin_x = -grid_size * resolution / 2.0
        self.origin_y = -grid_size * resolution / 2.0

    @staticmethod
    def _prob_to_log_odds(p: float) -> float:
        """Convert probability to log-odds: L = ln(p / (1-p))."""
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _log_odds_to_prob(log_odds: float) -> float:
        """Convert log-odds to probability: p = 1 / (1 + exp(-L))."""
        return float(1.0 / (1.0 + np.exp(-log_odds)))

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """
        Convert world coordinates (meters) to grid cell indices.

        Args:
            x, y: World coordinates in meters (in vehicle base_link frame).

        Returns:
            (row, col) grid indices, or None if outside grid bounds.
        """
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            return (row, col)
        return None

    def update(
        self,
        detections: List[dict],
        vehicle_x: float = 0.0,
        vehicle_y: float = 0.0,
    ) -> List[dict]:
        """
        Update the Bayesian grid with new detection results.

        For cells with detections:
          L_t = L_{t-1} + l_detection - l_prior

        For cells without detections (observed but clear):
          L_t = L_{t-1} + l_free - l_prior

        Unobserved cells decay toward the prior.

        Args:
            detections:  List of detection dicts from VDisparityDetector.
            vehicle_x:   Current vehicle X position (meters).
            vehicle_y:   Current vehicle Y position (meters).

        Returns:
            List of confirmed barrier dicts (cells exceeding threshold).
        """
        # Create observation mask (which cells were observed this frame)
        observed = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        detected = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        for det in detections:
            # Project detection depth to world coordinates
            # Detection is in front of the vehicle at det['depth_m'] distance
            world_x = vehicle_x + det["depth_m"]
            # Spread across the column range (approximate lateral extent)
            col_range = det["col_end"] - det["col_start"]
            lateral_extent = det["depth_m"] * col_range / 1000.0  # approximate

            for dy_offset in np.linspace(-lateral_extent / 2, lateral_extent / 2, 5):
                world_y = vehicle_y + dy_offset
                cell = self.world_to_grid(world_x, world_y)
                if cell is not None:
                    detected[cell] = True
                    observed[cell] = True

        # Mark forward observation cone as observed (even if no detection)
        for dx in np.arange(0.5, 4.0, self.resolution):
            for dy in np.arange(-2.0, 2.0, self.resolution):
                cell = self.world_to_grid(vehicle_x + dx, vehicle_y + dy)
                if cell is not None:
                    observed[cell] = True

        # --- Bayesian Update ---
        # Cells with void detection: increase log-odds
        self.log_odds_grid[detected] += self.l_detection - self.l_prior
        self.consecutive_hits[detected] += 1

        # Observed cells without detection: decrease log-odds
        clear = observed & ~detected
        self.log_odds_grid[clear] += self.l_free - self.l_prior
        self.consecutive_hits[clear] = 0

        # Unobserved cells: decay toward prior
        unobserved = ~observed
        self.log_odds_grid[unobserved] -= self.decay_rate
        self.log_odds_grid[unobserved] = np.maximum(self.log_odds_grid[unobserved], self.l_prior)

        # Clamp log-odds to prevent numerical overflow
        self.log_odds_grid = np.clip(self.log_odds_grid, -10.0, 15.0)

        # --- Extract Confirmed Barriers ---
        confirmed = []
        committed = (self.log_odds_grid >= self.threshold) & (self.consecutive_hits >= self.MIN_CONSECUTIVE_HITS)

        barrier_cells = np.argwhere(committed)
        for row, col in barrier_cells:
            world_x = self.origin_x + col * self.resolution
            world_y = self.origin_y + row * self.resolution
            confirmed.append(
                {
                    "grid_row": int(row),
                    "grid_col": int(col),
                    "world_x": world_x,
                    "world_y": world_y,
                    "log_odds": float(self.log_odds_grid[row, col]),
                    "probability": self._log_odds_to_prob(self.log_odds_grid[row, col]),
                    "consecutive_hits": int(self.consecutive_hits[row, col]),
                }
            )

        return confirmed

    def reset(self):
        """Reset the entire grid to prior values."""
        self.log_odds_grid[:] = self.l_prior
        self.consecutive_hits[:] = 0

    def get_probability_grid(self) -> np.ndarray:
        """Return the current probability grid for visualization."""
        return 1.0 / (1.0 + np.exp(-self.log_odds_grid))
