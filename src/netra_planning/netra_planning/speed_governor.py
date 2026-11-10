"""
NETRA-UGV Adaptive Speed Governor
==================================
Dynamically clamps maximum allowable forward velocity (v_max) and yaw rate
based on terrain semantics (tall grass brush, mud), slope steepness, elevation
roughness, and multi-tier failsafe operational modes.

Key Rules:
  1. Brush Zone: > 30% PLIANT_VEGETATION in forward corridor -> v_max <= 0.5 m/s
  2. Mud Hazard: > 20% MUD_HAZARD -> v_max <= 0.4 m/s (prevent wheel trenching)
  3. Steep Incline: slope > 12 deg -> proportional speed reduction
  4. Vision Degraded (Level 1): v_max <= 0.8 m/s
  5. Vision Critical (Level 2): controlled deceleration to halt (0.0 m/s)
  6. Security Tamper (Level 3): immediate hard zero (0.0 m/s)

Reference:
  MASTER_PROJECT_REPORT.md §4 & TECHNICAL_ARCHITECTURE.md §5.4
"""

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Terrain Semantic Class Constants (aligned with netra_msgs/TerrainClassification)
CLASS_SOLID_GROUND = 0
CLASS_PLIANT_VEGETATION = 1
CLASS_MUD_HAZARD = 2
CLASS_RIGID_OBSTACLE = 3

# Failsafe Mode Constants (aligned with netra_msgs/FailsafeState)
MODE_NOMINAL = 0
MODE_VISION_DEGRADED = 1
MODE_VISION_CRITICAL = 2
MODE_SECURITY_TAMPER = 3


class SpeedGovernor:
    """
    Adaptive velocity governor enforcing terrain-aware speed constraints.
    """

    def __init__(
        self,
        v_nominal_max: float = 1.5,
        v_brush_cap: float = 0.5,
        v_mud_cap: float = 0.4,
        v_degraded_cap: float = 0.8,
        slope_threshold_rad: float = 0.20,  # ~11.5 degrees
        slope_critical_rad: float = 0.35,  # ~20.0 degrees
        roughness_threshold: float = 0.04,  # elevation variance m^2
        deceleration_limit: float = 1.0,  # m/s^2 controlled deceleration
    ):
        self.v_nominal_max = v_nominal_max
        self.v_brush_cap = v_brush_cap
        self.v_mud_cap = v_mud_cap
        self.v_degraded_cap = v_degraded_cap
        self.slope_threshold_rad = slope_threshold_rad
        self.slope_critical_rad = slope_critical_rad
        self.roughness_threshold = roughness_threshold
        self.deceleration_limit = deceleration_limit

        self._current_clamped_v_max = v_nominal_max
        self._governor_reason = "Nominal"

    @property
    def current_v_max(self) -> float:
        """Currently active dynamic velocity ceiling."""
        return self._current_clamped_v_max

    @property
    def governor_reason(self) -> str:
        """Human-readable diagnostic reason for current velocity restriction."""
        return self._governor_reason

    def compute_governed_velocity(
        self,
        cmd_v: float,
        cmd_omega: float,
        failsafe_mode: int = MODE_NOMINAL,
        vegetation_ratio: float = 0.0,
        mud_ratio: float = 0.0,
        slope_rad: float = 0.0,
        roughness: float = 0.0,
        dt: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Evaluate all environmental and system health constraints to produce
        safely governed (v, omega) command outputs.

        Args:
            cmd_v:            Requested linear velocity (m/s)
            cmd_omega:        Requested angular velocity (rad/s)
            failsafe_mode:    Active failsafe level (0, 1, 2, 3)
            vegetation_ratio: Fraction of forward path covered by pliant vegetation (0.0 - 1.0)
            mud_ratio:        Fraction of forward path covered by mud (0.0 - 1.0)
            slope_rad:        Estimated ground inclination in radians
            roughness:        Elevation variance along the corridor
            dt:               Time elapsed since last cycle

        Returns:
            Tuple of:
              - governed_v:     Safe forward velocity
              - governed_omega: Safe angular velocity
              - v_max_allowed:  Dynamic upper limit applied this cycle
        """
        active_cap = self.v_nominal_max
        reasons = []

        # 1. Evaluate Failsafe Operational Mode
        if failsafe_mode == MODE_SECURITY_TAMPER:
            self._current_clamped_v_max = 0.0
            self._governor_reason = "CRITICAL SENSOR BLACKOUT (Level 3)"
            return 0.0, 0.0, 0.0

        elif failsafe_mode == MODE_VISION_CRITICAL:
            # Controlled deceleration to full stop
            decel_step = self.deceleration_limit * dt
            active_cap = max(0.0, self._current_clamped_v_max - decel_step)
            reasons.append("Vision Critical (Level 2 Limp-to-Halt)")

        elif failsafe_mode == MODE_VISION_DEGRADED:
            active_cap = min(active_cap, self.v_degraded_cap)
            reasons.append("Vision Degraded (Level 1)")

        # 2. Evaluate Semantic Terrain Constraints (if vision available)
        if failsafe_mode in (MODE_NOMINAL, MODE_VISION_DEGRADED):
            # Tall Brush / Pliant Vegetation Clamping
            if vegetation_ratio > 0.30:
                brush_v = self.v_brush_cap
                if vegetation_ratio > 0.60:
                    brush_v = max(0.3, self.v_brush_cap * 0.7)
                active_cap = min(active_cap, brush_v)
                reasons.append(f"Brush Zone ({vegetation_ratio*100:.0f}%)")

            # Mud Hazard Clamping
            if mud_ratio > 0.20:
                active_cap = min(active_cap, self.v_mud_cap)
                # Also clamp angular rate in mud to avoid turning-induced trenching
                cmd_omega = float(np.clip(cmd_omega, -0.8, 0.8))
                reasons.append(f"Mud Hazard ({mud_ratio*100:.0f}%)")

            # 3. Ground Incline Slope Clamping
            if slope_rad > self.slope_threshold_rad:
                if slope_rad >= self.slope_critical_rad:
                    slope_factor = 0.25
                else:
                    # Linear decay between threshold and critical
                    span = self.slope_critical_rad - self.slope_threshold_rad
                    slope_factor = 1.0 - 0.75 * ((slope_rad - self.slope_threshold_rad) / span)
                active_cap = min(active_cap, self.v_nominal_max * slope_factor)
                reasons.append(f"Slope Incline ({np.degrees(slope_rad):.1f} deg)")

            # 4. Roughness / Rocky Field Clamping
            if roughness > self.roughness_threshold:
                active_cap = min(active_cap, 0.6)
                reasons.append(f"Rough Terrain (var={roughness:.3f})")

        # Record active ceiling and reason
        self._current_clamped_v_max = active_cap
        self._governor_reason = ", ".join(reasons) if reasons else "Nominal Operation"

        # Apply clamping to requested velocity
        governed_v = float(np.clip(cmd_v, -0.5, active_cap))
        governed_omega = float(cmd_omega)

        return governed_v, governed_omega, active_cap
