"""
NETRA-UGV Hardware Safety Guards
=================================
Real-time physical protection guards protecting against:
  1. Tip-Over: Monitors pitch (> 22 deg) and roll (> 18 deg) angles.
  2. Suspension Shock: Monitors vertical z-axis acceleration (> 1.8g).

Returns:
  - GuardStatus: SAFE, WARNING, or EMERGENCY_STOP
  - Reason diagnostic string

Reference:
  MASTER_PROJECT_REPORT.md §3.3 & §8
"""

import logging
import math
from enum import IntEnum
from typing import Tuple

logger = logging.getLogger(__name__)


class GuardStatus(IntEnum):
    SAFE = 0
    WARNING = 1
    EMERGENCY_STOP = 2


class SafetyGuards:
    """
    Monitors IMU state to prevent vehicle overturn or structural impact damage.
    """

    def __init__(
        self,
        pitch_warn_deg: float = 17.0,
        pitch_crit_deg: float = 22.0,
        roll_warn_deg: float = 14.0,
        roll_crit_deg: float = 18.0,
        shock_threshold_g: float = 1.8,
        latch_duration_sec: float = 1.2,
    ):
        self.pitch_warn_rad = math.radians(pitch_warn_deg)
        self.pitch_crit_rad = math.radians(pitch_crit_deg)
        self.roll_warn_rad = math.radians(roll_warn_deg)
        self.roll_crit_rad = math.radians(roll_crit_deg)

        self.shock_threshold_ms2 = shock_threshold_g * 9.80665
        self.latch_duration_sec = latch_duration_sec

        self._latch_timer: float = 0.0
        self._latched_reason: str = ""

    def evaluate(
        self,
        roll_rad: float,
        pitch_rad: float,
        accel_z_ms2: float,
        dt: float = 0.05,
    ) -> Tuple[GuardStatus, str]:
        """
        Evaluate current attitude and dynamic shock.

        Args:
            roll_rad:     Vehicle roll in radians (positive right side down)
            pitch_rad:    Vehicle pitch in radians (positive nose up)
            accel_z_ms2:  Linear vertical acceleration from IMU
            dt:           Time elapsed since last evaluation

        Returns:
            Tuple of (GuardStatus, description)
        """
        # Service existing latch cooldown if in emergency stop
        if self._latch_timer > 0.0:
            self._latch_timer -= dt
            if self._latch_timer > 0.0:
                return (
                    GuardStatus.EMERGENCY_STOP,
                    f"LATCHED: {self._latched_reason} ({self._latch_timer:.1f}s remaining)",
                )

        abs_roll = abs(roll_rad)
        abs_pitch = abs(pitch_rad)

        # 1. Critical Tip-Over Condition
        if abs_pitch >= self.pitch_crit_rad:
            reason = (
                f"CRITICAL PITCH EXCEEDED ({math.degrees(abs_pitch):.1f}° >= {math.degrees(self.pitch_crit_rad):.1f}°)"
            )
            self._trigger_latch(reason)
            return GuardStatus.EMERGENCY_STOP, reason

        if abs_roll >= self.roll_crit_rad:
            reason = (
                f"CRITICAL ROLL EXCEEDED ({math.degrees(abs_roll):.1f}° >= {math.degrees(self.roll_crit_rad):.1f}°)"
            )
            self._trigger_latch(reason)
            return GuardStatus.EMERGENCY_STOP, reason

        # 2. Suspension Shock / Severe Drop Condition
        # Compensate for 1g static baseline
        dynamic_z = abs(accel_z_ms2 - 9.80665)
        if dynamic_z >= self.shock_threshold_ms2:
            g_force = dynamic_z / 9.80665
            reason = f"SUSPENSION SHOCK DETECTED ({g_force:.2f}g >= {self.shock_threshold_ms2/9.80665:.2f}g)"
            self._trigger_latch(reason)
            return GuardStatus.EMERGENCY_STOP, reason

        # 3. Warning Thresholds (Attitude approaching danger zone)
        if abs_pitch >= self.pitch_warn_rad:
            return GuardStatus.WARNING, f"PITCH WARNING ({math.degrees(abs_pitch):.1f}°)"

        if abs_roll >= self.roll_warn_rad:
            return GuardStatus.WARNING, f"ROLL WARNING ({math.degrees(abs_roll):.1f}°)"

        return GuardStatus.SAFE, "Attitude and Dynamics Nominal"

    def _trigger_latch(self, reason: str) -> None:
        """Lock in emergency stop for configured cooldown duration."""
        self._latch_timer = self.latch_duration_sec
        self._latched_reason = reason
        logger.critical(f"NETRA SAFETY GUARD TRIP: {reason}")
