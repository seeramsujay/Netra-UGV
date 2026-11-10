"""
NETRA-UGV IMU Pre-Integration Module
======================================
Accumulates high-rate IMU samples (500 Hz) between camera frames and
computes pre-integrated rotation (ΔR), velocity (Δv), and position (Δp)
using midpoint integration.

Also computes the pre-integration Jacobians and covariance propagation
matrices for injection into the EKF-MSCKF state estimator.

Reference: MASTER_PROJECT_REPORT.md §3.3
  500 Hz IMU pre-integration (3.8 ms total VIO cycle including this)
  Sensor: TDK InvenSense ICM-42688-P (ultra-low noise 0.07°/√hr)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Compute the 3x3 skew-symmetric matrix [v]× from a 3-vector."""
    return np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=np.float64,
    )


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """
    Exponential map from so(3) to SO(3) — Rodrigues' formula.

    Args:
        omega: 3-vector (axis-angle rotation).

    Returns:
        3x3 rotation matrix.
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + skew_symmetric(omega)

    k = omega / theta
    K = skew_symmetric(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class IMUPreintegrator:
    """
    Accumulates IMU measurements between two camera keyframes and
    produces pre-integrated motion increments (ΔR, Δv, Δp) with
    associated covariance and bias Jacobians.
    """

    # Gravity vector in the ENU world frame (z up, REP-103), pointing down
    GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    def __init__(
        self,
        gyro_noise_density: float = 1.2e-3,  # rad/s/√Hz
        accel_noise_density: float = 8.0e-3,  # m/s²/√Hz
        gyro_random_walk: float = 2.0e-5,  # rad/s²/√Hz
        accel_random_walk: float = 3.0e-4,  # m/s³/√Hz
    ):
        """
        Args:
            gyro_noise_density:  Gyroscope white noise (rad/s/√Hz).
            accel_noise_density: Accelerometer white noise (m/s²/√Hz).
            gyro_random_walk:    Gyroscope bias random walk (rad/s²/√Hz).
            accel_random_walk:   Accelerometer bias random walk (m/s³/√Hz).
        """
        self.sigma_g = gyro_noise_density
        self.sigma_a = accel_noise_density
        self.sigma_bg = gyro_random_walk
        self.sigma_ba = accel_random_walk

        # Continuous-time noise covariance (Q_c)
        self.Q_c = np.diag(
            [
                self.sigma_g**2,
                self.sigma_g**2,
                self.sigma_g**2,
                self.sigma_a**2,
                self.sigma_a**2,
                self.sigma_a**2,
                self.sigma_bg**2,
                self.sigma_bg**2,
                self.sigma_bg**2,
                self.sigma_ba**2,
                self.sigma_ba**2,
                self.sigma_ba**2,
            ]
        )

        self.reset()

    def reset(self):
        """Reset pre-integration state for a new integration interval."""
        self.delta_R = np.eye(3, dtype=np.float64)  # Rotation increment
        self.delta_v = np.zeros(3, dtype=np.float64)  # Velocity increment
        self.delta_p = np.zeros(3, dtype=np.float64)  # Position increment
        self.delta_t = 0.0  # Total time elapsed

        # Covariance of the pre-integrated measurements
        self.covariance = np.zeros((15, 15), dtype=np.float64)

        # Jacobians w.r.t. bias for first-order bias correction
        self.J_R_bg = np.zeros((3, 3), dtype=np.float64)  # ∂ΔR/∂bg
        self.J_v_bg = np.zeros((3, 3), dtype=np.float64)  # ∂Δv/∂bg
        self.J_v_ba = np.zeros((3, 3), dtype=np.float64)  # ∂Δv/∂ba
        self.J_p_bg = np.zeros((3, 3), dtype=np.float64)  # ∂Δp/∂bg
        self.J_p_ba = np.zeros((3, 3), dtype=np.float64)  # ∂Δp/∂ba

        self.num_measurements = 0

    def integrate(
        self,
        gyro: np.ndarray,
        accel: np.ndarray,
        dt: float,
        bias_gyro: Optional[np.ndarray] = None,
        bias_accel: Optional[np.ndarray] = None,
    ):
        """
        Integrate a single IMU measurement using midpoint method.

        Args:
            gyro:       3-vector gyroscope measurement (rad/s).
            accel:      3-vector accelerometer measurement (m/s²).
            dt:         Time step (seconds), typically 1/500 = 0.002 s.
            bias_gyro:  Current gyroscope bias estimate (3-vector).
            bias_accel: Current accelerometer bias estimate (3-vector).
        """
        if bias_gyro is None:
            bias_gyro = np.zeros(3, dtype=np.float64)
        if bias_accel is None:
            bias_accel = np.zeros(3, dtype=np.float64)

        # Bias-corrected measurements
        omega = gyro.astype(np.float64) - bias_gyro
        acc = accel.astype(np.float64) - bias_accel

        # Rotation increment for this step
        dR = exp_so3(omega * dt)

        # Midpoint integration for velocity and position
        acc_rotated = self.delta_R @ acc

        # Update pre-integrated values
        self.delta_p += self.delta_v * dt + 0.5 * acc_rotated * dt * dt
        self.delta_v += acc_rotated * dt
        self.delta_R = self.delta_R @ dR
        self.delta_t += dt

        # --- Jacobian propagation ---
        # ∂ΔR/∂bg
        self.J_R_bg = dR.T @ self.J_R_bg - np.eye(3) * dt

        # ∂Δv/∂bg and ∂Δv/∂ba
        self.J_v_bg -= self.delta_R @ skew_symmetric(acc) @ self.J_R_bg * dt
        self.J_v_ba -= self.delta_R * dt

        # ∂Δp/∂bg and ∂Δp/∂ba
        self.J_p_bg += self.J_v_bg * dt - 0.5 * self.delta_R @ skew_symmetric(acc) @ self.J_R_bg * dt * dt
        self.J_p_ba += self.J_v_ba * dt - 0.5 * self.delta_R * dt * dt

        # --- Covariance propagation (discrete-time) ---
        F = np.eye(15, dtype=np.float64)
        F[0:3, 0:3] = dR.T
        F[0:3, 9:12] = -np.eye(3) * dt
        F[3:6, 0:3] = -self.delta_R @ skew_symmetric(acc) * dt
        F[3:6, 6:9] = np.eye(3) * dt  # Not strictly used but kept for structure
        F[3:6, 12:15] = -self.delta_R * dt
        F[6:9, 3:6] = np.eye(3) * dt

        G = np.zeros((15, 12), dtype=np.float64)
        G[0:3, 0:3] = -np.eye(3) * dt
        G[3:6, 3:6] = -self.delta_R * dt
        G[9:12, 6:9] = np.eye(3) * dt
        G[12:15, 9:12] = np.eye(3) * dt

        self.covariance = F @ self.covariance @ F.T + G @ self.Q_c @ G.T * dt

        self.num_measurements += 1

    def get_preintegrated(self) -> dict:
        """
        Return the pre-integrated measurements and metadata.

        Returns:
            Dictionary with keys:
              delta_R, delta_v, delta_p, delta_t,
              covariance, J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba,
              num_measurements
        """
        return {
            "delta_R": self.delta_R.copy(),
            "delta_v": self.delta_v.copy(),
            "delta_p": self.delta_p.copy(),
            "delta_t": self.delta_t,
            "covariance": self.covariance.copy(),
            "J_R_bg": self.J_R_bg.copy(),
            "J_v_bg": self.J_v_bg.copy(),
            "J_v_ba": self.J_v_ba.copy(),
            "J_p_bg": self.J_p_bg.copy(),
            "J_p_ba": self.J_p_ba.copy(),
            "num_measurements": self.num_measurements,
        }
