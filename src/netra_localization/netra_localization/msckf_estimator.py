"""
NETRA-UGV Multi-State Constraint Kalman Filter (EKF-MSCKF)
============================================================
Core state estimator implementing the OpenVINS-style MSCKF algorithm.

State vector:
  x = [q_IMU, p_IMU, v_IMU, b_g, b_a, q_cam0, p_cam0, ..., q_camN, p_camN]
  - IMU state: orientation (quaternion), position, velocity, gyro bias, accel bias
  - Sliding window of N camera clone poses (orientation + position each)

Propagation: IMU pre-integrated measurements at 50 Hz
Update: MSCKF multi-constraint update when feature tracks are lost

Reference: MASTER_PROJECT_REPORT.md §3.3, TECHNICAL_ARCHITECTURE.md §3
  50 Hz propagation rate, 3.8 ms CPU, < 8% core usage on ARM A78AE
  Drift: < 1.2% over 500m in GPS blackout
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import logging

from netra_localization.imu_preintegrator import exp_so3, skew_symmetric

logger = logging.getLogger(__name__)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        return np.array([s / 4.0, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, s / 4.0,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         s / 4.0, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, s / 4.0])


class CameraClone:
    """A camera pose clone in the sliding window."""

    def __init__(self, rotation: np.ndarray, position: np.ndarray, timestamp: float):
        self.R = rotation.copy()   # 3x3 rotation matrix (world → camera)
        self.p = position.copy()   # 3-vector position in world frame
        self.timestamp = timestamp


class MSCKFEstimator:
    """
    EKF-MSCKF state estimator with sliding window of camera clones.
    """

    # IMU state dimension: 3 (orientation error) + 3 (position) + 3 (velocity)
    #                      + 3 (gyro bias) + 3 (accel bias) = 15
    IMU_STATE_DIM = 15

    # Each camera clone: 3 (orientation error) + 3 (position) = 6
    CLONE_STATE_DIM = 6

    # Maximum sliding window size
    MAX_CLONES = 20

    # World frame is ENU (z up, REP-103), like the IMU stream: a resting accelerometer
    # reads +9.81 on z, so gravity acts along -z.
    GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)

    def __init__(
        self,
        camera_intrinsics: np.ndarray = None,
        T_cam_imu: np.ndarray = None,
        max_clones: int = 20,
        pixel_noise: float = 1.0,
    ):
        """
        Args:
            camera_intrinsics: 3x3 camera intrinsic matrix K.
            T_cam_imu:         4x4 transform from IMU to camera frame.
            max_clones:        Maximum number of camera clones in sliding window.
            pixel_noise:       Standard deviation of pixel measurement noise (pixels).
        """
        self.MAX_CLONES = max_clones
        self.pixel_noise = pixel_noise

        # Camera intrinsics (default: approximate OAK-D)
        if camera_intrinsics is None:
            self.K = np.array([
                [500.0, 0.0, 640.0],
                [0.0, 500.0, 360.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        else:
            self.K = camera_intrinsics.astype(np.float64)

        # IMU-Camera extrinsic transform
        if T_cam_imu is None:
            self.R_cam_imu = np.eye(3, dtype=np.float64)
            self.p_cam_imu = np.zeros(3, dtype=np.float64)
        else:
            self.R_cam_imu = T_cam_imu[:3, :3].astype(np.float64)
            self.p_cam_imu = T_cam_imu[:3, 3].astype(np.float64)

        # --- IMU State ---
        self.orientation = np.eye(3, dtype=np.float64)  # R: world → IMU
        self.position = np.zeros(3, dtype=np.float64)
        self.velocity = np.zeros(3, dtype=np.float64)
        self.bias_gyro = np.zeros(3, dtype=np.float64)
        self.bias_accel = np.zeros(3, dtype=np.float64)

        # --- Sliding window of camera clones ---
        self.clones: List[CameraClone] = []

        # --- Covariance ---
        total_dim = self.IMU_STATE_DIM
        self.P = np.eye(total_dim, dtype=np.float64) * 0.01

        self.is_initialized = False

    @property
    def state_dim(self) -> int:
        return self.IMU_STATE_DIM + len(self.clones) * self.CLONE_STATE_DIM

    def initialize(
        self,
        orientation: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
    ):
        """Initialize the filter state."""
        self.orientation = orientation.copy()
        self.position = position.copy()
        self.velocity = velocity.copy()
        self.is_initialized = True
        logger.info("MSCKF estimator initialized.")

    def propagate(self, preintegrated: dict):
        """
        Propagate the IMU state using pre-integrated measurements.

        Args:
            preintegrated: Dictionary from IMUPreintegrator.get_preintegrated().
        """
        if not self.is_initialized:
            # Auto-initialize with identity orientation
            self.initialize(np.eye(3), np.zeros(3), np.zeros(3))

        dR = preintegrated['delta_R']
        dv = preintegrated['delta_v']
        dp = preintegrated['delta_p']
        dt = preintegrated['delta_t']

        if dt <= 0:
            return

        # State propagation
        R_prev = self.orientation.copy()
        self.position += self.velocity * dt + 0.5 * self.GRAVITY * dt * dt + R_prev @ dp
        self.velocity += self.GRAVITY * dt + R_prev @ dv
        self.orientation = R_prev @ dR

        # Covariance propagation (simplified continuous-time)
        F = np.eye(self.IMU_STATE_DIM, dtype=np.float64)
        F[0:3, 0:3] = dR.T
        F[3:6, 3:6] = np.eye(3)
        F[3:6, 6:9] = np.eye(3) * dt
        F[6:9, 0:3] = -R_prev @ skew_symmetric(dv)
        F[6:9, 9:12] = -R_prev * dt  # Simplified

        # Process noise
        Q = preintegrated['covariance'][:self.IMU_STATE_DIM, :self.IMU_STATE_DIM]

        # Propagate only the IMU block of P
        imu_dim = self.IMU_STATE_DIM
        P_ii = self.P[:imu_dim, :imu_dim]
        self.P[:imu_dim, :imu_dim] = F @ P_ii @ F.T + Q

        # Propagate cross-correlations between IMU and clones
        n_clones = len(self.clones)
        if n_clones > 0:
            clone_dim = n_clones * self.CLONE_STATE_DIM
            P_ic = self.P[:imu_dim, imu_dim:imu_dim + clone_dim]
            self.P[:imu_dim, imu_dim:imu_dim + clone_dim] = F @ P_ic
            self.P[imu_dim:imu_dim + clone_dim, :imu_dim] = \
                self.P[:imu_dim, imu_dim:imu_dim + clone_dim].T

    def augment_state(self, timestamp: float):
        """
        Augment the state with a new camera clone at the current IMU pose.

        This is called when a new camera frame arrives.
        """
        # Compute camera pose from IMU state
        R_cam = self.R_cam_imu @ self.orientation
        p_cam = self.position + self.orientation.T @ self.p_cam_imu

        clone = CameraClone(R_cam, p_cam, timestamp)
        self.clones.append(clone)

        # Augment covariance matrix
        old_dim = self.P.shape[0]
        new_dim = old_dim + self.CLONE_STATE_DIM

        P_new = np.zeros((new_dim, new_dim), dtype=np.float64)
        P_new[:old_dim, :old_dim] = self.P

        # Jacobian of clone state w.r.t. IMU state
        J = np.zeros((self.CLONE_STATE_DIM, self.IMU_STATE_DIM), dtype=np.float64)
        J[0:3, 0:3] = self.R_cam_imu  # ∂θ_cam / ∂θ_imu
        J[3:6, 3:6] = np.eye(3)        # ∂p_cam / ∂p_imu

        # Clone-IMU cross-covariance
        P_new[old_dim:, :self.IMU_STATE_DIM] = J @ self.P[:self.IMU_STATE_DIM, :self.IMU_STATE_DIM]
        P_new[:self.IMU_STATE_DIM, old_dim:] = P_new[old_dim:, :self.IMU_STATE_DIM].T

        # Clone self-covariance
        P_new[old_dim:, old_dim:] = J @ self.P[:self.IMU_STATE_DIM, :self.IMU_STATE_DIM] @ J.T

        self.P = P_new

        # Manage sliding window
        if len(self.clones) > self.MAX_CLONES:
            self._marginalize_oldest_clone()

    def msckf_update(self, lost_tracks: list, camera_intrinsics: np.ndarray = None):
        """
        Perform the MSCKF measurement update using lost feature tracks.

        For each lost track with ≥3 observations:
          1. Triangulate the 3D landmark from multiple camera clone poses.
          2. Construct the stacked measurement residual.
          3. Project onto the left nullspace (eliminate landmark from Jacobian).
          4. Apply EKF update.

        Args:
            lost_tracks: List of FeatureTrack objects with ≥3 observations.
            camera_intrinsics: Optional override for camera K matrix.
        """
        K = camera_intrinsics if camera_intrinsics is not None else self.K

        if not lost_tracks or len(self.clones) < 2:
            return

        # Collect all residuals and Jacobians
        H_total = []
        r_total = []

        for track in lost_tracks:
            if track.age < 3 or track.age > len(self.clones):
                continue

            # Get the camera clones that observed this feature
            n_obs = min(track.age, len(self.clones))
            clone_indices = list(range(len(self.clones) - n_obs, len(self.clones)))

            if len(clone_indices) < 2:
                continue

            observations = track.observations[-n_obs:]

            # Triangulate the 3D point
            p_w = self._triangulate(observations, clone_indices, K)
            if p_w is None:
                continue

            # Build residual and Jacobian for this feature
            H_f, r_f = self._compute_residual(
                p_w, observations, clone_indices, K
            )

            if H_f is not None and r_f is not None:
                H_total.append(H_f)
                r_total.append(r_f)

        if not H_total:
            return

        # Stack all residuals
        H = np.vstack(H_total)
        r = np.concatenate(r_total)

        # Measurement noise
        R_noise = np.eye(len(r)) * (self.pixel_noise ** 2)

        # EKF update
        S = H @ self.P @ H.T + R_noise
        try:
            K_gain = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("MSCKF update: singular innovation matrix, skipping.")
            return

        delta_x = K_gain @ r

        # Apply state correction
        self._apply_correction(delta_x)

        # Update covariance
        I_KH = np.eye(self.P.shape[0]) - K_gain @ H
        self.P = I_KH @ self.P @ I_KH.T + K_gain @ R_noise @ K_gain.T

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

    def _triangulate(
        self,
        observations: list,
        clone_indices: list,
        K: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Triangulate a 3D point from multiple observations using DLT."""
        K_inv = np.linalg.inv(K)
        A = []

        for obs, ci in zip(observations, clone_indices):
            if ci >= len(self.clones):
                continue
            clone = self.clones[ci]
            # Normalized image coordinates
            uv_h = np.array([obs[0], obs[1], 1.0])
            ray = K_inv @ uv_h

            P = clone.R @ np.hstack([np.eye(3), -clone.p.reshape(3, 1)])
            A.append(ray[0] * P[2, :] - P[0, :])
            A.append(ray[1] * P[2, :] - P[1, :])

        if len(A) < 4:
            return None

        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :]
        if abs(X[3]) < 1e-10:
            return None

        p_w = X[:3] / X[3]

        # Sanity check: point should be in front of cameras
        for ci in clone_indices:
            if ci < len(self.clones):
                p_cam = self.clones[ci].R @ (p_w - self.clones[ci].p)
                if p_cam[2] <= 0:
                    return None

        return p_w

    def _compute_residual(
        self,
        p_w: np.ndarray,
        observations: list,
        clone_indices: list,
        K: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute the stacked residual and Jacobian for one feature."""
        H_x_list = []
        H_f_list = []
        r_list = []

        for obs, ci in zip(observations, clone_indices):
            if ci >= len(self.clones):
                continue
            clone = self.clones[ci]

            # Project point into camera frame
            p_cam = clone.R @ (p_w - clone.p)
            if p_cam[2] <= 0.1:
                continue

            # Predicted pixel coordinates
            z_hat = K @ (p_cam / p_cam[2])
            z_pred = z_hat[:2]
            z_obs = np.array(obs[:2])
            r_i = z_obs - z_pred
            r_list.append(r_i)

            # Jacobian w.r.t. camera pose (6-dim) — simplified
            # Full derivation omitted for brevity; using first-order approximation
            J_proj = np.array([
                [K[0, 0] / p_cam[2], 0, -K[0, 0] * p_cam[0] / p_cam[2]**2],
                [0, K[1, 1] / p_cam[2], -K[1, 1] * p_cam[1] / p_cam[2]**2],
            ])

            # ∂p_cam/∂θ_cam = [p_cam]×, ∂p_cam/∂p_cam = -R
            H_theta = J_proj @ skew_symmetric(p_cam)
            H_pos = J_proj @ (-clone.R)
            H_clone = np.hstack([H_theta, H_pos])  # 2×6

            # ∂p_cam/∂p_w = R
            H_feature = J_proj @ clone.R  # 2×3

            # Place into full state Jacobian
            state_dim = self.state_dim
            H_xi = np.zeros((2, state_dim))
            clone_start = self.IMU_STATE_DIM + ci * self.CLONE_STATE_DIM
            if clone_start + self.CLONE_STATE_DIM <= state_dim:
                H_xi[:, clone_start:clone_start + self.CLONE_STATE_DIM] = H_clone

            H_x_list.append(H_xi)
            H_f_list.append(H_feature)

        if len(r_list) < 2:
            return None, None

        H_x = np.vstack(H_x_list)
        H_f = np.vstack(H_f_list)
        r = np.concatenate(r_list)

        # Nullspace projection: eliminate the feature position from the Jacobian
        # A = null(H_f^T) such that A^T H_f = 0
        U, S, Vt = np.linalg.svd(H_f, full_matrices=True)
        rank = np.sum(S > 1e-6)
        A = U[:, rank:]  # Nullspace basis

        if A.shape[1] == 0:
            return None, None

        H_proj = A.T @ H_x
        r_proj = A.T @ r

        return H_proj, r_proj

    def _apply_correction(self, delta_x: np.ndarray):
        """Apply the EKF state correction vector."""
        # IMU orientation correction
        d_theta = delta_x[0:3]
        self.orientation = exp_so3(d_theta) @ self.orientation

        # IMU position and velocity correction
        self.position += delta_x[3:6]
        self.velocity += delta_x[6:9]

        # Bias corrections
        self.bias_gyro += delta_x[9:12]
        self.bias_accel += delta_x[12:15]

        # Camera clone corrections
        for i, clone in enumerate(self.clones):
            idx = self.IMU_STATE_DIM + i * self.CLONE_STATE_DIM
            if idx + self.CLONE_STATE_DIM <= len(delta_x):
                d_theta_c = delta_x[idx:idx + 3]
                d_pos_c = delta_x[idx + 3:idx + 6]
                clone.R = exp_so3(d_theta_c) @ clone.R
                clone.p += d_pos_c

    def _marginalize_oldest_clone(self):
        """Remove the oldest camera clone from the sliding window."""
        if not self.clones:
            return

        self.clones.pop(0)

        # Remove corresponding rows/cols from covariance
        start = self.IMU_STATE_DIM
        end = start + self.CLONE_STATE_DIM
        indices = list(range(start))
        indices.extend(range(end, self.P.shape[0]))
        self.P = self.P[np.ix_(indices, indices)]

    def get_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current (R_world_imu, p_world) pose."""
        return self.orientation.copy(), self.position.copy()

    def get_velocity(self) -> np.ndarray:
        return self.velocity.copy()

    def get_biases(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.bias_gyro.copy(), self.bias_accel.copy()
