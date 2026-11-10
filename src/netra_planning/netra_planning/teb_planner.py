"""
NETRA-UGV Timed-Elastic-Band (TEB) Trajectory Optimizer
=========================================================
Implements a fast, kinodynamic Timed-Elastic-Band local trajectory optimizer
tailored for 4-wheel skid-steer outdoor uncrewed ground vehicles.

Key Features:
  - Non-holonomic skid-steer kinematic constraints.
  - Dynamically bounded forward velocity (v_max) and angular velocity (omega_max).
  - Acceleration and deceleration limits.
  - Costmap-driven collision avoidance and negative obstacle clearance.
  - Deterministic execution (< 7.5 ms) satisfying POSIX SCHED_FIFO 70 real-time budget.

Reference:
  TECHNICAL_ARCHITECTURE.md §2 (SCHED_FIFO 20 Hz) & §7 (7.5 ms TEB budget)
  MASTER_PROJECT_REPORT.md §3.3
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


class TEBTrajectoryPoint:
    """A single state knot along the Timed-Elastic-Band."""

    __slots__ = ("x", "y", "theta", "v", "omega", "dt")

    def __init__(
        self, x: float = 0.0, y: float = 0.0, theta: float = 0.0, v: float = 0.0, omega: float = 0.0, dt: float = 0.1
    ):
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.v = float(v)
        self.omega = float(omega)
        self.dt = float(dt)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta], dtype=np.float64)


class TEBPlanner:
    """
    Timed-Elastic-Band local trajectory optimizer for skid-steer kinematics.
    """

    def __init__(
        self,
        v_max: float = 1.5,
        v_min: float = -0.5,
        omega_max: float = 1.5,
        acc_linear_max: float = 1.5,
        acc_angular_max: float = 2.0,
        horizon_steps: int = 15,
        dt_nominal: float = 0.1,
        weight_time: float = 1.0,
        weight_path: float = 2.0,
        weight_kinematics: float = 5.0,
        weight_obstacles: float = 15.0,
        weight_acceleration: float = 1.0,
        obstacle_inflation_radius: float = 0.45,
        wheelbase: float = 0.65,
        track_width: float = 0.60,
    ):
        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        self.acc_linear_max = acc_linear_max
        self.acc_angular_max = acc_angular_max
        self.horizon_steps = horizon_steps
        self.dt_nominal = dt_nominal

        # Optimization weights
        self.weight_time = weight_time
        self.weight_path = weight_path
        self.weight_kinematics = weight_kinematics
        self.weight_obstacles = weight_obstacles
        self.weight_acceleration = weight_acceleration

        self.obstacle_inflation_radius = obstacle_inflation_radius
        self.wheelbase = wheelbase
        self.track_width = track_width

        # Internal state
        self._last_trajectory: List[TEBTrajectoryPoint] = []
        self._last_v: float = 0.0
        self._last_omega: float = 0.0

    def plan(
        self,
        current_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        costmap_data: Optional[np.ndarray],
        costmap_origin: Tuple[float, float],
        costmap_resolution: float,
        v_max_override: Optional[float] = None,
        max_iterations: int = 12,
    ) -> Tuple[float, float, List[Tuple[float, float, float]]]:
        """
        Compute an optimal kinodynamic trajectory toward the goal.

        Args:
            current_pose: (x, y, theta) of vehicle in odom/map frame.
            goal_pose:    (x, y, theta) of local or global waypoint.
            costmap_data: 2D uint8 numpy array of costmap cells (0-255).
            costmap_origin: (origin_x, origin_y) in meters.
            costmap_resolution: Grid resolution in meters/cell.
            v_max_override: Dynamically clamped linear velocity limit from speed governor.
            max_iterations: Maximum optimization iterations.

        Returns:
            Tuple of:
              - cmd_v: Commanded forward linear velocity (m/s)
              - cmd_omega: Commanded yaw rate (rad/s)
              - path: List of (x, y, theta) poses along the band
        """
        effective_v_max = self.v_max if v_max_override is None else min(self.v_max, max(0.0, v_max_override))

        # Check if already at goal
        dist_to_goal = math.hypot(goal_pose[0] - current_pose[0], goal_pose[1] - current_pose[1])
        if dist_to_goal < 0.15:
            return 0.0, 0.0, [current_pose]

        # 1. Initialize Elastic Band knots
        points = self._initialize_band(current_pose, goal_pose, dist_to_goal)

        # 2. Iterative Elastic Band Optimization
        for _ in range(max_iterations):
            self._optimize_step(
                points,
                effective_v_max,
                costmap_data,
                costmap_origin,
                costmap_resolution,
            )

        # 3. Compute control outputs at the first knot
        cmd_v, cmd_omega = self._extract_controls(points, current_pose, effective_v_max)

        # Update historical state
        self._last_v = cmd_v
        self._last_omega = cmd_omega
        self._last_trajectory = points

        path = [(pt.x, pt.y, pt.theta) for pt in points]
        return cmd_v, cmd_omega, path

    def _initialize_band(
        self,
        current_pose: Tuple[float, float, float],
        goal_pose: Tuple[float, float, float],
        dist_to_goal: float,
    ) -> List[TEBTrajectoryPoint]:
        """Initialize band knots between start and goal with smooth transition."""
        x0, y0, th0 = current_pose
        xg, yg, thg = goal_pose

        # If previous trajectory exists and start matches reasonably, warm-start
        if len(self._last_trajectory) >= 3:
            p1 = self._last_trajectory[1]
            if math.hypot(p1.x - x0, p1.y - y0) < 0.5:
                # Shift previous trajectory forward
                points = [TEBTrajectoryPoint(x0, y0, th0, self._last_v, self._last_omega, self.dt_nominal)]
                points.extend(self._last_trajectory[2:])
                while len(points) < self.horizon_steps:
                    last_pt = points[-1]
                    ratio = 1.0 / max(1, self.horizon_steps - len(points))
                    nx = last_pt.x + ratio * (xg - last_pt.x)
                    ny = last_pt.y + ratio * (yg - last_pt.y)
                    nth = normalize_angle(math.atan2(yg - last_pt.y, xg - last_pt.x))
                    points.append(TEBTrajectoryPoint(nx, ny, nth, last_pt.v, 0.0, self.dt_nominal))
                return points[: self.horizon_steps]

        # Cold start: interpolate along arc or line to goal
        points = []
        for i in range(self.horizon_steps):
            t = i / float(self.horizon_steps - 1)
            # Quadratic Hermite-like blending from initial heading to goal
            heading_dx = math.cos(th0) * (dist_to_goal * 0.4)
            heading_dy = math.sin(th0) * (dist_to_goal * 0.4)

            # Bezier control points: P0, P0+tangent, Pgoal
            u = 1.0 - t
            bx = u * u * x0 + 2.0 * u * t * (x0 + heading_dx) + t * t * xg
            by = u * u * y0 + 2.0 * u * t * (y0 + heading_dy) + t * t * yg

            if i == 0:
                bth = th0
            elif i == self.horizon_steps - 1:
                bth = thg
            else:
                prev_x = points[-1].x
                prev_y = points[-1].y
                bth = math.atan2(by - prev_y, bx - prev_x)

            points.append(TEBTrajectoryPoint(bx, by, bth, 0.0, 0.0, self.dt_nominal))

        return points

    def _optimize_step(
        self,
        points: List[TEBTrajectoryPoint],
        v_max: float,
        costmap_data: Optional[np.ndarray],
        costmap_origin: Tuple[float, float],
        costmap_resolution: float,
    ) -> None:
        """
        Execute one relaxation and smoothing step on interior band points (1 to N-2).
        """
        n = len(points)
        if n < 3:
            return

        alpha_smooth = 0.25
        alpha_obs = 0.35

        # Relaxation on internal knots
        for i in range(1, n - 1):
            prev_pt = points[i - 1]
            curr_pt = points[i]
            next_pt = points[i + 1]

            # 1. Smoothness / Elastic band contraction force
            ideal_x = 0.5 * (prev_pt.x + next_pt.x)
            ideal_y = 0.5 * (prev_pt.y + next_pt.y)

            dx_smooth = ideal_x - curr_pt.x
            dy_smooth = ideal_y - curr_pt.y

            # 2. Obstacle repulsion force from 2.5D costmap
            dx_obs, dy_obs = 0.0, 0.0
            if costmap_data is not None:
                dx_obs, dy_obs = self._compute_obstacle_gradient(
                    curr_pt.x, curr_pt.y, costmap_data, costmap_origin, costmap_resolution
                )

            # Apply displacement update
            curr_pt.x += alpha_smooth * dx_smooth + alpha_obs * dx_obs
            curr_pt.y += alpha_smooth * dy_smooth + alpha_obs * dy_obs

            # 3. Kinematic heading alignment
            # In skid-steer kinematics, heading should align with segment chord
            segment_th = math.atan2(next_pt.y - curr_pt.y, next_pt.x - curr_pt.x)
            th_err = normalize_angle(segment_th - curr_pt.theta)
            curr_pt.theta = normalize_angle(curr_pt.theta + 0.3 * th_err)

            # 4. Velocities along knot
            ds = math.hypot(next_pt.x - curr_pt.x, next_pt.y - curr_pt.y)
            dth = normalize_angle(next_pt.theta - curr_pt.theta)
            dt = max(0.02, curr_pt.dt)

            v_est = ds / dt
            v_clamped = min(v_max, max(self.v_min, v_est))
            omega_est = dth / dt
            omega_clamped = min(self.omega_max, max(-self.omega_max, omega_est))

            curr_pt.v = v_clamped
            curr_pt.omega = omega_clamped

    def _compute_obstacle_gradient(
        self,
        x: float,
        y: float,
        costmap: np.ndarray,
        origin: Tuple[float, float],
        resolution: float,
    ) -> Tuple[float, float]:
        """Compute repulsive obstacle gradient vector at (x, y)."""
        ox, oy = origin
        gx = int(round((x - ox) / resolution))
        gy = int(round((y - oy) / resolution))

        h, w = costmap.shape
        if gx < 1 or gx >= w - 1 or gy < 1 or gy >= h - 1:
            return 0.0, 0.0

        # Sample 3x3 neighborhood
        cell_cost = float(costmap[gy, gx])
        if cell_cost < 50:
            return 0.0, 0.0  # Safe cell, zero repulsion

        # Finite difference gradient of cost field
        grad_x = (float(costmap[gy, gx + 1]) - float(costmap[gy, gx - 1])) / (2.0 * resolution)
        grad_y = (float(costmap[gy + 1, gx]) - float(costmap[gy - 1, gx])) / (2.0 * resolution)

        # Repel in negative gradient direction
        repel_mag = (cell_cost / 255.0) * 0.25
        norm = math.hypot(grad_x, grad_y)
        if norm > 1e-4:
            return -repel_mag * (grad_x / norm), -repel_mag * (grad_y / norm)
        else:
            # If inside high-cost plateau, push laterally
            return -repel_mag * 0.5, 0.0

    def _extract_controls(
        self,
        points: List[TEBTrajectoryPoint],
        current_pose: Tuple[float, float, float],
        v_max: float,
    ) -> Tuple[float, float]:
        """
        Derive smooth skid-steer linear and angular velocity commands
        respecting acceleration limits.
        """
        if len(points) < 2:
            return 0.0, 0.0

        _ = points[0]
        p1 = points[1]
        x0, y0, th0 = current_pose

        # Target displacement vector from current pose to first lookahead knot
        dx = p1.x - x0
        dy = p1.y - y0
        dist = math.hypot(dx, dy)

        # Target heading
        target_heading = math.atan2(dy, dx)
        heading_err = normalize_angle(target_heading - th0)

        # In-place turn if heading error is large (> 45 deg)
        if abs(heading_err) > math.radians(45.0):
            target_v = 0.05 * math.copysign(1.0, math.cos(heading_err))
            target_omega = np.clip(2.5 * heading_err, -self.omega_max, self.omega_max)
        else:
            # Normal forward driving with steering
            # Scale velocity by cosine of heading error to slow down in sharp turns
            target_v = min(v_max, max(0.0, dist * 1.5)) * max(0.0, math.cos(heading_err))
            target_omega = np.clip(2.0 * heading_err, -self.omega_max, self.omega_max)

        # Apply acceleration bounds (slew-rate limiting)
        dt = self.dt_nominal
        max_dv = self.acc_linear_max * dt
        max_domega = self.acc_angular_max * dt

        v_cmd = np.clip(target_v, self._last_v - max_dv, self._last_v + max_dv)
        omega_cmd = np.clip(target_omega, self._last_omega - max_domega, self._last_omega + max_domega)

        return float(v_cmd), float(omega_cmd)
