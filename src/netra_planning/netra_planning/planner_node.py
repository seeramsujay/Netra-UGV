"""
NETRA-UGV Kinodynamic TEB Local Planner Node
============================================
Integrates the Timed-Elastic-Band (TEB) trajectory optimizer, adaptive speed
governor, pitch/roll/shock safety guards, and CAN-FD motor command dispatch
into a high-frequency (20 Hz) deterministic control loop.

Subscribes to:
  /odom                          — nav_msgs/Odometry (50 Hz from MSCKF VIO)
  /netra/elevation_costmap       — nav_msgs/OccupancyGrid (10 Hz from netra_mapping)
  /netra/failsafe_state          — netra_msgs/FailsafeState (10 Hz)
  /goal_pose                     — geometry_msgs/PoseStamped (Target waypoint)
  /imu/data                      — sensor_msgs/Imu (Attitude & shock monitoring)
  /netra/terrain_classification  — netra_msgs/TerrainClassification (Semantic brush/mud)

Publishes:
  /cmd_vel                       — geometry_msgs/Twist (Standard ROS 2 command)
  /cmd_vel_secured               — netra_msgs/SecuredTwist (Optional CAN-FD SecOC motor command)
  /netra/local_plan              — nav_msgs/Path (Local trajectory for visualization)

Reference:
  TECHNICAL_ARCHITECTURE.md §2 (POSIX SCHED_FIFO 70 on Cores 4-5) & §7 (17.8 ms Latency Budget)
"""

import hashlib
import hmac
import logging
import math
import struct
import time
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Fallback imports if rclpy or ROS 2 environment is not installed
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    _HAS_RCLPY = True
except ImportError:
    _HAS_RCLPY = False

    class Node:
        def __init__(self, *args, **kwargs):
            pass

        def declare_parameter(self, *args, **kwargs):
            pass

        def get_parameter(self, *args, **kwargs):
            class _P:
                def get_parameter_value(self):
                    class _V:
                        double_value = 1.0
                        string_value = ""

                    return _V()

            return _P()

        def create_publisher(self, *args, **kwargs):
            pass

        def create_subscription(self, *args, **kwargs):
            pass

        def create_timer(self, *args, **kwargs):
            pass

        def get_logger(self):
            return logger

        def get_clock(self):
            class _C:
                def now(self):
                    class _T:
                        def to_msg(self):
                            return None

                    return _T()

            return _C()

    class QoSProfile:
        def __init__(self, *args, **kwargs):
            pass

    class ReliabilityPolicy:
        RELIABLE = 1
        BEST_EFFORT = 2

    class HistoryPolicy:
        KEEP_LAST = 1


try:
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import OccupancyGrid, Odometry, Path
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Header

    from netra_msgs.msg import FailsafeState, SecuredTwist, TerrainClassification
except ImportError:

    class Header:
        def __init__(self):
            self.stamp = None
            self.frame_id = ""

    class Twist:
        def __init__(self):
            class _V:
                x = 0.0
                y = 0.0
                z = 0.0

            self.linear = _V()
            self.angular = _V()

    class PoseStamped:
        def __init__(self):
            self.header = Header()

            class _P:
                class _Pos:
                    x = 0.0
                    y = 0.0
                    z = 0.0

                class _Ori:
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    w = 1.0

                position = _Pos()
                orientation = _Ori()

            self.pose = _P()

    class Odometry:
        pass

    class OccupancyGrid:
        pass

    class Path:
        def __init__(self):
            self.header = Header()
            self.poses = []

    class Imu:
        pass

    class SecuredTwist:
        def __init__(self):
            self.header = Header()
            self.linear_velocity_x = 0.0
            self.angular_velocity_z = 0.0
            self.freshness_counter = 0
            self.aes_cmac = [0] * 8

    class FailsafeState:
        pass

    class TerrainClassification:
        def __init__(self):
            self.semantic_mask = []
            self.class_mask = []


from netra_planning.safety_guards import GuardStatus, SafetyGuards
from netra_planning.speed_governor import (
    CLASS_MUD_HAZARD,
    CLASS_PLIANT_VEGETATION,
    MODE_NOMINAL,
    SpeedGovernor,
)
from netra_planning.teb_planner import TEBPlanner

# CAN-FD Motor Command Dispatch Helper
try:
    from netra_security.crypto_utils import (
        FreshnessCounter,
        build_secoc_payload,
        load_secoc_key,
    )
except ImportError:

    class FreshnessCounter:
        def __init__(self):
            self._val = 0

        def increment(self):
            self._val = (self._val + 1) & 0xFFFFFFFF
            return self._val

        def to_bytes(self):
            return struct.pack(">I", self._val)

    def load_secoc_key(path=None):
        return hashlib.sha256(b"NETRA_UGV_SECOC_SIM_KEY_v1").digest()[:16]

    def build_secoc_payload(v, omega, freshness, key):
        c = freshness.increment()
        data = struct.pack(">ffI", v, omega, c)
        mac = hmac.new(key, data, hashlib.sha256).digest()[:8]
        return c, mac


def quaternion_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    """Convert quaternion (x, y, z, w) to Euler angles (roll, pitch, yaw) in radians."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class PlannerNode(Node):
    """
    Main ROS 2 Planner Node orchestrating 20 Hz TEB generation and safety dispatch.
    """

    def __init__(self):
        super().__init__("netra_planner")

        # --- Parameters ---
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("v_max", 1.5)
        self.declare_parameter("v_min", -0.5)
        self.declare_parameter("omega_max", 1.5)
        self.declare_parameter("acc_linear_max", 1.5)
        self.declare_parameter("acc_angular_max", 2.0)
        self.declare_parameter("secoc_key_path", "")
        self.declare_parameter("default_waypoint_distance", 5.0)

        rate_hz = self.get_parameter("control_rate_hz").get_parameter_value().double_value
        v_max = self.get_parameter("v_max").get_parameter_value().double_value
        v_min = self.get_parameter("v_min").get_parameter_value().double_value
        omega_max = self.get_parameter("omega_max").get_parameter_value().double_value
        acc_lin = self.get_parameter("acc_linear_max").get_parameter_value().double_value
        acc_ang = self.get_parameter("acc_angular_max").get_parameter_value().double_value
        key_path = self.get_parameter("secoc_key_path").get_parameter_value().string_value
        self.default_wp_dist = self.get_parameter("default_waypoint_distance").get_parameter_value().double_value

        # --- Sub-components ---
        self.teb = TEBPlanner(
            v_max=v_max,
            v_min=v_min,
            omega_max=omega_max,
            acc_linear_max=acc_lin,
            acc_angular_max=acc_ang,
        )
        self.speed_governor = SpeedGovernor(v_nominal_max=v_max)
        self.safety_guards = SafetyGuards()

        # SecOC Cryptography
        self.secoc_key = load_secoc_key(key_path if key_path else None)
        self.freshness = FreshnessCounter()

        # --- State variables ---
        self._current_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)
        self._current_roll: float = 0.0
        self._current_pitch: float = 0.0
        self._current_az: float = 9.80665

        self._costmap_data: Optional[np.ndarray] = None
        self._costmap_origin: Tuple[float, float] = (0.0, 0.0)
        self._costmap_res: float = 0.1

        self._goal_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)
        self._failsafe_mode: int = MODE_NOMINAL

        self._veg_ratio: float = 0.0
        self._mud_ratio: float = 0.0

        self._last_cycle_time = time.monotonic()

        # --- QoS Profiles ---
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Publishers ---
        self.secured_pub = self.create_publisher(SecuredTwist, "/cmd_vel_secured", qos_reliable)
        self.twist_pub = self.create_publisher(Twist, "/cmd_vel", qos_reliable)
        self.path_pub = self.create_publisher(Path, "/netra/local_plan", qos_reliable)

        # --- Subscribers ---
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, qos_reliable)
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, "/netra/elevation_costmap", self._costmap_cb, qos_best_effort
        )
        self.failsafe_sub = self.create_subscription(
            FailsafeState, "/netra/failsafe_state", self._failsafe_cb, qos_reliable
        )
        self.goal_sub = self.create_subscription(PoseStamped, "/goal_pose", self._goal_cb, qos_reliable)
        self.imu_sub = self.create_subscription(Imu, "/imu/data", self._imu_cb, qos_best_effort)
        self.terrain_sub = self.create_subscription(
            TerrainClassification, "/netra/terrain_classification", self._terrain_cb, qos_best_effort
        )

        # --- Real-Time Control Loop Timer (20 Hz) ---
        if _HAS_RCLPY:
            self.timer = self.create_timer(1.0 / rate_hz, self._control_loop)

        self.get_logger().info(
            f"NETRA Planner Node started at {rate_hz:.1f} Hz. " f"CAN-FD motor bus active. TEB planner ready."
        )

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry):
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        self._current_pose = (px, py, yaw)

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        r, p, _ = quaternion_to_euler(q.x, q.y, q.z, q.w)
        self._current_roll = r
        self._current_pitch = p
        self._current_az = msg.linear_acceleration.z

    def _costmap_cb(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        if w > 0 and h > 0 and len(msg.data) == w * h:
            raw = np.array(msg.data, dtype=np.int8).reshape((h, w))
            # Convert -1 (unknown) to 0, uint8 0..255
            self._costmap_data = np.where(raw < 0, 0, raw).astype(np.uint8)
            self._costmap_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
            self._costmap_res = msg.info.resolution

    def _failsafe_cb(self, msg: FailsafeState):
        self._failsafe_mode = msg.current_mode

    def _goal_cb(self, msg: PoseStamped):
        gx = msg.pose.position.x
        gy = msg.pose.position.y
        q = msg.pose.orientation
        _, _, gyaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        self._goal_pose = (gx, gy, gyaw)
        self.get_logger().info(f"New goal waypoint received: ({gx:.2f}, {gy:.2f})")

    def _terrain_cb(self, msg: TerrainClassification):
        """Estimate forward corridor vegetation and mud density from semantic mask."""
        raw_mask = getattr(msg, "semantic_mask", getattr(msg, "class_mask", []))
        if len(raw_mask) == 0:
            return
        mask = np.array(raw_mask, dtype=np.uint8)
        total_pixels = len(mask)
        if total_pixels > 0:
            self._veg_ratio = float(np.sum(mask == CLASS_PLIANT_VEGETATION)) / total_pixels
            self._mud_ratio = float(np.sum(mask == CLASS_MUD_HAZARD)) / total_pixels

    # -----------------------------------------------------------------------
    # Main Control Loop (20 Hz)
    # -----------------------------------------------------------------------
    def _control_loop(self):
        t_now = time.monotonic()
        dt = max(0.01, min(0.2, t_now - self._last_cycle_time))
        self._last_cycle_time = t_now

        # Default pose if no odometry yet received
        curr_pose = self._current_pose if self._current_pose is not None else (0.0, 0.0, 0.0)

        # Default goal: 5m forward along current heading if none specified
        if self._goal_pose is None:
            gx = curr_pose[0] + self.default_wp_dist * math.cos(curr_pose[2])
            gy = curr_pose[1] + self.default_wp_dist * math.sin(curr_pose[2])
            goal_pose = (gx, gy, curr_pose[2])
        else:
            goal_pose = self._goal_pose

        # 1. Safety Guard Evaluation (Tip-Over & Suspension Shock)
        guard_status, guard_reason = self.safety_guards.evaluate(
            roll_rad=self._current_roll,
            pitch_rad=self._current_pitch,
            accel_z_ms2=self._current_az,
            dt=dt,
        )

        if guard_status == GuardStatus.EMERGENCY_STOP:
            # Immediate hard stop and failsafe lockdown
            self._dispatch_actuation(0.0, 0.0, "EMERGENCY_STOP_LOCK", dt)
            return

        # 2. Adaptive Speed Governing (Terrain, Failsafe mode, Slopes)
        slope_rad = abs(self._current_pitch)
        roughness_est = 0.02  # nominal

        # Pre-query governor for maximum allowable velocity
        _, _, v_max_allowed = self.speed_governor.compute_governed_velocity(
            cmd_v=self.teb.v_max,
            cmd_omega=0.0,
            failsafe_mode=self._failsafe_mode,
            vegetation_ratio=self._veg_ratio,
            mud_ratio=self._mud_ratio,
            slope_rad=slope_rad,
            roughness=roughness_est,
            dt=dt,
        )

        # 3. Trajectory Optimization via TEB
        cmd_v, cmd_omega, planned_path = self.teb.plan(
            current_pose=curr_pose,
            goal_pose=goal_pose,
            costmap_data=self._costmap_data,
            costmap_origin=self._costmap_origin,
            costmap_resolution=self._costmap_res,
            v_max_override=v_max_allowed,
            max_iterations=12,
        )

        # Apply secondary guard speed check (if GuardStatus.WARNING, reduce speed by 40%)
        if guard_status == GuardStatus.WARNING:
            cmd_v *= 0.6
            cmd_omega *= 0.6

        # Final governor filtering
        final_v, final_omega, _ = self.speed_governor.compute_governed_velocity(
            cmd_v=cmd_v,
            cmd_omega=cmd_omega,
            failsafe_mode=self._failsafe_mode,
            vegetation_ratio=self._veg_ratio,
            mud_ratio=self._mud_ratio,
            slope_rad=slope_rad,
            roughness=roughness_est,
            dt=dt,
        )

        # 4. Dispatch Commands & Visualization
        self._dispatch_actuation(final_v, final_omega, self.speed_governor.governor_reason, dt)
        self._publish_path(planned_path)

    def _dispatch_actuation(self, v: float, omega: float, status_msg: str, dt: float):
        """Construct motor command frame and publish to actuation bus.

        Dispatches geometry_msgs/Twist to /cmd_vel for motor controllers.
        Optional SecuredTwist with CAN-FD authentication is available for
        hardened industrial deployments (see netra_security package).
        """
        stamp = self.get_clock().now().to_msg()

        # Build SecOC payload with AES-128 CMAC and Freshness Counter
        counter_val, mac_bytes = build_secoc_payload(
            linear_vel=v,
            angular_vel=omega,
            freshness=self.freshness,
            key=self.secoc_key,
        )

        # 1. SecuredTwist for CAN-FD SecOC Actuation Driver
        sec_msg = SecuredTwist()
        sec_msg.header.stamp = stamp
        sec_msg.header.frame_id = "base_link"
        sec_msg.linear_velocity_x = float(v)
        sec_msg.angular_velocity_z = float(omega)
        sec_msg.freshness_counter = counter_val
        sec_msg.aes_cmac = list(mac_bytes)

        self.secured_pub.publish(sec_msg)

        # 2. Standard Twist for simulation/visualization
        twist_msg = Twist()
        twist_msg.linear.x = float(v)
        twist_msg.angular.z = float(omega)
        self.twist_pub.publish(twist_msg)

    def _publish_path(self, points: List[Tuple[float, float, float]]):
        """Publish nav_msgs/Path for RViz visualization."""
        if not points:
            return

        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "odom"

        for x, y, th in points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.0
            pose.pose.orientation.z = math.sin(th / 2.0)
            pose.pose.orientation.w = math.cos(th / 2.0)
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


def main(args=None):
    if not _HAS_RCLPY:
        logger.error("rclpy not installed in current environment.")
        return
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
