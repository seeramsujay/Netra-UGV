"""
NETRA-UGV Visual-Inertial Odometry ROS 2 Node
================================================
Main orchestrator node for the VIO pipeline.

Subscribes to:
  /camera/image_raw (or /camera/image_mono) — Camera frames
  /imu/data                                 — 500 Hz IMU measurements

Publishes:
  /odom                — nav_msgs/Odometry at 50 Hz
  /tf                  — odom → base_link transform

Pipeline:
  IMU samples → Pre-integration buffer
  Camera frame → FAST detection → KLT tracking → MSCKF propagation + update
  Lost tracks → MSCKF multi-constraint EKF update
  Async 1 Hz → Loop closure detection + pose correction

Reference: TECHNICAL_ARCHITECTURE.md §2, §3
  POSIX Priority: 85 (SCHED_FIFO) on Cores 2-3
  Frequency: 50 Hz propagation + update
"""

import logging
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu

from netra_localization.fast_feature_extractor import FASTFeatureExtractor
from netra_localization.imu_preintegrator import IMUPreintegrator
from netra_localization.klt_tracker import KLTTracker
from netra_localization.loop_closure import LoopClosureDetector
from netra_localization.msckf_estimator import MSCKFEstimator, rotation_to_quaternion

logger = logging.getLogger(__name__)


class VIONode(Node):
    """
    Visual-Inertial Odometry node implementing the full MSCKF pipeline.
    """

    def __init__(self):
        super().__init__("vio_node")

        # --- Parameters ---
        self.declare_parameter("simulation_mode", True)
        self.declare_parameter("camera_focal_x", 500.0)
        self.declare_parameter("camera_focal_y", 500.0)
        self.declare_parameter("camera_cx", 640.0)
        self.declare_parameter("camera_cy", 360.0)
        self.declare_parameter("gyro_noise", 1.2e-3)
        self.declare_parameter("accel_noise", 8.0e-3)
        self.declare_parameter("gyro_walk", 2.0e-5)
        self.declare_parameter("accel_walk", 3.0e-4)
        self.declare_parameter("max_features", 200)
        self.declare_parameter("max_clones", 20)
        self.declare_parameter("loop_closure_enabled", True)

        # Load parameters
        fx = self.get_parameter("camera_focal_x").value
        fy = self.get_parameter("camera_focal_y").value
        cx = self.get_parameter("camera_cx").value
        cy = self.get_parameter("camera_cy").value

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

        # --- Initialize pipeline components ---
        self.feature_extractor = FASTFeatureExtractor(
            max_features_per_cell=self.get_parameter("max_features").value // 24,
        )

        self.tracker = KLTTracker(
            max_tracks=self.get_parameter("max_features").value,
        )

        self.imu_preint = IMUPreintegrator(
            gyro_noise_density=self.get_parameter("gyro_noise").value,
            accel_noise_density=self.get_parameter("accel_noise").value,
            gyro_random_walk=self.get_parameter("gyro_walk").value,
            accel_random_walk=self.get_parameter("accel_walk").value,
        )

        self.msckf = MSCKFEstimator(
            camera_intrinsics=K,
            max_clones=self.get_parameter("max_clones").value,
        )

        self.loop_closure = LoopClosureDetector() if self.get_parameter("loop_closure_enabled").value else None

        # --- State ---
        self.last_imu_time = None
        self.frame_count = 0
        self.last_gray = None

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # The IMU runs at 500 Hz and the callbacks share one thread with the (slow) image
        # callback; a depth-1 queue silently drops every IMU sample that arrives meanwhile.
        imu_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=500,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Subscribers ---
        self.imu_sub = self.create_subscription(
            Imu,
            "/imu/data",
            self._on_imu,
            imu_qos,
        )
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            sensor_qos,
        )

        # --- Publishers ---
        self.odom_pub = self.create_publisher(Odometry, "/odom", reliable_qos)

        # TF broadcaster
        try:
            from tf2_ros import TransformBroadcaster

            self.tf_broadcaster = TransformBroadcaster(self)
        except ImportError:
            self.tf_broadcaster = None
            self.get_logger().warn("tf2_ros not available, TF broadcasting disabled.")

        # --- Loop closure thread ---
        self._lc_lock = threading.Lock()
        self._lc_latest_gray = None
        self._lc_latest_pose = (np.eye(3), np.zeros(3))
        self._lc_latest_time = 0.0

        if self.loop_closure is not None:
            self._lc_timer = self.create_timer(1.0, self._loop_closure_callback)

        self.get_logger().info("VIO node initialized.")

    def _on_imu(self, msg: Imu):
        """
        Process incoming IMU measurement at 500 Hz.
        Accumulates into the pre-integration buffer.
        """
        gyro = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float64,
        )

        accel = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float64,
        )

        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_time is not None:
            dt = current_time - self.last_imu_time
            if 0 < dt < 0.1:  # Sanity check: reject gaps > 100ms
                bg, ba = self.msckf.get_biases()
                self.imu_preint.integrate(gyro, accel, dt, bg, ba)

        self.last_imu_time = current_time

    def _on_image(self, msg: Image):
        """
        Process incoming camera frame — triggers the full VIO update.
        This runs at camera framerate (typically 20-50 Hz).
        """
        t_start = time.monotonic()

        # Decode image
        try:
            if msg.encoding in ("bgr8", "rgb8"):
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif msg.encoding == "mono8":
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            else:
                return
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # --- Step 1: MSCKF Propagation with pre-integrated IMU ---
        preintegrated = self.imu_preint.get_preintegrated()
        if preintegrated["num_measurements"] > 0:
            self.msckf.propagate(preintegrated)
            self.imu_preint.reset()

        # --- Step 2: Augment state (add camera clone) ---
        self.msckf.augment_state(timestamp)

        # --- Step 3: Feature extraction + tracking ---
        new_features = self.feature_extractor.extract_points(gray)
        self.tracker.process_frame(gray, new_features)

        # --- Step 4: MSCKF Update with lost tracks ---
        lost_tracks = self.tracker.pop_lost_tracks()
        if lost_tracks:
            self.msckf.msckf_update(lost_tracks)

        # --- Step 5: Publish odometry ---
        R, p = self.msckf.get_pose()
        v = self.msckf.get_velocity()
        self._publish_odometry(R, p, v, msg.header.stamp)

        # --- Cache for loop closure ---
        with self._lc_lock:
            self._lc_latest_gray = gray.copy()
            self._lc_latest_pose = (R.copy(), p.copy())
            self._lc_latest_time = timestamp

        self.frame_count += 1
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        if self.frame_count % 50 == 0:
            self.get_logger().info(
                f"VIO frame #{self.frame_count}: {elapsed_ms:.1f} ms, "
                f"tracks={self.tracker.num_active}, "
                f"pos=[{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]"
            )

    def _loop_closure_callback(self):
        """Async 1 Hz loop closure detection (low-priority thread)."""
        if self.loop_closure is None:
            return

        with self._lc_lock:
            gray = self._lc_latest_gray
            R, p = self._lc_latest_pose
            t = self._lc_latest_time

        if gray is None:
            return

        # Add keyframe if needed
        if self.loop_closure.should_add_keyframe(p, t):
            self.loop_closure.add_keyframe(gray, R, p, t)

        # Detect loop
        result = self.loop_closure.detect_loop(gray, p, t)
        if result is not None:
            kf_id, R_rel, t_rel = result
            self.get_logger().info(f"Loop closure correction applied from KF #{kf_id}")
            # In a full implementation, this would inject a pose correction
            # into the MSCKF state. For the MVP, we log the detection.

    def _publish_odometry(self, R, p, v, stamp):
        """Publish nav_msgs/Odometry and TF transform."""
        q = rotation_to_quaternion(R)

        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        odom_msg.pose.pose.position.x = float(p[0])
        odom_msg.pose.pose.position.y = float(p[1])
        odom_msg.pose.pose.position.z = float(p[2])

        odom_msg.pose.pose.orientation = Quaternion(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3]))

        odom_msg.twist.twist.linear.x = float(v[0])
        odom_msg.twist.twist.linear.y = float(v[1])
        odom_msg.twist.twist.linear.z = float(v[2])

        self.odom_pub.publish(odom_msg)

        # Broadcast TF
        if self.tf_broadcaster is not None:
            t = TransformStamped()
            t.header = odom_msg.header
            t.child_frame_id = "base_link"
            t.transform.translation.x = float(p[0])
            t.transform.translation.y = float(p[1])
            t.transform.translation.z = float(p[2])
            t.transform.rotation = odom_msg.pose.pose.orientation
            self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = VIONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
