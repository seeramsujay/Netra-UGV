"""
NETRA-UGV Perception Node — Main ROS 2 Orchestrator
=====================================================
Orchestrates the complete terrain perception pipeline:

  Camera Image → Horizon ROI Crop → BiSeNetV2 Inference → TerrainClassification
  Disparity Map → v-Disparity Raycaster → Bayesian Filter → NegativeObstacleVoid

Both pipelines run in parallel via ROS 2 executor callbacks.

Reference: TECHNICAL_ARCHITECTURE.md §3 (Node Interaction Diagram)
  POSIX Priority: 40 (SCHED_FIFO) on Cores 4-5
  Frequency: 15 Hz semantic inference, up to 50 Hz v-disparity
"""

import logging
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, Imu

from netra_msgs.msg import NegativeObstacleVoid, TerrainClassification
from netra_perception.bayesian_filter import BayesianVoidFilter
from netra_perception.bisenetv2_inference import BiSeNetV2Inference
from netra_perception.horizon_roi_crop import HorizonROICropper
from netra_perception.vdisparity_detector import VDisparityDetector

logger = logging.getLogger(__name__)


class PerceptionNode(Node):
    """
    Main perception orchestrator node.

    Subscribes to:
      /camera/image_raw      — BGR camera frames
      /stereo/disparity       — Stereo disparity map (float32 or uint8)
      /imu/data               — IMU data for pitch angle extraction

    Publishes:
      /netra/terrain_classification — 4-class semantic terrain mask
      /netra/negative_obstacle      — Bayesian-confirmed ditch alerts
    """

    def __init__(self):
        super().__init__("perception_node")

        # --- Declare Parameters ---
        self.declare_parameter("simulation_mode", True)
        self.declare_parameter("tensorrt_engine_path", "")
        self.declare_parameter("onnx_model_path", "")
        self.declare_parameter("camera_focal_y", 500.0)
        self.declare_parameter("camera_cy", 360.0)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("horizon_margin_px", 50)
        self.declare_parameter("chassis_crop_px", 40)
        self.declare_parameter("stereo_baseline_m", 0.075)
        self.declare_parameter("max_disparity", 128)
        self.declare_parameter("bayesian_threshold", 3.5)
        self.declare_parameter("inference_rate_hz", 15.0)

        # --- Load Parameters ---
        self.sim_mode = self.get_parameter("simulation_mode").value
        engine_path = self.get_parameter("tensorrt_engine_path").value
        onnx_path = self.get_parameter("onnx_model_path").value
        f_y = self.get_parameter("camera_focal_y").value
        c_y = self.get_parameter("camera_cy").value
        img_h = self.get_parameter("image_height").value
        img_w = self.get_parameter("image_width").value
        horizon_margin = self.get_parameter("horizon_margin_px").value
        chassis_crop = self.get_parameter("chassis_crop_px").value
        baseline = self.get_parameter("stereo_baseline_m").value
        max_disp = self.get_parameter("max_disparity").value
        bayesian_thresh = self.get_parameter("bayesian_threshold").value
        inference_rate = self.get_parameter("inference_rate_hz").value

        # --- Initialize Pipeline Components ---
        self.roi_cropper = HorizonROICropper(
            focal_length_y=f_y,
            principal_point_y=c_y,
            image_height=img_h,
            image_width=img_w,
            horizon_margin_px=horizon_margin,
            chassis_crop_px=chassis_crop,
        )

        self.segmenter = BiSeNetV2Inference(
            engine_path=engine_path,
            onnx_path=onnx_path,
            input_width=1024,
            input_height=448,
        )

        self.vdisparity = VDisparityDetector(
            max_disparity=max_disp,
            stereo_baseline_m=baseline,
            focal_length_px=f_y,
        )

        self.bayesian = BayesianVoidFilter(
            log_odds_threshold=bayesian_thresh,
        )

        # --- State ---
        self.current_pitch_rad = 0.0
        self.last_image = None
        self.last_disparity = None
        self.frame_count = 0

        # --- QoS Profiles ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # --- Subscribers ---
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._on_image,
            sensor_qos,
        )

        self.disparity_sub = self.create_subscription(
            Image,
            "/stereo/disparity",
            self._on_disparity,
            sensor_qos,
        )

        self.imu_sub = self.create_subscription(
            Imu,
            "/imu/data",
            self._on_imu,
            sensor_qos,
        )

        # --- Publishers ---
        self.terrain_pub = self.create_publisher(
            TerrainClassification,
            "/netra/terrain_classification",
            reliable_qos,
        )

        self.obstacle_pub = self.create_publisher(
            NegativeObstacleVoid,
            "/netra/negative_obstacle",
            reliable_qos,
        )

        # --- Inference Timer ---
        self.inference_timer = self.create_timer(
            1.0 / inference_rate,
            self._inference_callback,
        )

        self.get_logger().info(
            f"Perception node initialized "
            f"[backend={self.segmenter.backend}, provider={getattr(self.segmenter, 'active_provider', 'N/A')}, "
            f"resolution={self.segmenter.input_w}x{self.segmenter.input_h}, sim={self.sim_mode}]"
        )
        if getattr(self.segmenter, "cuda_fallback_warning", False):
            self.get_logger().warn(
                "CUDAExecutionProvider was available but session initialized with CPUExecutionProvider! "
                "Ensure CUDA 12 and cuDNN DLLs are in system PATH."
            )

    def _on_imu(self, msg: Imu):
        """Extract pitch angle from IMU orientation quaternion."""
        q = msg.orientation
        # Pitch from quaternion: atan2(2(qw*qy - qz*qx), 1 - 2(qx² + qy²))
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        sinp = np.clip(sinp, -1.0, 1.0)
        self.current_pitch_rad = math.asin(sinp)

    def _on_image(self, msg: Image):
        """Cache the latest camera frame for processing."""
        try:
            # Decode image from ROS message
            if msg.encoding in ("bgr8", "rgb8"):
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                if msg.encoding == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "mono8":
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                self.get_logger().warn(f"Unsupported encoding: {msg.encoding}")
                return
            self.last_image = img
        except Exception as e:
            self.get_logger().error(f"Image decode error: {e}")

    def _on_disparity(self, msg: Image):
        """Cache the latest stereo disparity map."""
        try:
            if msg.encoding == "32FC1":
                self.last_disparity = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            elif msg.encoding == "mono8":
                self.last_disparity = (
                    np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width).astype(np.float32)
                )
            else:
                self.get_logger().warn(f"Unsupported disparity encoding: {msg.encoding}")
        except Exception as e:
            self.get_logger().error(f"Disparity decode error: {e}")

    def _inference_callback(self):
        """
        Main perception pipeline callback (runs at inference_rate_hz).

        Pipeline A: Image → ROI Crop → BiSeNetV2 → TerrainClassification
        Pipeline B: Disparity → v-Disparity → Bayesian → NegativeObstacleVoid
        """
        t_start = time.monotonic()

        # --- Pipeline A: Terrain Segmentation ---
        if self.last_image is not None:
            self._run_segmentation_pipeline()

        # --- Pipeline B: Negative Obstacle Detection ---
        if self.last_disparity is not None:
            self._run_vdisparity_pipeline()

        self.frame_count += 1
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        if self.frame_count % 50 == 0:
            self.get_logger().info(f"Perception frame #{self.frame_count}: {elapsed_ms:.1f} ms")

    def _run_segmentation_pipeline(self):
        """Execute terrain segmentation: ROI crop → inference → publish."""
        # Step 1: Dynamic horizon ROI crop
        cropped, row_start, row_end = self.roi_cropper.crop(self.last_image, self.current_pitch_rad)

        # Step 2: BiSeNetV2 inference
        mask, confidence = self.segmenter.infer(cropped)

        # Periodic logging of inference metrics (every 15 frames ≈ 1 sec at 15 Hz)
        if self.frame_count % 15 == 0:
            latency = getattr(self.segmenter, "last_latency_ms", 0.0)
            dist_str = getattr(self.segmenter, "last_dist_str", "N/A")
            provider = getattr(self.segmenter, "active_provider", self.segmenter.backend)
            self.get_logger().info(
                f"[BiSeNetV2 AI] Latency: {latency:.1f} ms | Provider: {provider} | "
                f"Dist: [{dist_str}] | Conf: {confidence:.2f}"
            )

        # Step 3: Publish TerrainClassification
        msg = TerrainClassification()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.semantic_mask = mask.flatten().tolist()
        msg.width = mask.shape[1]
        msg.height = mask.shape[0]
        msg.mean_confidence = float(confidence)

        self.terrain_pub.publish(msg)

    def _run_vdisparity_pipeline(self):
        """Execute negative obstacle detection: v-disparity → Bayesian → publish."""
        # Step 1: v-Disparity negative obstacle detection
        detections = self.vdisparity.detect_negative_obstacles(self.last_disparity)

        # Step 2: Bayesian temporal confirmation
        confirmed = self.bayesian.update(detections)

        # Step 3: Publish NegativeObstacleVoid for each confirmed barrier
        for barrier in confirmed:
            msg = NegativeObstacleVoid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.hazard_detected = True
            msg.edge_start_point.x = barrier["world_x"]
            msg.edge_start_point.y = barrier["world_y"] - 0.5
            msg.edge_start_point.z = 0.0
            msg.edge_end_point.x = barrier["world_x"]
            msg.edge_end_point.y = barrier["world_y"] + 0.5
            msg.edge_end_point.z = 0.0
            msg.estimated_drop_depth = 0.5  # Conservative
            msg.bayesian_log_odds = barrier["log_odds"]
            msg.consecutive_frame_hits = barrier["consecutive_hits"]

            self.obstacle_pub.publish(msg)

        # Also publish a "clear" message if no obstacles confirmed
        if not confirmed:
            msg = NegativeObstacleVoid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.hazard_detected = False
            msg.bayesian_log_odds = 0.0
            msg.consecutive_frame_hits = 0
            self.obstacle_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
