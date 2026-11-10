"""
NETRA-UGV Costmap Node — Main ROS 2 Orchestrator
===================================================
Fuses terrain classification, stereo depth, odometry, and negative obstacle
alerts into a unified 2.5D risk-traversability costmap.

Subscribes to:
  /odom                          — Vehicle odometry (grid scrolling)
  /netra/terrain_classification  — 4-class semantic terrain mask
  /netra/negative_obstacle       — Bayesian-confirmed ditch alerts
  /stereo/depth                  — Stereo depth image (optional)

Publishes:
  /netra/elevation_costmap       — nav_msgs/OccupancyGrid at 10 Hz

Reference: TECHNICAL_ARCHITECTURE.md §5.4
  POSIX Priority: 60 (SCHED_FIFO) on Cores 4-5
  Frequency: 10 Hz
"""

import logging
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from netra_mapping.barrier_injector import BarrierInjector
from netra_mapping.cost_function import CostFunction
from netra_mapping.elevation_grid import ElevationGrid
from netra_msgs.msg import NegativeObstacleVoid, TerrainClassification

logger = logging.getLogger(__name__)


class CostmapNode(Node):
    """
    2.5D risk-traversability costmap generator.
    """

    def __init__(self):
        super().__init__("costmap_node")

        # --- Parameters ---
        self.declare_parameter("grid_size", 100)
        self.declare_parameter("resolution", 0.1)
        self.declare_parameter("update_rate_hz", 10.0)
        self.declare_parameter("w_slope", 80.0)
        self.declare_parameter("w_roughness", 60.0)
        self.declare_parameter("w_semantic", 1.0)
        self.declare_parameter("w_void", 255.0)
        self.declare_parameter("inflation_radius_m", 0.3)

        grid_size = self.get_parameter("grid_size").value
        resolution = self.get_parameter("resolution").value
        rate = self.get_parameter("update_rate_hz").value

        # --- Initialize components ---
        self.grid = ElevationGrid(grid_size=grid_size, resolution=resolution)

        self.cost_fn = CostFunction(
            w_slope=self.get_parameter("w_slope").value,
            w_roughness=self.get_parameter("w_roughness").value,
            w_semantic=self.get_parameter("w_semantic").value,
            w_void=self.get_parameter("w_void").value,
            resolution=resolution,
        )

        self.barrier_injector = BarrierInjector(
            inflation_radius_m=self.get_parameter("inflation_radius_m").value,
            resolution=resolution,
        )

        # --- State ---
        self.vehicle_x = 0.0
        self.vehicle_y = 0.0
        self.latest_semantic = None
        self.update_count = 0

        # --- QoS ---
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
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self._on_odom,
            sensor_qos,
        )
        self.terrain_sub = self.create_subscription(
            TerrainClassification,
            "/netra/terrain_classification",
            self._on_terrain,
            reliable_qos,
        )
        self.obstacle_sub = self.create_subscription(
            NegativeObstacleVoid,
            "/netra/negative_obstacle",
            self._on_obstacle,
            reliable_qos,
        )

        # --- Publisher ---
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            "/netra/elevation_costmap",
            reliable_qos,
        )

        # --- Update timer ---
        self.timer = self.create_timer(1.0 / rate, self._update_callback)

        self.get_logger().info(f"Costmap node initialized [{grid_size}x{grid_size} @ {resolution}m]")

    def _on_odom(self, msg: Odometry):
        """Update vehicle position for grid scrolling."""
        self.vehicle_x = msg.pose.pose.position.x
        self.vehicle_y = msg.pose.pose.position.y
        self.grid.update_vehicle_pose(self.vehicle_x, self.vehicle_y)

    def _on_terrain(self, msg: TerrainClassification):
        """Cache latest terrain classification for costmap fusion."""
        try:
            mask = np.array(msg.semantic_mask, dtype=np.uint8).reshape(msg.height, msg.width)
            self.latest_semantic = mask
        except Exception as e:
            self.get_logger().warn(f"Terrain msg decode error: {e}")

    def _on_obstacle(self, msg: NegativeObstacleVoid):
        """Inject confirmed negative obstacles as lethal barriers."""
        if not msg.hazard_detected:
            return

        self.barrier_injector.inject_from_msg(
            self.grid,
            msg.edge_start_point.x,
            msg.edge_start_point.y,
            msg.edge_end_point.x,
            msg.edge_end_point.y,
        )

    def _update_callback(self):
        """Compute and publish the costmap at the configured rate."""
        t_start = time.monotonic()

        # Update semantic layer from latest classification
        if self.latest_semantic is not None:
            # Project semantic mask into grid cells (simplified: use forward cells)
            h, w = self.latest_semantic.shape
            for row in range(0, h, 8):  # Subsample for speed
                for col in range(0, w, 8):
                    # Approximate projection: forward distance ~ row index
                    wx = self.vehicle_x + (h - row) * self.grid.res * 0.5
                    wy = self.vehicle_y + (col - w / 2) * self.grid.res * 0.3
                    cls = int(self.latest_semantic[row, col])
                    cell = self.grid.world_to_grid(wx, wy)
                    if cell is not None:
                        self.grid.semantic[cell] = cls

        # Compute cost
        variance = self.grid.get_variance_grid()
        costmap = self.cost_fn.compute(
            self.grid.elevation,
            variance,
            self.grid.semantic,
            self.grid.void_flag,
        )

        # Publish as OccupancyGrid
        self._publish_costmap(costmap)

        self.update_count += 1
        elapsed_ms = (time.monotonic() - t_start) * 1000.0

        if self.update_count % 50 == 0:
            self.get_logger().info(f"Costmap #{self.update_count}: {elapsed_ms:.1f} ms")

    def _publish_costmap(self, costmap: np.ndarray):
        """Convert and publish costmap as OccupancyGrid."""
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"

        msg.info = MapMetaData()
        msg.info.resolution = self.grid.res
        msg.info.width = self.grid.N
        msg.info.height = self.grid.N
        msg.info.origin = Pose()
        msg.info.origin.position.x = self.grid.origin_x
        msg.info.origin.position.y = self.grid.origin_y

        # OccupancyGrid expects values in [0, 100] or -1 (unknown)
        # Map our [0, 255] costmap to [0, 100]
        occupancy = (costmap.astype(np.float32) / 255.0 * 100.0).astype(np.int8)
        msg.data = occupancy.flatten().tolist()

        self.costmap_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CostmapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
