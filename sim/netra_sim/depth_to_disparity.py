"""Depth image -> stereo disparity, so an RGB-D simulator camera can stand in for the stereo rig.

    /camera/depth_image (32FC1, metres)  ->  /stereo/disparity (32FC1, pixels)

    disparity = fx * baseline / depth        invalid / out-of-range -> 0

The perception pipeline treats 0 as "no stereo match", which is exactly what a
real matcher returns for a void, so ditches show up as disparity holes.
"""
import numpy as np


def depth_to_disparity(depth, fx, baseline, min_depth, max_depth):
    """Vectorised conversion; depth in metres, returns float32 pixels."""
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    disp = np.zeros_like(depth)
    np.divide(fx * baseline, depth, out=disp, where=valid)
    return disp


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import CameraInfo, Image

    class DepthToDisparity(Node):
        def __init__(self):
            super().__init__('depth_to_disparity')
            self.declare_parameter('baseline_m', 0.075)      # perception_params.yaml: stereo_baseline_m
            self.declare_parameter('fx', 0.0)                # 0 -> take from camera_info
            self.declare_parameter('min_depth_m', 0.3)
            self.declare_parameter('max_depth_m', 20.0)
            self.declare_parameter('max_disparity', 128.0)   # perception_params.yaml: max_disparity
            self.baseline = self.get_parameter('baseline_m').value
            self.fx = self.get_parameter('fx').value
            self.min_depth = self.get_parameter('min_depth_m').value
            self.max_depth = self.get_parameter('max_depth_m').value
            self.max_disp = self.get_parameter('max_disparity').value

            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
            self.pub = self.create_publisher(Image, '/stereo/disparity', qos)
            self.create_subscription(Image, '/camera/depth_image', self._on_depth, qos)
            self.create_subscription(CameraInfo, '/camera/camera_info', self._on_info, qos)
            self._warned = False

        def _on_info(self, msg):
            if self.get_parameter('fx').value == 0.0 and msg.k[0] > 0.0 and msg.k[0] != self.fx:
                self.fx = msg.k[0]
                self.get_logger().info('fx = %.1f px from camera_info' % self.fx)

        def _on_depth(self, msg):
            if self.fx <= 0.0:
                if not self._warned:
                    self.get_logger().warn('waiting for camera_info (fx unknown)')
                    self._warned = True
                return
            if msg.encoding != '32FC1':
                self.get_logger().error('unsupported depth encoding %s' % msg.encoding, once=True)
                return
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
            disp = depth_to_disparity(depth, self.fx, self.baseline, self.min_depth, self.max_depth)
            disp[disp > self.max_disp] = 0.0     # closer than the matcher can resolve

            out = Image()
            out.header = msg.header
            out.height, out.width = msg.height, msg.width
            out.encoding = '32FC1'
            out.is_bigendian = 0
            out.step = msg.width * 4
            out.data = disp.tobytes()
            self.pub.publish(out)

    rclpy.init(args=args)
    node = DepthToDisparity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
