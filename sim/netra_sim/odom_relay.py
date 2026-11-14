"""Ground-truth localization: /odom_ground_truth -> /odom (+ TF odom -> base_link).

Lets the rest of the stack (perception, costmap, planner) be exercised on its
own, independent of the VIO. Like a VIO, the published frame starts at the
origin where the vehicle was spawned, so planner goals from mission_manager
line up with it.
"""

from netra_sim import scenario
from netra_sim.geometry import anchor_pose, quaternion_to_rpy, rpy_to_quaternion


def rebase_odometry_pose(x, y, z, quat_xyzw, start, z0):
    """World-frame pose -> pose in the spawn-anchored odom frame (roll/pitch kept)."""
    roll, pitch, yaw = quaternion_to_rpy(*quat_xyzw)
    px, py, pyaw = anchor_pose(x, y, yaw, start, (0.0, 0.0, 0.0))
    return (px, py, z - z0), rpy_to_quaternion(roll, pitch, pyaw)


def main(args=None):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster

    class OdomRelay(Node):
        def __init__(self):
            super().__init__("odom_relay")
            self.declare_parameter("start", list(scenario.START))
            self.start = tuple(self.get_parameter("start").value)
            self.z0 = None
            self.pub = self.create_publisher(Odometry, "/odom", 10)
            self.tf = TransformBroadcaster(self)
            self.create_subscription(Odometry, "/odom_ground_truth", self._on_odom, 10)

        def _on_odom(self, msg):
            p, o = msg.pose.pose.position, msg.pose.pose.orientation
            if self.z0 is None:
                self.z0 = p.z
            (x, y, z), (qx, qy, qz, qw) = rebase_odometry_pose(p.x, p.y, p.z, (o.x, o.y, o.z, o.w), self.start, self.z0)
            out = Odometry()
            out.header.stamp = msg.header.stamp
            out.header.frame_id, out.child_frame_id = "odom", "base_link"
            out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = x, y, z
            out.pose.pose.orientation.x, out.pose.pose.orientation.y = qx, qy
            out.pose.pose.orientation.z, out.pose.pose.orientation.w = qz, qw
            out.twist = msg.twist
            self.pub.publish(out)

            t = TransformStamped()
            t.header = out.header
            t.child_frame_id = "base_link"
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
            t.transform.rotation = out.pose.pose.orientation
            self.tf.sendTransform(t)

    rclpy.init(args=args)
    node = OdomRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
