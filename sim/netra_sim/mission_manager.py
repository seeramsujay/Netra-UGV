"""Feeds the NETRA waypoint mission to the planner, whichever simulator is running.

Listens to ground-truth odometry (world frame) and publishes the current
waypoint on /goal_pose. The planner works in the VIO odometry frame, which
starts at the origin where the robot was spawned, so by default goals are
re-expressed relative to the spawn pose (`goals_relative_to_start`).
"""
import math

from netra_sim import scenario
from netra_sim.geometry import world_to_start_frame


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    class MissionManager(Node):
        def __init__(self):
            super().__init__('mission_manager')
            sx, sy, syaw = scenario.START
            self.declare_parameter('odom_topic', '/odom_ground_truth')
            self.declare_parameter('arrival_radius_m', 1.2)
            self.declare_parameter('goals_relative_to_start', True)
            self.declare_parameter('start', [sx, sy, syaw])
            self.declare_parameter('republish_period_s', 1.0)
            self.radius = self.get_parameter('arrival_radius_m').value
            self.relative = self.get_parameter('goals_relative_to_start').value
            self.start = tuple(self.get_parameter('start').value)
            self.waypoints = list(scenario.WAYPOINTS)
            self.index = 0

            reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=5)
            latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 history=HistoryPolicy.KEEP_LAST, depth=1)
            self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', reliable)
            self.status_pub = self.create_publisher(String, '/mission/status', latched)
            self.create_subscription(Odometry, self.get_parameter('odom_topic').value,
                                     self._on_odom, reliable)
            self.create_timer(self.get_parameter('republish_period_s').value, self._publish_goal)
            self._status('RUNNING waypoint 1/%d' % len(self.waypoints))

        def _status(self, text):
            self.get_logger().info(text)
            self.status_pub.publish(String(data=text))

        def _publish_goal(self):
            if self.index >= len(self.waypoints):
                return
            wx, wy = self.waypoints[self.index]
            if self.relative:
                wx, wy = world_to_start_frame(wx, wy, self.start)
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'odom'
            msg.pose.position.x, msg.pose.position.y = float(wx), float(wy)
            msg.pose.orientation.w = 1.0
            self.goal_pub.publish(msg)

        def _on_odom(self, msg):
            if self.index >= len(self.waypoints):
                return
            p = msg.pose.pose.position
            wx, wy = self.waypoints[self.index]
            if math.hypot(wx - p.x, wy - p.y) < self.radius:
                self.index += 1
                if self.index >= len(self.waypoints):
                    self._status('MISSION COMPLETE')
                else:
                    self._status('RUNNING waypoint %d/%d' % (self.index + 1, len(self.waypoints)))
                    self._publish_goal()

    rclpy.init(args=args)
    node = MissionManager()
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
