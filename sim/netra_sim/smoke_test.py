"""Checks that a running simulator (Gazebo or CARLA) honours the NETRA sensor/actuator contract.

    ros2 run netra_sim smoke_test            # exit code 0 = every check passed

Contract the stack depends on (perception_params.yaml, vio_params.yaml):
    /camera/image_raw 1280x720 rgb8|bgr8 | /camera/camera_info fx = 500 px
    /stereo/disparity 32FC1 (0 = no match) | /imu/data with gravity | /odom_ground_truth
    /cmd_vel moves the vehicle
"""

import math
import sys
import time

import numpy as np


def main(args=None):
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, Imu

    class Probe(Node):
        def __init__(self):
            super().__init__("sim_smoke_test")
            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)
            self.n = {}
            self.sim_t = 0.0
            self.create_subscription(
                Clock, "/clock", lambda m: setattr(self, "sim_t", m.clock.sec + m.clock.nanosec * 1e-9), 10
            )
            self.last = {}
            self.imu_acc = []
            for name, typ, topic in (
                ("image", Image, "/camera/image_raw"),
                ("depth", Image, "/camera/depth_image"),
                ("disparity", Image, "/stereo/disparity"),
                ("info", CameraInfo, "/camera/camera_info"),
                ("imu", Imu, "/imu/data"),
                ("odom", Odometry, "/odom_ground_truth"),
            ):
                self.n[name] = 0
                self.create_subscription(typ, topic, self._cb(name), qos)
            self.cmd = self.create_publisher(Twist, "/cmd_vel", 10)

        def _cb(self, name):
            def cb(msg):
                self.n[name] += 1
                self.last[name] = msg
                if name == "imu":
                    a = msg.linear_acceleration
                    self.imu_acc.append(math.sqrt(a.x**2 + a.y**2 + a.z**2))

            return cb

        def spin_for(self, seconds, publish=None):
            """Spin for `seconds` of SIMULATION time (CARLA at 720p runs slower than real time)."""
            start, wall_end = self.sim_t, time.monotonic() + 30 * seconds + 5
            while self.sim_t - start < seconds and time.monotonic() < wall_end:
                if publish:
                    publish()
                rclpy.spin_once(self, timeout_sec=0.02)
            return self.sim_t - start

    rclpy.init(args=args)
    node = Probe()
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print("[%s] %-34s %s" % ("PASS" if ok else "FAIL", name, detail))

    # wait for the simulator to come up
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and not all(node.n[k] for k in ("image", "imu", "odom", "info")):
        rclpy.spin_once(node, timeout_sec=0.2)

    # stationary window: rates, IMU gravity, odometry rest pose
    for k in node.n:
        node.n[k] = 0
    node.imu_acc.clear()
    window = node.spin_for(4.0)
    rate = {k: v / max(window, 1e-6) for k, v in node.n.items()}
    for key, minimum in (("image", 10), ("depth", 10), ("disparity", 5), ("imu", 100), ("odom", 10)):
        check("%s rate" % key, rate[key] >= minimum, "%.1f Hz sim-time (need >= %d)" % (rate[key], minimum))

    img = node.last.get("image")
    if img:
        check(
            "image 16:9 rgb8/bgr8",
            img.width * 9 == img.height * 16 and img.encoding in ("rgb8", "bgr8"),
            "%dx%d %s" % (img.width, img.height, img.encoding),
        )
        check("image frame_id", img.header.frame_id.endswith("camera_optical_frame"), img.header.frame_id)
    info = node.last.get("info")
    if info:
        want = 500.0 * info.width / 1280.0  # 500 px at 1280 wide = the stack's ~104 deg FOV
        check(
            "fx matches the 104 deg FOV",
            abs(info.k[0] - want) < 0.01 * want,
            "fx = %.1f px at %d px wide (want %.1f)" % (info.k[0], info.width, want),
        )
    disp = node.last.get("disparity")
    if disp:
        d = np.frombuffer(disp.data, dtype=np.float32).reshape(disp.height, disp.width)
        valid = float(np.mean(d > 0))
        check(
            "disparity 32FC1, mostly valid",
            disp.encoding == "32FC1" and valid > 0.3,
            "%.0f%% valid pixels, max %.1f px" % (100 * valid, float(d.max())),
        )
        check("disparity within matcher range", float(d.max()) <= 128.0, "max %.1f px" % float(d.max()))
    if node.imu_acc:
        g = float(np.mean(node.imu_acc))
        check("IMU includes gravity", abs(g - 9.81) < 1.0, "mean |a| = %.2f m/s^2" % g)

    # actuation: 0.5 m/s forward for 3 s
    odom0 = node.last["odom"].pose.pose.position if "odom" in node.last else None
    cmd = Twist()
    cmd.linear.x = 0.5
    node.spin_for(3.0, publish=lambda: node.cmd.publish(cmd))
    node.cmd.publish(Twist())
    node.spin_for(0.5)
    if odom0 is not None:
        p = node.last["odom"].pose.pose.position
        moved = math.hypot(p.x - odom0.x, p.y - odom0.y)
        check("/cmd_vel moves the vehicle", moved > 0.8, "moved %.2f m in 3 s (sim) at 0.5 m/s" % moved)

    node.destroy_node()
    rclpy.shutdown()
    print("\n%d/%d checks passed" % (sum(results), len(results)))
    return 0 if all(results) and results else 1


if __name__ == "__main__":
    sys.exit(main())
