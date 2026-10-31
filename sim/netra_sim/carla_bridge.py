"""CARLA <-> ROS 2 bridge exposing the same interface as the Gazebo backend.

CARLA itself is reached through a backend: `inprocess` imports the carla client here
(0.9.16, Python 3.12), `sidecar` talks to netra_sim.carla_sidecar running under another
Python (0.9.15 needs <= 3.10).

    sensors -> ROS   /camera/image_raw  /camera/depth_image  /camera/camera_info
                     /imu/data  /odom_ground_truth  /clock
    ROS -> vehicle   /cmd_vel (geometry_msgs/Twist)

so the NETRA stack, depth_to_disparity and mission_manager run unchanged on top
of either simulator. The CARLA server runs the physics in synchronous mode and
this node is the only client that ticks it.

CARLA is left-handed (x fwd, y right, z up) and uses degrees; ROS is ENU
(x fwd, y left, z up) and radians. Everything crossing the boundary goes
through the helpers below. Importing this module does not need `carla`.
"""
import math
import time

import numpy as np

from netra_sim import scenario
from netra_sim.carla_backend import DEFAULTS, CarlaBackend, fov_deg_from_fx  # noqa: F401  (re-exported)
from netra_sim.geometry import anchor_pose, rpy_to_quaternion, wrap_angle  # noqa: F401  (re-exported)

# --------------------------------------------------------------------------
# Pure helpers (unit-tested without a CARLA server)
# --------------------------------------------------------------------------


def carla_to_ros_position(x, y, z):
    """Left-handed CARLA point -> right-handed ENU point."""
    return x, -y, z


def carla_to_ros_vector_axial(x, y, z):
    """Axial (pseudo) vector, e.g. a gyroscope reading: mirroring y flips x and z."""
    return -x, y, -z


def carla_rotation_to_rpy(roll_deg, pitch_deg, yaw_deg):
    """CARLA Rotation (deg; yaw clockwise, pitch up) -> ROS roll/pitch/yaw (rad)."""
    return math.radians(roll_deg), -math.radians(pitch_deg), -math.radians(yaw_deg)


def decode_depth_bgra(raw, width, height):
    """CARLA 'sensor.camera.depth' BGRA bytes -> float32 metres (far plane 1000 m)."""
    a = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).astype(np.float32)
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    return (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0


class SpeedController:
    """PI speed loop -> (throttle, brake, reverse) for a CARLA VehicleControl."""

    def __init__(self, kp=0.8, ki=0.3, kb=0.6, i_max=1.0):
        self.kp, self.ki, self.kb, self.i_max = kp, ki, kb, i_max
        self.integral = 0.0

    def step(self, v_cmd, v_meas, dt):
        if abs(v_cmd) < 0.02:
            self.integral = 0.0
            return 0.0, 1.0, False                      # hold position
        direction = 1.0 if v_cmd > 0 else -1.0
        speed = v_meas * direction                      # speed along the commanded direction
        if speed < -0.2:
            self.integral = 0.0
            return 0.0, 1.0, direction < 0              # still rolling the wrong way: brake first
        err = abs(v_cmd) - speed
        if err >= 0.0:
            self.integral = min(self.i_max, self.integral + err * dt)
            return min(1.0, max(0.0, self.kp * err + self.ki * self.integral)), 0.0, direction < 0
        self.integral = 0.0
        return 0.0, min(1.0, -self.kb * err), direction < 0


def steer_command(v_cmd, w_cmd, wheelbase, max_steer_rad):
    """Bicycle-model steering in CARLA's convention (+1 = full right).

    w = v * tan(delta) / L  =>  delta = atan(L * w / v); the signed v makes reversing work.
    """
    if abs(w_cmd) < 1e-3:
        return 0.0
    if abs(v_cmd) < 1e-3:
        delta = math.copysign(max_steer_rad, w_cmd)
    else:
        delta = math.atan(wheelbase * w_cmd / v_cmd)
    return max(-1.0, min(1.0, -delta / max_steer_rad))   # ROS +w is a left turn, CARLA +steer is right


def twist_to_commands(v_cmd, w_cmd, v_meas, ctrl, wheelbase, max_steer_rad, dt, min_turn_speed=0.8):
    """Ackermann stand-in for a skid-steer command.

    A car cannot spin in place, so a pure rotation request is turned into a slow
    forward crawl with full steering lock.
    """
    if abs(v_cmd) < min_turn_speed and abs(w_cmd) > 0.05:
        v_cmd = math.copysign(min_turn_speed, v_cmd if abs(v_cmd) > 0.02 else 1.0)
    throttle, brake, reverse = ctrl.step(v_cmd, v_meas, dt)
    return throttle, steer_command(v_cmd, w_cmd, wheelbase, max_steer_rad), brake, reverse


class SensorSelfCheck:
    """Catches IMU frame/gravity mix-ups that cannot be verified without a live server."""

    def __init__(self, log):
        self.log = log
        self.rest_samples = []
        self.sign_votes = []
        self.gravity_checked = False
        self.seen = 0
        self.sign_checked = False

    def imu(self, accel_norm, gyro_z, gt_yaw_rate, gt_speed):
        self.seen += 1
        if not self.gravity_checked and gt_speed < 0.05 and self.seen > 400:   # skip the spawn drop (~2 s)
            self.rest_samples.append(accel_norm)
            if len(self.rest_samples) >= 50:
                mean = float(np.median(self.rest_samples))      # first samples after spawn are transients
                if abs(mean - 9.81) > 2.0:
                    self.log('IMU at rest reads |a| = %.2f m/s^2 (expected ~9.81): gravity is not in '
                             'the accelerometer; set the imu_gravity parameter' % mean)
                else:
                    self.log('IMU gravity check ok (|a| = %.2f m/s^2)' % mean)
                self.gravity_checked = True
        if not self.sign_checked and abs(gt_yaw_rate) > 0.2:
            self.sign_votes.append(gyro_z * gt_yaw_rate > 0)
            if len(self.sign_votes) >= 25:
                if sum(self.sign_votes) < len(self.sign_votes) / 2:
                    self.log('IMU gyro-z has the opposite sign to the vehicle yaw rate: frame conversion is wrong')
                else:
                    self.log('IMU gyro sign check ok')
                self.sign_checked = True


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class RemoteBackend:
    """Same interface as CarlaBackend, served by carla_sidecar over localhost."""

    def __init__(self, cfg, port, connect_timeout_s=30.0):
        import socket
        self.cfg = cfg
        deadline = time.monotonic() + connect_timeout_s
        while True:
            try:
                self.sock = socket.create_connection(('127.0.0.1', port), timeout=5.0)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError('carla_sidecar not reachable on 127.0.0.1:%d' % port)
                time.sleep(0.5)
        self.sock.settimeout(120.0)           # a step includes physics + rendering

    def _call(self, *req):
        from netra_sim.carla_sidecar import recv_msg, send_msg
        send_msg(self.sock, req)
        reply = recv_msg(self.sock)
        if reply is None:
            raise RuntimeError('carla_sidecar closed the connection')
        if reply[0] != 'ok':
            raise RuntimeError(reply[1])
        return reply[1]

    def open(self):
        return self._call('open', self.cfg)

    def step(self, throttle, steer, brake, reverse):
        return self._call('step', (throttle, steer, brake, reverse))

    def close(self):
        from netra_sim.carla_sidecar import send_msg
        try:
            send_msg(self.sock, ('close',))
            self.sock.close()
        except OSError:
            pass


# --------------------------------------------------------------------------
# ROS node
# --------------------------------------------------------------------------

def main(args=None):
    import rclpy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, Imu

    def stamp(t):
        sec = int(t)
        return sec, int((t - sec) * 1e9)

    class CarlaBridge(Node):
        def __init__(self):
            super().__init__('carla_bridge')
            for key, default in DEFAULTS.items():
                # rclpy cannot infer the type of an empty list; spawn=[] means "use spawn_index"
                self.declare_parameter(key, [0.0] if key == 'spawn' else default)
            self.declare_parameter('backend', 'inprocess')       # 'inprocess' | 'sidecar'
            self.declare_parameter('sidecar_port', 5555)
            self.declare_parameter('cmd_timeout_s', 0.5)
            self.declare_parameter('anchor_to_scenario_start', True)

            g = lambda k: self.get_parameter(k).value             # noqa: E731
            self.cfg = {k: g(k) for k in DEFAULTS}
            self.dt = self.cfg['fixed_delta_s']
            self.cmd_timeout = g('cmd_timeout_s')
            self.anchor = g('anchor_to_scenario_start')
            self.width, self.height, self.fx = (self.cfg['image_width'], self.cfg['image_height'],
                                                self.cfg['fx'])

            sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                    history=HistoryPolicy.KEEP_LAST, depth=1)
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)
            self.image_pub = self.create_publisher(Image, '/camera/image_raw', sensor_qos)
            self.depth_pub = self.create_publisher(Image, '/camera/depth_image', sensor_qos)
            self.info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', sensor_qos)
            self.imu_pub = self.create_publisher(Imu, '/imu/data', sensor_qos)
            self.odom_pub = self.create_publisher(Odometry, '/odom_ground_truth', 10)
            self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)

            self.cmd, self.cmd_time = (0.0, 0.0), None
            self.ctrl = SpeedController()
            self.check = SensorSelfCheck(self.get_logger().warn)
            self.prev_yaw = None
            self.speed_fwd = 0.0
            self.backend = None
            try:
                if g('backend') == 'sidecar':
                    self.backend = RemoteBackend(self.cfg, g('sidecar_port'))
                else:
                    self.backend = CarlaBackend(self.cfg)
                info = self.backend.open()
            except Exception:
                self.shutdown()                  # never leave the server stuck in synchronous mode
                raise
            self.wheelbase, self.max_steer = info['wheelbase'], info['max_steer_rad']
            self.get_logger().info('CARLA %s, map %s, vehicle %s, wheelbase %.2f m' % (
                info['server_version'], info['map'], info['vehicle'], self.wheelbase))
            if not info.get('gravity_api', True):
                self.get_logger().info('this CARLA version has no IMU gravity switch; relying on the self-check')
            self.spawn_pose = None

        def _on_cmd(self, msg):
            self.cmd = (msg.linear.x, msg.angular.z)
            self.cmd_time = time.monotonic()

        def step(self):
            stale = self.cmd_time is None or time.monotonic() - self.cmd_time > self.cmd_timeout
            v_cmd, w_cmd = (0.0, 0.0) if stale else self.cmd
            throttle, steer, brake, reverse = twist_to_commands(
                v_cmd, w_cmd, self.speed_fwd, self.ctrl, self.wheelbase, self.max_steer, self.dt)
            d = self.backend.step(throttle, steer, brake, reverse)

            sim_t = d['sim_t']
            clock = Clock()
            clock.clock.sec, clock.clock.nanosec = stamp(sim_t)
            self.clock_pub.publish(clock)

            lx, ly, lz, roll, pitch, yaw = d['transform']
            rpy = carla_rotation_to_rpy(roll, pitch, yaw)
            x, y, z = carla_to_ros_position(lx, ly, lz)
            if self.spawn_pose is None:
                self.spawn_pose = (x, y, rpy[2])
                self.get_logger().info('spawned at (%.1f, %.1f, yaw %.2f rad)' % self.spawn_pose)
            vx, vy, vz = carla_to_ros_position(*d['velocity'])
            self.speed_fwd = vx * math.cos(rpy[2]) + vy * math.sin(rpy[2])

            quat = self._publish_odom(sim_t, (x, y, z), rpy, (vx, vy, vz), d['angular_velocity'])
            self._publish_imu(d['imu'], sim_t, rpy, quat)
            if d['rgb'] is not None:
                t, w, h, raw = d['rgb']
                bgr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)[..., :3]
                self._publish_image(self.image_pub, t, bgr, 'bgr8', 3)
                self._publish_info(t, w, h)
            if d['depth'] is not None:
                t, w, h, raw = d['depth']
                self._publish_image(self.depth_pub, t, decode_depth_bgra(raw, w, h), '32FC1', 4)

        def _publish_odom(self, sim_t, pos, rpy, vel_ros, ang_vel):
            x, y, z = pos
            yaw = rpy[2]
            if self.anchor:
                x, y, yaw = anchor_pose(x, y, yaw, self.spawn_pose, scenario.START)
            qx, qy, qz, qw = rpy_to_quaternion(rpy[0], rpy[1], yaw)
            msg = Odometry()
            msg.header.stamp.sec, msg.header.stamp.nanosec = stamp(sim_t)
            msg.header.frame_id, msg.child_frame_id = 'odom', 'base_footprint'
            p = msg.pose.pose
            p.position.x, p.position.y, p.position.z = x, y, z
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
            c, s = math.cos(rpy[2]), math.sin(rpy[2])          # body-frame velocities (REP-105)
            msg.twist.twist.linear.x = vel_ros[0] * c + vel_ros[1] * s
            msg.twist.twist.linear.y = -vel_ros[0] * s + vel_ros[1] * c
            msg.twist.twist.linear.z = vel_ros[2]
            msg.twist.twist.angular.z = -math.radians(ang_vel[2])
            self.odom_pub.publish(msg)
            return qx, qy, qz, qw

        def _publish_imu(self, samples, sim_t, rpy, quat):
            """Publish every IMU sample of this step (the VIO integrates them; none may be dropped)."""
            for _, t, a, w in samples:
                msg = Imu()
                msg.header.stamp.sec, msg.header.stamp.nanosec = stamp(t)
                msg.header.frame_id = 'imu_link'
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = a[0], -a[1], a[2]
                gx, gy, gz = carla_to_ros_vector_axial(*w)
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gx, gy, gz
                # world-referenced attitude (the planner's tip-over guard reads roll/pitch from it)
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = quat
                msg.orientation_covariance[0] = msg.orientation_covariance[4] = msg.orientation_covariance[8] = 1e-3
                self.imu_pub.publish(msg)
            if not samples:
                return
            yaw_rate = 0.0
            if self.prev_yaw is not None:
                yaw_rate = wrap_angle(rpy[2] - self.prev_yaw[0]) / max(sim_t - self.prev_yaw[1], 1e-6)
            self.prev_yaw = (rpy[2], sim_t)
            _, _, a, w = samples[-1]
            self.check.imu(math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2), -w[2], yaw_rate, abs(self.speed_fwd))

        def _publish_image(self, pub, t, array, encoding, bytes_per_px):
            msg = Image()
            msg.header.stamp.sec, msg.header.stamp.nanosec = stamp(t)
            msg.header.frame_id = 'camera_optical_frame'
            msg.height, msg.width = array.shape[0], array.shape[1]
            msg.encoding, msg.is_bigendian = encoding, 0
            msg.step = msg.width * bytes_per_px
            msg.data = np.ascontiguousarray(array).tobytes()
            pub.publish(msg)

        def _publish_info(self, t, w, h):
            msg = CameraInfo()
            msg.header.stamp.sec, msg.header.stamp.nanosec = stamp(t)
            msg.header.frame_id = 'camera_optical_frame'
            msg.width, msg.height = w, h
            msg.distortion_model = 'plumb_bob'
            msg.d = [0.0] * 5
            f, cx, cy = self.fx, w / 2.0, h / 2.0
            msg.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
            msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            msg.p = [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.info_pub.publish(msg)

        def shutdown(self):
            if self.backend is not None:
                self.backend.close()
                self.backend = None

    rclpy.init(args=args)
    node = None
    try:
        node = CarlaBridge()
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.step()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
