"""Runs the real carla_bridge.main() against an in-process fake `carla` module.

This exercises everything on the ROS side (parameter declaration, publishers,
message fields, the /cmd_vel -> VehicleControl path, clean shutdown) without a
CARLA server. It cannot prove the server-side behaviour; see sim/README.md.
"""

import os
import sys
import threading
import time
import types

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

W, H = 64, 36  # tiny images keep the test fast


class V3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Location(V3):
    pass


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location, self.rotation = location or Location(), rotation or Rotation()


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle, self.steer, self.brake, self.reverse = throttle, steer, brake, reverse


class WeatherParameters:
    pass


for _name in ("ClearNoon", "ClearNight", "HardRainNoon"):
    setattr(WeatherParameters, _name, WeatherParameters())


class Blueprint:
    def __init__(self, type_id):
        self.type_id, self.attrs = type_id, {}

    def set_attribute(self, k, v):
        self.attrs[k] = v


class Library:
    def find(self, type_id):
        return Blueprint(type_id)


class Wheel:
    def __init__(self, x, y, steer):
        self.position, self.max_steer_angle = V3(x * 100, y * 100), steer  # CARLA: cm


class Actor:
    def __init__(self, world, bp, transform, parent=None):
        self.world, self.type_id, self.transform, self.parent = world, bp.type_id, transform, parent
        self.callback, self.destroyed, self.last_control = None, False, None
        self.speed = 0.0

    def listen(self, cb):
        self.callback = cb

    def stop(self):
        self.callback = None

    def destroy(self):
        self.destroyed = True

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return V3(self.speed, 0.0, 0.0)

    def get_angular_velocity(self):
        return V3()

    def apply_control(self, c):
        self.last_control = c

    def get_physics_control(self):
        return types.SimpleNamespace(
            wheels=[Wheel(1.4, -0.8, 70), Wheel(1.4, 0.8, 70), Wheel(-1.5, -0.8, 0), Wheel(-1.5, 0.8, 0)]
        )


class Settings:
    synchronous_mode, fixed_delta_seconds = False, 0.0


class World:
    def __init__(self):
        self.frame, self.settings, self.actors, self.weather, self.max_ticks = 0, Settings(), [], None, 1000
        self.gravity = None
        self.history = []

    def get_map(self):
        return types.SimpleNamespace(
            name="Town_Fake", get_spawn_points=lambda: [Transform(Location(100.0, -40.0, 0.5), Rotation(yaw=90.0))]
        )

    def get_settings(self):
        s = Settings()
        s.__dict__.update(self.settings.__dict__)
        return s

    def apply_settings(self, s):
        self.settings = s
        self.history.append((s.synchronous_mode, s.fixed_delta_seconds))

    def set_weather(self, w):
        self.weather = w

    def set_imui_sensor_gravity(self, g):
        self.gravity = g

    def get_blueprint_library(self):
        return Library()

    def get_spectator(self):
        self.spectator = getattr(self, "spectator", types.SimpleNamespace(transform=None))
        self.spectator.set_transform = lambda tf: setattr(self.spectator, "transform", tf)
        self.spectator.get_transform = lambda: self.spectator.transform or Transform()
        return self.spectator

    def spawn_actor(self, bp, transform, attach_to=None):
        a = Actor(self, bp, transform, attach_to)
        self.actors.append(a)
        return a

    def tick(self):
        if self.frame >= self.max_ticks:
            raise KeyboardInterrupt  # ends the bridge loop through its normal shutdown path
        time.sleep(0.004)  # a real server paces ticks; also lets DDS discovery finish
        self.frame += 1
        t = self.frame * self.settings.fixed_delta_seconds
        for a in self.actors:
            if a.callback is None:
                continue
            if a.type_id == "sensor.other.imu":
                a.callback(
                    types.SimpleNamespace(
                        frame=self.frame, timestamp=t, accelerometer=V3(0.0, 0.0, 9.81), gyroscope=V3(0.0, 0.0, 0.0)
                    )
                )
            elif self.frame % 6 == 0:  # cameras: sensor_tick > physics step
                raw = np.zeros((H, W, 4), np.uint8)
                if a.type_id == "sensor.camera.depth":
                    raw[..., :3] = (0, 0, 8)  # constant depth code
                a.callback(
                    types.SimpleNamespace(frame=self.frame, timestamp=t, width=W, height=H, raw_data=raw.tobytes())
                )
        return self.frame

    def get_snapshot(self):
        return types.SimpleNamespace(
            timestamp=types.SimpleNamespace(elapsed_seconds=self.frame * self.settings.fixed_delta_seconds)
        )


class Client:
    world = None

    def __init__(self, host, port):
        self.host, self.port = host, port
        Client.world = Client.world or World()

    def set_timeout(self, t):
        pass

    def get_available_maps(self):
        return ["/Game/Carla/Maps/Town_Fake", "/Game/Carla/Maps/Mine_01"]

    def load_world(self, name):
        pass

    def get_world(self):
        return Client.world

    def get_server_version(self):
        return "0.9.16-fake"


def make_fake_carla():
    mod = types.ModuleType("carla")
    for cls in (Client, Location, Rotation, Transform, VehicleControl, WeatherParameters):
        setattr(mod, cls.__name__, cls)
    return mod


@pytest.fixture
def fake_carla(monkeypatch):
    Client.world = None
    monkeypatch.setitem(sys.modules, "carla", make_fake_carla())
    monkeypatch.setenv("ROS_DOMAIN_ID", str(70 + os.getpid() % 100))
    yield
    Client.world = None


def run_bridge(extra_args=()):
    from netra_sim import carla_bridge

    err = []

    def target():
        try:
            carla_bridge.main(["--ros-args", *extra_args])
        except BaseException as e:  # surfaced by the test
            err.append(e)

    th = threading.Thread(target=target, daemon=True)
    th.start()
    return th, err


def test_bridge_publishes_contract_and_drives_vehicle(fake_carla):
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, Imu

    ctx = rclpy.Context()  # separate from the bridge's default context
    rclpy.init(context=ctx)
    probe = Node("probe", context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(probe)
    best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=50)
    got = {k: [] for k in ("image", "depth", "info", "imu", "odom")}
    for key, typ, topic in (
        ("image", Image, "/camera/image_raw"),
        ("depth", Image, "/camera/depth_image"),
        ("info", CameraInfo, "/camera/camera_info"),
        ("imu", Imu, "/imu/data"),
        ("odom", Odometry, "/odom_ground_truth"),
    ):
        probe.create_subscription(typ, topic, got[key].append, best_effort if key != "odom" else 10)
    cmd_pub = probe.create_publisher(Twist, "/cmd_vel", 10)

    th, err = run_bridge(
        [
            "-p",
            "image_width:=%d" % W,
            "-p",
            "image_height:=%d" % H,
            "-p",
            "weather:=ClearNight",
            "-p",
            "imu_gravity:=9.81",
            "-p",
            "map:=Mine_01",
        ]
    )
    try:
        cmd = Twist()
        cmd.linear.x, cmd.angular.z = 1.0, 0.2
        deadline = time.monotonic() + 20
        while th.is_alive() and time.monotonic() < deadline:
            cmd_pub.publish(cmd)
            executor.spin_once(timeout_sec=0.02)
        th.join(timeout=5)
    finally:
        executor.shutdown()
        probe.destroy_node()
        rclpy.shutdown(context=ctx)

    assert not th.is_alive(), "bridge did not shut down"
    assert not err, err
    world = Client.world
    vehicle = world.actors[0]

    # setup
    assert world.history[0] == (True, pytest.approx(0.005))  # synchronous, 200 Hz
    assert world.weather is WeatherParameters.ClearNight and world.gravity == 9.81
    assert {a.type_id for a in world.actors} == {
        vehicle.type_id,
        "sensor.camera.rgb",
        "sensor.camera.depth",
        "sensor.other.imu",
    }
    rgb = next(a for a in world.actors if a.type_id == "sensor.camera.rgb")
    assert float(rgb.world is world) and rgb.parent is vehicle
    # clean shutdown: actors destroyed, world settings restored
    assert all(a.destroyed for a in world.actors) and world.history[-1] == (False, 0.0)

    # first-person view: the CARLA window follows the camera
    assert world.spectator.transform is rgb.transform

    # published data
    assert len(got["imu"]) >= 30 and len(got["image"]) >= 3 and len(got["depth"]) >= 3 and got["odom"]
    imu = got["imu"][-1]
    assert imu.linear_acceleration.z == pytest.approx(9.81) and imu.header.frame_id == "imu_link"
    assert imu.orientation_covariance[0] > 0  # attitude provided for the planner
    img = got["image"][-1]
    assert (img.width, img.height, img.encoding) == (W, H, "bgr8") and len(img.data) == W * H * 3
    d = np.frombuffer(got["depth"][-1].data, np.float32)
    assert got["depth"][-1].encoding == "32FC1" and np.all(d > 0)
    ci = got["info"][-1]
    assert ci.k[0] == pytest.approx(500.0) and ci.k[2] == W / 2 and ci.header.frame_id == "camera_optical_frame"
    # ground truth is re-anchored onto the test field's start pose
    from netra_sim import scenario

    p = got["odom"][-1].pose.pose.position
    assert (p.x, p.y) == pytest.approx(scenario.START[:2], abs=1e-6)

    # /cmd_vel 1.0 m/s, +0.2 rad/s (left) -> throttle, steer left (negative in CARLA), no brake
    c = vehicle.last_control
    assert c.throttle > 0 and c.brake == 0 and c.steer < 0 and not c.reverse


def test_bridge_rejects_unknown_map_and_weather(fake_carla):
    th, err = run_bridge(["-p", "map:=NoSuchMap"])
    th.join(timeout=10)
    assert err and "NoSuchMap" in str(err[0]) and "Mine_01" in str(err[0])
    assert Client.world is None or Client.world.settings.synchronous_mode is False


def test_failed_setup_leaves_server_in_async_mode(fake_carla):
    th, err = run_bridge(["-p", "weather:=NoSuchWeather"])
    th.join(timeout=10)
    assert err and "NoSuchWeather" in str(err[0]) and "ClearNoon" in str(err[0])
    assert Client.world.settings.synchronous_mode is False and not Client.world.actors


def test_recorder_writes_a_dataset(tmp_path):
    from netra_sim.carla_backend import Recorder

    cfg = {"record_every_s": 0.5, "record_max": 3}
    rec = Recorder(str(tmp_path), cfg)
    w, h = 8, 4
    bgra = np.zeros((h, w, 4), np.uint8)
    sem = bgra.copy()
    sem[..., 2] = 7  # class id lives in the red channel
    for i in range(10):  # frames every 0.2 s -> saved every 0.6 s
        t = 0.2 * i
        rec.maybe_save(
            {
                "rgb": (t, w, h, bgra.tobytes()),
                "depth": (t, w, h, bgra.tobytes()),
                "semantic": (t, w, h, sem.tobytes()),
                "transform": (1, 2, 3, 0, 0, 90),
                "velocity": (0, 0, 0),
                "imu": [(1, t, (0, 0, 9.8), (0, 0, 0))],
            },
            "ClearNoon",
        )
    rec.close()
    assert rec.count == 3 and rec.done  # honours record_max
    files = sorted(p.name for p in (tmp_path / rec.dir.split("/")[-1] / "rgb").iterdir())
    assert files == ["000000.png", "000001.png", "000002.png"]
    import json

    meta = [json.loads(line) for line in open(rec.dir + "/meta.jsonl")]
    assert [m["id"] for m in meta] == ["000000", "000001", "000002"] and meta[0]["weather"] == "ClearNoon"
    assert np.load(rec.dir + "/depth/000000.npy").shape == (h, w)
    import cv2

    assert int(cv2.imread(rec.dir + "/semantic/000000.png", cv2.IMREAD_UNCHANGED).max()) == 7
