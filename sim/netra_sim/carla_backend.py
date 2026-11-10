"""Everything that touches the CARLA client API, with no ROS imports.

Used in-process by carla_bridge (CARLA 0.9.16 client on Python 3.12) or inside
carla_sidecar (any client version, e.g. 0.9.15 on Python 3.10). Only numpy and
the standard library are needed besides `carla`, and step() returns plain
Python/bytes so it can be sent over a socket.
"""

import math
import queue

import numpy as np

DEFAULTS = {
    "host": "localhost",
    "port": 2000,
    "timeout_s": 20.0,
    "map": "",  # '' keeps the loaded map
    "weather": "ClearNoon",
    "vehicle_blueprint": "vehicle.nissan.patrol",
    "spawn": [],
    "spawn_index": 0,  # spawn = [x, y, z, yaw_deg] in CARLA coordinates
    "fixed_delta_s": 0.005,
    "camera_hz": 30.0,
    "image_width": 1280,
    "image_height": 720,
    "fx": 500.0,
    "camera_xyz": [1.0, 0.0, 1.7],
    "camera_pitch_deg": -5.0,
    "imu_xyz": [0.0, 0.0, 0.8],
    "imu_gravity": -1.0,  # >= 0: world.set_imui_sensor_gravity(value)
    # --- first-person view, self-driving and automatic data collection ---
    "spectator": "first_person",  # 'first_person' = the CARLA window shows the vehicle camera | 'none'
    "autopilot": False,  # CARLA's traffic-manager autopilot drives (ignores /cmd_vel)
    "npc_vehicles": 0,  # other autopilot cars for realistic scenes
    "weather_cycle": "",  # e.g. 'ClearNoon,ClearSunset,HardRainNoon,ClearNight'
    "weather_cycle_s": 60.0,  # sim seconds per weather
    "record_dir": "",  # '' = no recording; else a dataset run is created inside it
    "record_every_s": 0.5,  # sim seconds between saved frames
    "record_max": 0,  # stop saving after N frames (0 = unlimited)
    "record_semantic": True,  # also save semantic-segmentation labels (training targets)
}


def fov_deg_from_fx(fx, width):
    return math.degrees(2.0 * math.atan2(width / 2.0, fx))


class Recorder:
    """Writes a dataset run: rgb/*.png, depth/*.npy (float16 m), semantic/*.png (class ids), meta.jsonl."""

    def __init__(self, root, cfg):
        import json
        import os
        import time

        self.json, self.os = json, os
        self.dir = os.path.join(root, "run_" + time.strftime("%Y%m%d_%H%M%S"))
        for sub in ("rgb", "depth", "semantic"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2, default=str)
        self.meta = open(os.path.join(self.dir, "meta.jsonl"), "w")
        self.count, self.last_t = 0, -1e9
        self.every, self.limit = float(cfg["record_every_s"]), int(cfg["record_max"])
        try:
            import cv2

            self.cv2 = cv2
        except ImportError:
            self.cv2 = None  # falls back to .npy images

    @property
    def done(self):
        return bool(self.limit) and self.count >= self.limit

    def _save_image(self, sub, name, arr):
        path = self.os.path.join(self.dir, sub, name)
        if self.cv2 is not None:
            self.cv2.imwrite(path + ".png", arr)
        else:
            np.save(path + ".npy", arr)

    def maybe_save(self, d, weather):
        """d is a CarlaBackend.step() result; saves when an RGB frame is due."""
        if self.done or d["rgb"] is None or d["rgb"][0] - self.last_t < self.every:
            return
        t, w, h, raw = d["rgb"]
        self.last_t = t
        name = "%06d" % self.count
        bgr = np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., :3]
        self._save_image("rgb", name, bgr)
        if d["depth"] is not None:
            _, dw, dh, draw = d["depth"]
            a = np.frombuffer(draw, np.uint8).reshape(dh, dw, 4).astype(np.float32)
            metres = (a[..., 2] + a[..., 1] * 256.0 + a[..., 0] * 65536.0) / (256.0**3 - 1.0) * 1000.0
            np.save(self.os.path.join(self.dir, "depth", name + ".npy"), metres.astype(np.float16))
        if d.get("semantic") is not None:
            _, sw, sh, sraw = d["semantic"]
            self._save_image("semantic", name, np.frombuffer(sraw, np.uint8).reshape(sh, sw, 4)[..., 2].copy())
        x, y, z, roll, pitch, yaw = d["transform"]
        self.meta.write(
            self.json.dumps(
                {
                    "id": name,
                    "sim_t": t,
                    "weather": weather,
                    "pos_carla": [x, y, z],
                    "rpy_deg": [roll, pitch, yaw],
                    "velocity": d["velocity"],
                    "imu_accel": d["imu"][-1][2] if d["imu"] else None,
                    "imu_gyro": d["imu"][-1][3] if d["imu"] else None,
                }
            )
            + "\n"
        )
        self.meta.flush()
        self.count += 1

    def close(self):
        self.meta.close()


class CarlaBackend:
    def __init__(self, cfg, carla_module=None):
        if carla_module is None:
            import carla as carla_module
        self.carla = carla_module
        self.cfg = {**DEFAULTS, **cfg}
        self.queues = {n: queue.Queue() for n in ("rgb", "depth", "semantic", "imu")}
        self.actors = []
        self.world = None
        self.original_settings = None
        self.vehicle = None
        self.frames = 0
        self.tm = None
        self.cam_actor = None
        self.recorder = None
        self.weather_names, self.weather_i, self.next_weather_t = [], 0, 0.0
        self.presets = {}

    # ---- setup -------------------------------------------------------------
    def open(self):
        """Connect, configure and spawn. On any failure the server is left as found."""
        try:
            return self._open()
        except Exception:
            self.close()
            raise

    def _open(self):
        carla, c = self.carla, self.cfg
        client = carla.Client(c["host"], int(c["port"]))
        client.set_timeout(float(c["timeout_s"]))
        if c["map"]:
            available = [m.split("/")[-1] for m in client.get_available_maps()]
            if c["map"] not in available:
                raise RuntimeError(
                    "map '%s' not on the server; available: %s" % (c["map"], ", ".join(sorted(available)))
                )
            client.load_world(c["map"])
        self.world = client.get_world()

        presets = {
            n: getattr(carla.WeatherParameters, n)
            for n in dir(carla.WeatherParameters)
            if isinstance(getattr(carla.WeatherParameters, n), carla.WeatherParameters)
        }
        if c["weather"] not in presets:
            raise RuntimeError("unknown weather '%s'; choose from: %s" % (c["weather"], ", ".join(sorted(presets))))

        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(c["fixed_delta_s"])
        self.world.apply_settings(settings)
        self.presets = presets
        self.world.set_weather(presets[c["weather"]])
        self.weather_names = [n.strip() for n in str(c["weather_cycle"]).split(",") if n.strip()]
        for n in self.weather_names:
            if n not in presets:
                raise RuntimeError(
                    "unknown weather '%s' in weather_cycle; choose from: %s" % (n, ", ".join(sorted(presets)))
                )
        self.current_weather = c["weather"]
        if c["imu_gravity"] >= 0.0 and hasattr(self.world, "set_imui_sensor_gravity"):
            self.world.set_imui_sensor_gravity(float(c["imu_gravity"]))

        bp_lib = self.world.get_blueprint_library()
        spawn = list(c["spawn"])
        if len(spawn) == 4:
            transform = carla.Transform(
                carla.Location(x=spawn[0], y=spawn[1], z=spawn[2]), carla.Rotation(yaw=spawn[3])
            )
        else:
            points = self.world.get_map().get_spawn_points()
            transform = points[int(c["spawn_index"]) % len(points)]
        self.vehicle = self.world.spawn_actor(bp_lib.find(c["vehicle_blueprint"]), transform)
        self.actors.append(self.vehicle)

        phys = self.vehicle.get_physics_control()
        front = np.mean([[w.position.x, w.position.y] for w in phys.wheels[:2]], axis=0)
        rear = np.mean([[w.position.x, w.position.y] for w in phys.wheels[2:]], axis=0)
        raw = float(np.linalg.norm(front - rear))
        wheelbase = raw / 100.0 if raw > 20.0 else raw  # CARLA reports wheel positions in cm
        self._spawn_sensors(bp_lib)
        if c["autopilot"] or int(c["npc_vehicles"]) > 0:
            self._start_traffic(client, bp_lib)
        if c["record_dir"]:
            self.recorder = Recorder(c["record_dir"], c)
        self.world.tick()  # settle; sensors register
        self._reset_queues()
        return {
            "server_version": client.get_server_version(),
            "map": self.world.get_map().name,
            "vehicle": self.vehicle.type_id,
            "wheelbase": wheelbase,
            "max_steer_rad": math.radians(phys.wheels[0].max_steer_angle),
            "gravity_api": hasattr(self.world, "set_imui_sensor_gravity"),
        }

    def _spawn_sensors(self, bp_lib):
        carla, c = self.carla, self.cfg
        fov = fov_deg_from_fx(c["fx"], c["image_width"])
        cx, cy, cz = c["camera_xyz"]
        cam_tf = carla.Transform(carla.Location(x=cx, y=cy, z=cz), carla.Rotation(pitch=c["camera_pitch_deg"]))
        cams = [("rgb", "sensor.camera.rgb"), ("depth", "sensor.camera.depth")]
        if c["record_dir"] and c["record_semantic"]:
            cams.append(("semantic", "sensor.camera.semantic_segmentation"))
        for key, bp_id in cams:
            bp = bp_lib.find(bp_id)
            bp.set_attribute("image_size_x", str(int(c["image_width"])))
            bp.set_attribute("image_size_y", str(int(c["image_height"])))
            bp.set_attribute("fov", "%.4f" % fov)
            bp.set_attribute("sensor_tick", "%.5f" % (1.0 / c["camera_hz"]))
            sensor = self.world.spawn_actor(bp, cam_tf, attach_to=self.vehicle)
            sensor.listen(self.queues[key].put)
            self.actors.append(sensor)
            if key == "rgb":
                self.cam_actor = sensor
        ix, iy, iz = c["imu_xyz"]
        imu_bp = bp_lib.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", "0.0")  # every physics step
        imu = self.world.spawn_actor(imu_bp, carla.Transform(carla.Location(x=ix, y=iy, z=iz)), attach_to=self.vehicle)
        imu.listen(self.queues["imu"].put)
        self.actors.append(imu)

    def _start_traffic(self, client, bp_lib):
        """Autopilot for the ego vehicle and optional NPC cars, on a synchronous traffic manager."""
        c = self.cfg
        self.tm = client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        port = self.tm.get_port()
        if c["autopilot"]:
            self.vehicle.set_autopilot(True, port)
        points = list(self.world.get_map().get_spawn_points())
        spawned = 0
        for tf in points[1:]:
            if spawned >= int(c["npc_vehicles"]):
                break
            cars = [b for b in bp_lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
            npc = self.world.try_spawn_actor(cars[spawned % len(cars)], tf)
            if npc is not None:
                npc.set_autopilot(True, port)
                self.actors.append(npc)
                spawned += 1

    def _reset_queues(self):
        for q in self.queues.values():
            while not q.empty():
                q.get_nowait()

    # ---- stepping ----------------------------------------------------------
    def step(self, throttle, steer, brake, reverse):
        """Apply a control, advance one physics step, return everything measured."""
        carla = self.carla
        if not self.cfg["autopilot"]:
            self.vehicle.apply_control(
                carla.VehicleControl(
                    throttle=float(throttle), steer=float(steer), brake=float(brake), reverse=bool(reverse)
                )
            )
        frame = self.world.tick()
        self.frames += 1
        sim_t = self.world.get_snapshot().timestamp.elapsed_seconds
        tf, vel, av = self.vehicle.get_transform(), self.vehicle.get_velocity(), self.vehicle.get_angular_velocity()
        if self.cfg["spectator"] == "first_person" and self.cam_actor is not None:
            self.world.get_spectator().set_transform(self.cam_actor.get_transform())
            if self.frames % 400 == 0:  # ~2 s of sim time
                sp = self.world.get_spectator().get_transform().location
                print(
                    "[backend] t=%.1f ego=(%.1f,%.1f) speed=%.2f spectator=(%.1f,%.1f) autopilot=%s"
                    % (sim_t, tf.location.x, tf.location.y, vel.x**2 + vel.y**2, sp.x, sp.y, self.cfg["autopilot"]),
                    flush=True,
                )
        if self.weather_names and sim_t >= self.next_weather_t:
            self.current_weather = self.weather_names[self.weather_i % len(self.weather_names)]
            self.world.set_weather(self.presets[self.current_weather])
            self.weather_i += 1
            self.next_weather_t = sim_t + float(self.cfg["weather_cycle_s"])

        imu = []
        q = self.queues["imu"]
        try:
            while not imu or imu[-1][0] < frame:  # every sample up to this frame
                m = q.get(timeout=1.0)
                imu.append(
                    (
                        m.frame,
                        m.timestamp,
                        (m.accelerometer.x, m.accelerometer.y, m.accelerometer.z),
                        (m.gyroscope.x, m.gyroscope.y, m.gyroscope.z),
                    )
                )
            while True:
                m = q.get_nowait()
                imu.append(
                    (
                        m.frame,
                        m.timestamp,
                        (m.accelerometer.x, m.accelerometer.y, m.accelerometer.z),
                        (m.gyroscope.x, m.gyroscope.y, m.gyroscope.z),
                    )
                )
        except queue.Empty:
            pass

        out = {
            "frame": frame,
            "sim_t": sim_t,
            "transform": (
                tf.location.x,
                tf.location.y,
                tf.location.z,
                tf.rotation.roll,
                tf.rotation.pitch,
                tf.rotation.yaw,
            ),
            "velocity": (vel.x, vel.y, vel.z),
            "angular_velocity": (av.x, av.y, av.z),
            "imu": imu,
            "rgb": None,
            "depth": None,
            "semantic": None,
        }
        for key in ("rgb", "depth", "semantic"):
            latest = None
            try:
                while True:
                    latest = self.queues[key].get_nowait()
            except queue.Empty:
                pass
            if latest is not None:
                out[key] = (latest.timestamp, latest.width, latest.height, bytes(latest.raw_data))
        if self.recorder is not None:
            self.recorder.maybe_save(out, self.current_weather)
        out["recorded"] = self.recorder.count if self.recorder else 0
        return out

    # ---- teardown ----------------------------------------------------------
    def close(self):
        """Always hand the server back asynchronous, with our actors removed."""
        for actor in reversed(self.actors):
            try:
                if hasattr(actor, "set_autopilot"):
                    actor.set_autopilot(False)
                if hasattr(actor, "stop"):
                    actor.stop()
                actor.destroy()
            except Exception:
                pass
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        self.actors = []
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except Exception:
                pass
            self.tm = None
        if self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                pass
            self.original_settings = None
