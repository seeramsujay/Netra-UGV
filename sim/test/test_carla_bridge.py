import math

import numpy as np
import pytest

from netra_sim import carla_bridge as cb


def test_handedness_conversion():
    assert cb.carla_to_ros_position(1, 2, 3) == (1, -2, 3)
    # CARLA yaw is clockwise: +90 deg (facing CARLA +y, i.e. to the right) is -90 deg in ENU
    assert cb.carla_rotation_to_rpy(0, 0, 90)[2] == pytest.approx(-math.pi / 2)
    assert cb.carla_rotation_to_rpy(0, 10, 0)[1] == pytest.approx(-math.radians(10))
    assert cb.carla_to_ros_vector_axial(1, 2, 3) == (-1, 2, -3)


def test_quaternion_is_unit_and_yaw_only_case():
    q = cb.rpy_to_quaternion(0.1, -0.2, 0.7)
    assert np.linalg.norm(q) == pytest.approx(1.0)
    x, y, z, w = cb.rpy_to_quaternion(0, 0, math.pi / 2)
    assert (x, y) == (0, 0) and z == pytest.approx(math.sin(math.pi / 4)) and w == pytest.approx(math.cos(math.pi / 4))


def test_anchor_maps_spawn_onto_anchor_and_preserves_motion():
    spawn, anchor = (100.0, -40.0, 1.0), (-22.0, -14.0, 0.6)
    assert cb.anchor_pose(*spawn, spawn, anchor) == pytest.approx(anchor)
    # drive 5 m straight ahead of the spawn pose: must be 5 m straight ahead of the anchor pose
    x = spawn[0] + 5 * math.cos(spawn[2]); y = spawn[1] + 5 * math.sin(spawn[2])
    ax, ay, ayaw = cb.anchor_pose(x, y, spawn[2], spawn, anchor)
    assert (ax, ay, ayaw) == pytest.approx((anchor[0] + 5 * math.cos(anchor[2]), anchor[1] + 5 * math.sin(anchor[2]), anchor[2]))
    # turning left by 0.3 rad stays a left turn
    assert cb.anchor_pose(*spawn[:2], spawn[2] + 0.3, spawn, anchor)[2] == pytest.approx(anchor[2] + 0.3)


def test_decode_depth_bgra():
    # CARLA packs depth as R + G*256 + B*65536 over (256^3 - 1), spanning 0..1000 m
    px = np.zeros((1, 3, 4), dtype=np.uint8)
    px[0, 0] = (0, 0, 0, 255)                                  # 0 m
    px[0, 1] = (0xFF, 0xFF, 0xFF, 255)                         # far plane
    mid = int((256 ** 3 - 1) * 0.01)                           # 10 m
    px[0, 2] = (mid >> 16 & 255, mid >> 8 & 255, mid & 255, 255)   # B, G, R
    out = cb.decode_depth_bgra(px.tobytes(), 3, 1)
    assert out[0, 0] == 0.0 and out[0, 1] == pytest.approx(1000.0, rel=1e-4) and out[0, 2] == pytest.approx(10.0, rel=1e-3)


def test_fov_for_stack_intrinsics():
    # fx = 500 px on a 1280 px wide image -> the same 104.0 deg the Gazebo camera uses
    assert cb.fov_deg_from_fx(500.0, 1280) == pytest.approx(math.degrees(1.81517), abs=0.01)


def test_steering_signs_and_limits():
    L, lock = 2.9, math.radians(70)
    assert cb.steer_command(2.0, 0.0, L, lock) == 0.0
    assert cb.steer_command(2.0, 0.3, L, lock) < 0        # ROS left (+w) -> CARLA negative (left)
    assert cb.steer_command(2.0, -0.3, L, lock) > 0
    assert cb.steer_command(-1.0, 0.3, L, lock) > 0      # reversing flips the geometry
    assert cb.steer_command(2.0, 50.0, L, lock) == -1.0   # saturates
    assert cb.steer_command(0.0, 0.5, L, lock) == -1.0    # no forward speed: full lock


def test_speed_controller_converges_without_overshoot():
    ctrl, v, dt = cb.SpeedController(), 0.0, 0.01
    for _ in range(1500):
        throttle, brake, reverse = ctrl.step(1.5, v, dt)
        v += (4.0 * throttle - 6.0 * brake - 0.1 * v) * dt   # crude vehicle
        assert not reverse
        assert v < 1.5 * 1.15
    assert v == pytest.approx(1.5, abs=0.1)


def test_speed_controller_brakes_when_told_to_stop_and_reverses():
    ctrl = cb.SpeedController()
    assert ctrl.step(0.0, 2.0, 0.01) == (0.0, 1.0, False)
    t, b, rev = ctrl.step(-1.0, 1.0, 0.01)                # still rolling forward: brake before reversing
    assert (t, b) == (0.0, 1.0)
    t, b, rev = ctrl.step(-1.0, 0.0, 0.01)
    assert rev and t > 0 and b == 0.0


def test_pure_rotation_request_becomes_a_forward_crawl():
    ctrl = cb.SpeedController()
    throttle, steer, brake, reverse = cb.twist_to_commands(0.0, 0.8, 0.0, ctrl, 2.9, math.radians(70), 0.01)
    assert throttle > 0 and steer < 0 and not reverse


def test_self_check_flags_missing_gravity_and_bad_gyro_sign():
    msgs = []
    chk = cb.SensorSelfCheck(msgs.append)
    for _ in range(450):
        chk.imu(0.0, 0.0, 0.0, 0.0)                       # accelerometer without gravity
    assert any('gravity' in m for m in msgs)
    msgs.clear()
    chk = cb.SensorSelfCheck(msgs.append)
    for _ in range(25):
        chk.imu(9.8, -0.5, 0.5, 1.0)                      # gyro opposite to the real yaw rate
    assert any('opposite' in m for m in msgs)
    msgs.clear()
    chk = cb.SensorSelfCheck(msgs.append)
    for _ in range(25):
        chk.imu(9.8, 0.5, 0.5, 1.0)
    assert any('ok' in m for m in msgs)


def test_self_check_ignores_a_startup_transient():
    msgs = []
    chk = cb.SensorSelfCheck(msgs.append)
    chk.imu(55308.0, 0.0, 0.0, 0.0)                       # first sample after spawn
    for _ in range(449):
        chk.imu(9.81, 0.0, 0.0, 0.0)
    assert msgs == ['IMU gravity check ok (|a| = 9.81 m/s^2)']
