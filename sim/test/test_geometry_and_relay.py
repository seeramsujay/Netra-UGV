import math

import pytest
from netra_sim import scenario
from netra_sim.geometry import quaternion_to_rpy, rpy_to_quaternion, world_to_start_frame
from netra_sim.odom_relay import rebase_odometry_pose


def test_quaternion_roundtrip():
    for rpy in [(0.1, -0.2, 0.7), (0.0, 0.0, -3.0), (-0.4, 0.3, 2.9)]:
        assert quaternion_to_rpy(*rpy_to_quaternion(*rpy)) == pytest.approx(rpy)


def test_start_pose_maps_to_origin_and_goal_is_distance_preserving():
    sx, sy, syaw = scenario.START
    assert world_to_start_frame(sx, sy, scenario.START) == pytest.approx((0.0, 0.0))
    gx, gy = scenario.GOAL
    x, y = world_to_start_frame(gx, gy, scenario.START)
    assert math.hypot(x, y) == pytest.approx(math.hypot(gx - sx, gy - sy))
    # a point straight ahead of the spawn heading is on the +x axis of the start frame
    ahead = (sx + 4 * math.cos(syaw), sy + 4 * math.sin(syaw))
    assert world_to_start_frame(*ahead, scenario.START) == pytest.approx((4.0, 0.0))


def test_odom_relay_starts_at_identity_and_keeps_tilt():
    sx, sy, syaw = scenario.START
    q = rpy_to_quaternion(0.1, -0.05, syaw)
    (x, y, z), q_out = rebase_odometry_pose(sx, sy, 0.31, q, scenario.START, z0=0.31)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    roll, pitch, yaw = quaternion_to_rpy(*q_out)
    assert (roll, pitch, yaw) == pytest.approx((0.1, -0.05, 0.0), abs=1e-9)
