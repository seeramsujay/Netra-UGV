"""Frame maths shared by the simulator bridges (pure Python, no ROS)."""
import math


def wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def rpy_to_quaternion(roll, pitch, yaw):
    """ZYX Euler -> (x, y, z, w)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def quaternion_to_rpy(x, y, z, w):
    """(x, y, z, w) -> ZYX Euler roll, pitch, yaw."""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def anchor_pose(x, y, yaw, spawn, anchor):
    """Re-express a planar pose so that `spawn` coincides with `anchor` (x, y, yaw each).

    anchor = (0, 0, 0) gives the pose in the frame a VIO starts in; anchor =
    scenario.START re-bases an arbitrary simulator map onto the test field.
    """
    dx, dy = x - spawn[0], y - spawn[1]
    c, s = math.cos(-spawn[2]), math.sin(-spawn[2])
    lx, ly = dx * c - dy * s, dx * s + dy * c
    c, s = math.cos(anchor[2]), math.sin(anchor[2])
    return (anchor[0] + lx * c - ly * s, anchor[1] + lx * s + ly * c,
            wrap_angle(anchor[2] + (yaw - spawn[2])))


def world_to_start_frame(wx, wy, start):
    """A world point expressed in the frame anchored at the spawn pose (x, y, yaw)."""
    return anchor_pose(wx, wy, start[2], start, (0.0, 0.0, 0.0))[:2]
