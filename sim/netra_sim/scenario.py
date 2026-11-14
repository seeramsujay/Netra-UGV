"""Single source of truth for the NETRA tactical test field.

Every simulator backend (Webots, Gazebo, headless render, CARLA mission goals)
builds its world from this module so terrain, ditches, boulders and waypoints
cannot drift apart. Standard library only, so it also imports inside Webots'
bundled Python.

World frame: ENU, metres, z up. The field is 50 m x 36 m centred on the origin.
"""

import math
import random
from dataclasses import dataclass

# Terrain grid. Vertex (i, j) sits at (X0 + i*SP, Y0 + j*SP); this matches the
# corner-anchored layout of a Webots ElevationGrid.
NX, NY, SP = 126, 91, 0.4
X0, Y0 = -25.0, -18.0
X1, Y1 = X0 + (NX - 1) * SP, Y0 + (NY - 1) * SP

DITCH_DEPTH = 0.55
SEED = 7
ROCK_COUNT = 28

# (cx, cy, radius, amplitude)
HILLS = [(-5, 2, 3.5, 1.1), (8, -6, 4, 1.4), (14, 10, 3, 1.0), (-14, -8, 4, 0.9), (0, 12, 3.5, 1.2), (20, -2, 3, 1.0)]
# (cx, cy, width_x, length_y): axis-aligned negative obstacles
DITCHES = [(-8, -4, 1.3, 8), (4, 3, 10, 1.3), (15, -12, 1.3, 7), (-2, -13, 8, 1.3)]

START = (-22.0, -14.0, 0.6)  # x, y, yaw
GOAL = (21.0, 13.0)
# Integer coordinates: the Webots controller HMAC-signs them as "%d,%d".
WAYPOINTS = [(-12, -10), (-3, -9), (6, -5), (12, -4), (21, 13)]


@dataclass(frozen=True)
class Rock:
    x: float
    y: float
    size: float
    yaw: float

    @property
    def dims(self):
        """Box extents (length, width, height)."""
        return (self.size * 1.6, self.size * 1.3, self.size * 0.9)


def in_ditch(x, y, pad=0.0):
    return any(abs(x - cx) < w / 2 + pad and abs(y - cy) < h / 2 + pad for cx, cy, w, h in DITCHES)


def terrain_height(x, y, ditches=True):
    z = sum(a * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (r / 1.6) ** 2)) for cx, cy, r, a in HILLS)
    z += 0.03 * math.sin(x * 1.7) * math.cos(y * 1.3)
    # keep the start area flat
    z *= min(1.0, max(0.0, (math.hypot(x - START[0], y - START[1]) - 4) / 4))
    if ditches and in_ditch(x, y):
        z -= DITCH_DEPTH
    return z


def grid_xs():
    return [X0 + i * SP for i in range(NX)]


def grid_ys():
    return [Y0 + j * SP for j in range(NY)]


def height_grid():
    """Row-major heights, index j * NX + i."""
    return [terrain_height(x, y) for y in grid_ys() for x in grid_xs()]


def ground_z(x, y, grid=None):
    """Bilinear ground height of the sampled grid at (x, y)."""
    grid = grid or height_grid()
    fx = min(NX - 1.0, max(0.0, (x - X0) / SP))
    fy = min(NY - 1.0, max(0.0, (y - Y0) / SP))
    i, j = min(int(fx), NX - 2), min(int(fy), NY - 2)
    tx, ty = fx - i, fy - j
    h00, h10 = grid[j * NX + i], grid[j * NX + i + 1]
    h01, h11 = grid[(j + 1) * NX + i], grid[(j + 1) * NX + i + 1]
    return (h00 * (1 - tx) + h10 * tx) * (1 - ty) + (h01 * (1 - tx) + h11 * tx) * ty


def build_rocks():
    rng = random.Random(SEED)
    rocks = []
    while len(rocks) < ROCK_COUNT:
        x, y = rng.uniform(-22, 22), rng.uniform(-16, 16)
        if math.hypot(x - START[0], y - START[1]) < 5 or math.hypot(x - GOAL[0], y - GOAL[1]) < 3:
            continue
        if in_ditch(x, y, pad=1.0):
            continue
        rocks.append(Rock(x, y, rng.uniform(0.25, 0.5), rng.uniform(0, 6.28)))
    return rocks


def in_rock(x, y, rocks, pad=0.0):
    return any(math.hypot(x - r.x, y - r.y) < r.size * 0.85 + pad for r in rocks)
