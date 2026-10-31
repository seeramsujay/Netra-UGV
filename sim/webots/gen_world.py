"""Regenerate worlds/netra_tactical.wbt from the shared NETRA scenario.

Run from anywhere:  python3 sim/webots/gen_world.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from netra_sim import scenario as sc  # noqa: E402

PROTO = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects"
BASIC_TIME_STEP_MS = 16   # controller steps at the same period (see netra_ugv.py)

grid = sc.height_grid()
rocks = sc.build_rocks()

rows = "\n".join(
    "        " + " ".join("%.3f" % v for v in grid[j * sc.NX:(j + 1) * sc.NX])
    for j in range(sc.NY))

out = [
    "#VRML_SIM R2025a utf8",
    'EXTERNPROTO "%s/objects/backgrounds/protos/TexturedBackground.proto"' % PROTO,
    'EXTERNPROTO "%s/objects/backgrounds/protos/TexturedBackgroundLight.proto"' % PROTO,
    'EXTERNPROTO "%s/robots/adept/pioneer3/protos/Pioneer3at.proto"' % PROTO,
    "",
    "WorldInfo {\n  basicTimeStep %d\n}" % BASIC_TIME_STEP_MS,
    'Viewpoint {\n  orientation -0.25 0.25 0.93 1.6\n  position -34 -30 22\n'
    '  follow "Netra UGV"\n  followType "Mounted Shot"\n}',
    "TexturedBackground {}\nTexturedBackgroundLight {}",
    # The terrain translation is the grid corner; an ElevationGrid spans 0..(N-1)*SP.
    'Solid {\n  name "terrain"\n  translation %.3f %.3f 0\n  children [ Shape {\n'
    '    appearance PBRAppearance { baseColor 0.55 0.47 0.33 roughness 1 metalness 0 }\n'
    '    geometry DEF TERRAIN ElevationGrid {\n      xDimension %d\n      xSpacing %g\n'
    '      yDimension %d\n      ySpacing %g\n      height [\n%s\n      ]\n    }\n  } ]\n'
    '  boundingObject USE TERRAIN\n}' % (sc.X0, sc.Y0, sc.NX, sc.SP, sc.NY, sc.SP, rows),
]

for i, r in enumerate(rocks):
    lx, ly, lz = r.dims
    out.append(
        'Solid {\n  name "rock%d"\n  translation %.2f %.2f %.2f\n  rotation 0 0 1 %.2f\n'
        '  children [ Shape { appearance PBRAppearance { baseColor 0.4 0.4 0.42 roughness 1 metalness 0 }'
        ' geometry DEF R%d Box { size %.2f %.2f %.2f } } ]\n  boundingObject USE R%d\n}'
        % (i, r.x, r.y, sc.ground_z(r.x, r.y, grid) + r.size * 0.4, r.yaw, i, lx, ly, lz, i))

out.append(
    'Solid {\n  name "goal marker"\n  translation %g %g %.2f\n'
    '  children [ Shape { appearance PBRAppearance { baseColor 1 0.85 0.1 emissiveColor 0.6 0.5 0 }'
    ' geometry Cylinder { height 0.05 radius 0.8 } } ]\n}'
    % (sc.GOAL[0], sc.GOAL[1], sc.ground_z(*sc.GOAL, grid) + 0.03))

sx, sy, syaw = sc.START
out.append("""Pioneer3at {
  translation %g %g 0.3
  rotation 0 0 1 %g
  name "Netra UGV"
  controller "netra_ugv"
  supervisor FALSE
  extensionSlot [
    GPS { translation 0 0 0.3 name "gps" }
    InertialUnit { translation 0 0 0.3 name "imu" }
    Gyro { translation 0 0 0.3 name "gyro" }
    Accelerometer { translation 0 0 0.3 name "accel" }
    Camera { translation 0.3 0 0.35 name "camera" width 320 height 240 fieldOfView 1.2 }
    Lidar { translation 0.3 0 0.35 name "lidar" horizontalResolution 180 fieldOfView 2.4 numberOfLayers 1 maxRange 12 minRange 0.2 }
    Lidar { translation 0.3 0 0.4 rotation 0 1 0 0.35 name "lidar_down" horizontalResolution 45 fieldOfView 0.9 numberOfLayers 1 maxRange 8 minRange 0.2 }
  ]
}""" % (sx, sy, syaw))

path = os.path.join(HERE, "worlds", "netra_tactical.wbt")
with open(path, "w") as f:
    f.write("\n\n".join(out) + "\n")
print("wrote %s (%d rocks)" % (path, len(rocks)))
