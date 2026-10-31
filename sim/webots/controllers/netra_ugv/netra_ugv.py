"""Netra UGV Webots controller (behavioural model of the Netra-UGV pipeline).
perception : lidar (obstacles) + downward lidar (negative-obstacle / ground-loss check)
localization: GPS + IMU (stand-in for OpenVINS EKF)
planning   : goal-seeking with lidar avoidance, roll/shock guards
security   : HMAC-authenticated waypoint commands (SecOC-style), tamper -> key zeroization
Keys: T = physical tamper (zeroize)   S = inject spoofed command   R = re-provision keys
"""
import math, hmac, hashlib, os, sys
from controller import Robot, Keyboard

# shared mission definition (sim/netra_sim/scenario.py)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from netra_sim import scenario

robot = Robot()
dt_ms = int(robot.getBasicTimeStep())   # follow the world's WorldInfo.basicTimeStep
kb = robot.getKeyboard(); kb.enable(dt_ms)
def dev(n, en=True):
    d = robot.getDevice(n)
    if en and hasattr(d, "enable"): d.enable(dt_ms)
    return d
gps, imu, gyro, acc = dev("gps"), dev("imu"), dev("gyro"), dev("accel")
cam = dev("camera"); lid = dev("lidar"); lidd = dev("lidar_down")
motors = [robot.getDevice(n) for n in ("front left wheel", "back left wheel", "front right wheel", "back right wheel")]
for m in motors: m.setPosition(float("inf")); m.setVelocity(0)
def drive(l, r):
    for i, m in enumerate(motors): m.setVelocity(l if i < 2 else r)

# ---- security: signed mission waypoints ----
KEY = bytearray(b"netra-demo-key-do-not-use-in-prod")
def sign(msg, key=None): return hmac.new(bytes(key or KEY), msg.encode(), hashlib.sha256).hexdigest()
WAYPOINTS = scenario.WAYPOINTS
signed = [(wp, sign("%d,%d" % (round(wp[0]), round(wp[1])))) for wp in WAYPOINTS]
def verify(wp, tag): return KEY and hmac.compare_digest(sign("%d,%d" % (round(wp[0]), round(wp[1]))), tag)
zeroized, ok_frames, bad_frames = False, 0, 0

# ---- tuning ----
CRUISE, TURN_GAIN = 5.0, 3.0
ROLL_SLOW, ROLL_STOP = 0.30, 0.50      # rad
SHOCK_STOP = 25.0                      # m/s^2 above gravity
DOWN_EXPECT = 0.75                     # m ground range straight ahead is checked against ratio
hres, dres = lid.getHorizontalResolution(), lidd.getHorizontalResolution()
lfov, dfov = lid.getFov(), lidd.getFov()
idx, wp_i, state, hold = 0, 0, "NAVIGATING", 0.0
def sector(r, lo, hi):
    n = len(r); vals = [x for x in r[int(lo*n):int(hi*n)] if x == x and x > 0.05]
    return min(vals) if vals else 99.0
def wrap(a): return math.atan2(math.sin(a), math.cos(a))
ground_ref = None

while robot.step(dt_ms) != -1:
    k = kb.getKey()
    while k != -1:
        if k in (ord("T"), ord("t")) and not zeroized:
            for i in range(len(KEY)): KEY[i] = 0
            zeroized = True; print("[SEC] Tamper detected: keys zeroized, drive inhibited")
        elif k in (ord("S"), ord("s")):
            if not verify((99, 99), "deadbeef"):
                bad_frames += 1; print("[SEC] Spoofed frame rejected (MAC mismatch) total=%d" % bad_frames)
        elif k in (ord("R"), ord("r")):
            KEY[:] = b"netra-demo-key-do-not-use-in-prod"; zeroized = False; print("[SEC] Keys re-provisioned")
        k = kb.getKey()
    if zeroized: drive(0, 0); continue

    x, y, _ = gps.getValues(); roll, pitch, yaw = imu.getRollPitchYaw()
    ax, ay, az = acc.getValues(); shock = abs(math.sqrt(ax*ax + ay*ay + az*az) - 9.81)
    if wp_i >= len(signed): drive(0, 0); state = "ARRIVED"; continue
    wp, tag = signed[wp_i]
    if not verify(wp, tag): bad_frames += 1; drive(0, 0); state = "AUTH FAIL"; continue
    ok_frames += 1
    gx, gy = wp
    if math.hypot(gx - x, gy - y) < 1.2:
        wp_i += 1; print("[NAV] waypoint %d reached" % wp_i); continue

    # perception: obstacle sectors (left / centre / right) from horizontal lidar
    r = lid.getRangeImage()
    left, centre, right = sector(r, 0.65, 1.0), sector(r, 0.35, 0.65), sector(r, 0.0, 0.35)
    # assumes range-image index 0 = right side of the robot (ENU); swap left/right above if your build differs
    # negative-obstacle check: downward lidar centre should return roughly the ground range
    d = lidd.getRangeImage(); dc = sector(d, 0.35, 0.65)
    d_ok = [v for v in d[int(.35*len(d)):int(.65*len(d))] if v == v]
    far = (not d_ok) or all(v > 8.0 or v == float("inf") for v in d_ok)
    if ground_ref is None and d_ok and d_ok[0] < 8: ground_ref = dc
    neg = far or (ground_ref and dc > ground_ref * 1.9)

    # planning: heading to goal + avoidance
    err = wrap(math.atan2(gy - y, gx - x) - yaw)
    w = max(-1.0, min(1.0, err * 1.5)); v = CRUISE * max(0.25, math.cos(err))
    state = "NAVIGATING"
    if centre < 2.2 or left < 1.2 or right < 1.2 or neg:
        # steer toward the more open side
        w = 1.0 if left > right else -1.0
        v = 0.5 if (centre < 1.0 or neg) else 2.0
        state = "NEG-OBSTACLE AVOID" if neg else "AVOIDING"
        hold = 0.6 if neg else hold
    if hold > 0: hold -= dt_ms / 1000.0; w = 1.0 if left > right else -1.0; v = min(v, 1.0)
    # safety guards
    tilt = max(abs(roll), abs(pitch))
    if tilt > ROLL_SLOW: v *= 0.4; state = "ROLL GUARD"
    if tilt > ROLL_STOP or shock > SHOCK_STOP: v = 0; w = 0; state = "SAFE STOP"
    L, R_ = v - w * TURN_GAIN, v + w * TURN_GAIN
    drive(L, R_)
