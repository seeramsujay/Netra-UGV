"""Headless kinematic replay of the NETRA tactical mission (no simulator required).

This is a 2D behavioural model of the Webots controller, not a physics run: it
shares the terrain, boulders, ditches and waypoints with every simulator
backend through netra_sim.scenario, and renders the trajectory + HUD.

    python3 sim/webots/render.py [--out DIR]
"""
import argparse, math, os, shutil, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from netra_sim import scenario as sc

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=HERE, help="output directory (default: next to this script)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

NX, NY, SP, OX, OY = sc.NX, sc.NY, sc.SP, sc.X0, sc.Y0
ditches = sc.DITCHES
Z = np.array(sc.height_grid()).reshape(NY, NX)
rocks = [(r.x, r.y, r.size, r.yaw) for r in sc.build_rocks()]

def hg(x, y):
    return sc.terrain_height(x, y, ditches=False)

def hz(x, y):
    i = min(NX - 1, max(0, round((x - OX) / SP)))
    j = min(NY - 1, max(0, round((y - OY) / SP)))
    return Z[j, i]

def in_ditch(x, y):
    return sc.in_ditch(x, y)

def in_rock(x, y, pad=0):
    return any(math.hypot(x - rx, y - ry) < s * 0.85 + pad for rx, ry, s, _ in rocks)

def ray(x, y, a, maxr=12):
    for r in np.arange(0.2, maxr, 0.15):
        px, py = x + r * math.cos(a), y + r * math.sin(a)
        if in_rock(px, py) or abs(px) > 25 or abs(py) > 18:
            return r
    return 99

WP = sc.WAYPOINTS
x, y, yaw = sc.START
v = 0.0
w_ = 0.0
wi = 0
hold = 0
dt = 0.05
frames = []
events = []
t = 0.0
state = "NAVIGATING"
dist = 0.0
avoid = 0
zero = False
bad = 0
arrived_t = None
wrap = lambda a: math.atan2(math.sin(a), math.cos(a))

while t < 200:
    ev = ""
    if int(round(t / dt)) == int(18 / dt):
        bad += 1
        ev = "SPOOFED CAN FRAME REJECTED (SecOC MAC mismatch)"
    if arrived_t and t > arrived_t + 4 and not zero:
        zero = True
        ev = "TAMPER: KEYS ZEROIZED, DRIVE LOCKED"
    if zero:
        v = 0
    elif wi >= len(WP):
        v = 0
        state = "ARRIVED"
        if arrived_t is None:
            arrived_t = t
    else:
        gx, gy = WP[wi]
        if math.hypot(gx - x, gy - y) < 1.2:
            wi += 1
            continue
        L = min(ray(x, y, yaw + a) for a in np.linspace(0.35, 1.2, 6))
        R = min(ray(x, y, yaw - a) for a in np.linspace(0.35, 1.2, 6))
        C = min(ray(x, y, yaw + a) for a in np.linspace(-0.3, 0.3, 7))
        # downward lidar: ground loss ~2 m ahead
        neg = any(in_ditch(x + d * math.cos(yaw + a), y + d * math.sin(yaw + a)) for d in (1.6, 2.2) for a in (-0.15, 0, 0.15))
        err = wrap(math.atan2(gy - y, gx - x) - yaw)
        w = max(-1, min(1, err * 1.5))
        vt = 1.3 * max(0.25, math.cos(err))
        state = "NAVIGATING"
        if C < 2.2 or L < 1.2 or R < 1.2 or neg:
            w = 1.0 if L > R else -1.0
            vt = 0.15 if (C < 1 or neg) else 0.5
            state = "NEG-OBSTACLE AVOID" if neg else "AVOIDING"
            avoid += 1
            if neg:
                hold = 0.6
        if hold > 0:
            hold -= dt
            w = 1.0 if L > R else -1.0
            vt = min(vt, 0.3)
        slope = abs(hg(x + 0.4, y) - hg(x - 0.4, y)) / 0.8
        roll = slope
        if roll > 0.3:
            vt *= 0.4
            state = "ROLL GUARD"
        v += (vt - v) * 0.3
        w_ += (w * 1.0 - w_) * 0.3
        yaw += w_ * dt
        nx, ny = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt
        if not (in_rock(nx, ny, 0.2) or in_ditch(nx, ny)):
            dist += math.hypot(nx - x, ny - y)
            x, y = nx, ny
    if zero:
        state = "LOCKED"
    if ev:
        events.append((t, ev))
    frames.append((t, x, y, yaw, state, wi, dist, list(events[-1:]) if ev else [], zero, bad))
    t += dt
    if zero and t > arrived_t + 9:
        break

print(f"Simulation run complete: {len(frames)} frames, reached wp {wi}, final time: {round(t)}s")

# ---- Visual Output Rendering ----
fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0f1412")
ax = fig.add_axes([0.02, 0.04, 0.70, 0.92])
hud = fig.add_axes([0.74, 0.04, 0.25, 0.92])
hud.axis("off")

ax.imshow(Z, extent=[OX - SP / 2, OX + NX * SP - SP / 2, OY - SP / 2, OY + NY * SP - SP / 2], origin="lower", cmap="copper", alpha=0.9, vmin=-0.6, vmax=1.6)
for cx, cy, ww, hh in ditches:
    ax.add_patch(plt.Rectangle((cx - ww / 2, cy - hh / 2), ww, hh, fc="k", ec="#e5484d", lw=1.2))
for rx, ry, s, a in rocks:
    ax.add_patch(Circle((rx, ry), s * 0.85, fc="#8a8f8c", ec="#333"))
ax.add_patch(Circle(sc.GOAL, 0.8, fc="#f5d547", alpha=0.8))
ax.text(sc.GOAL[0], sc.GOAL[1] + 1.3, "GOAL", color="#f5d547", ha="center", fontsize=9)
ax.plot([sc.START[0]] + [p[0] for p in WP], [sc.START[1]] + [p[1] for p in WP], "--", color="#4ade80", lw=1, alpha=0.5)
for i, p in enumerate(WP):
    ax.plot(*p, "o", color="#4ade80", ms=4)

ax.set_xlim(-25, 25)
ax.set_ylim(-18, 18)
ax.set_aspect("equal")
ax.set_facecolor("#0c110f")
ax.tick_params(colors="#888")
ax.set_title("Netra UGV — Tactical Mission Model (headless kinematic replay) Trajectory & HUD", color="#e4ebe6", fontsize=11)

trail, = ax.plot([], [], color="#5fcf8a", lw=2)
body = Polygon([[0, 0]] * 4, closed=True, fc="#5fcf8a", ec="w", zorder=5)
ax.add_patch(body)
cone = Polygon([[0, 0]] * 3, closed=True, fc="#50a0ff", alpha=0.15, zorder=4)
ax.add_patch(cone)

tx = [f[1] for f in frames]
ty = [f[2] for f in frames]
trail.set_data(tx, ty)

last_f = frames[-1]
t_end, x_end, y_end, yaw_end, st_end, wi_end, dist_end, evs_end, zero_end, bad_end = last_f
c, s = math.cos(yaw_end), math.sin(yaw_end)
P = np.array([[0.6, 0.3], [-0.6, 0.3], [-0.6, -0.3], [0.6, -0.3]])
body.set_xy([[x_end + px * c - py * s, y_end + px * s + py * c] for px, py in P])
body.set_facecolor("#e5484d" if zero_end else "#5fcf8a")
cone.set_xy([[x_end, y_end], [x_end + 10 * math.cos(yaw_end + 1.2), y_end + 10 * math.sin(yaw_end + 1.2)], [x_end + 10 * math.cos(yaw_end - 1.2), y_end + 10 * math.sin(yaw_end - 1.2)]])

all_logs = []
for f in frames:
    for e in f[7]:
        all_logs.append("t=%3.0fs %s" % (e[0], e[1]))

hud_text = f"NETRA UGV TELEMETRY\n\nTime:     {t_end:5.1f} s\nState:    {st_end}\nWaypoint: {min(wi_end + 1, 5)} / 5\nDistance: {dist_end:5.1f} m\nAltitude: {hz(x_end, y_end):+.2f} m\nCAN Bad:  {bad_end}\nKeys:     {'ZEROIZED' if zero_end else 'ARMED'}"
hud.text(0, 1.0, hud_text, color="#e4ebe6", va="top", fontsize=10, family="monospace")

log_str = "EVENT AUDIT LOG\n" + "\n".join(l[:32] for l in all_logs[-6:])
hud.text(0, 0.42, log_str, color="#9fb0a6", va="top", fontsize=9, family="monospace")

banner = ax.text(0, 0, "KEYS ZEROIZED — SYSTEM LOCKED", color="w", fontsize=14, ha="center", va="center", weight="bold", zorder=9, bbox=dict(fc="#b02a2a", alpha=0.85, ec="none"))
banner.set_position((x_end, y_end + 3 if y_end < 12 else y_end - 3))

summary_plot_path = os.path.join(args.out, "netra_ugv_sim_trajectory.png")
fig.savefig(summary_plot_path, dpi=120)
print(f"[OK] Saved mission trajectory summary plot: {summary_plot_path}")

out_mp4 = os.path.join(args.out, "netra_ugv_sim.mp4")
has_ffmpeg = shutil.which("ffmpeg") is not None

if has_ffmpeg:
    try:
        from matplotlib.animation import FFMpegWriter
        w = FFMpegWriter(fps=20, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        tx, ty = [], []
        step = 2
        with w.saving(fig, out_mp4, 100):
            for k in range(0, len(frames), step):
                t_k, x_k, y_k, yaw_k, st_k, wi_k, dist_k, evs_k, zero_k, bad_k = frames[k]
                tx.append(x_k); ty.append(y_k); trail.set_data(tx, ty)
                c_k, s_k = math.cos(yaw_k), math.sin(yaw_k)
                body.set_xy([[x_k + px * c_k - py * s_k, y_k + px * s_k + py * c_k] for px, py in P])
                body.set_facecolor("#e5484d" if zero_k else "#5fcf8a")
                cone.set_xy([[x_k, y_k], [x_k + 10 * math.cos(yaw_k + 1.2), y_k + 10 * math.sin(yaw_k + 1.2)], [x_k + 10 * math.cos(yaw_k - 1.2), y_k + 10 * math.sin(yaw_k - 1.2)]])
                w.grab_frame()
        print(f"[OK] Saved MP4 animation: {out_mp4}")
    except Exception as e:
        print(f"[NOTE] MP4 export skipped ({e}). Summary trajectory image was generated successfully.")
else:
    print("[NOTE] ffmpeg not installed. Full mission trajectory image generated successfully.")
