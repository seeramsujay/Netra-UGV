# NETRA-UGV simulation (`netra_sim`)

Three backends, one test field. `netra_sim/scenario.py` defines the 50 x 36 m field
(hills, four 0.55 m ditches, 28 boulders, start, goal, waypoints) and every backend is built from it.

| Backend | What it is for | Status |
|---|---|---|
| **Gazebo Harmonic** (ROS 2 Jazzy) | Physics, ditches, full ROS stack loop | Verified end to end (see `exports/results`) |
| **CARLA 0.9.15** (sidecar) / 0.9.16 | Photorealistic camera/lighting/weather + dataset collection | Verified on a real 0.9.15 server; 0.9.16 in-process path only tested with a fake server |
| **Webots R2025a** | Original demo + SecOC key-zeroization keys (T/S/R) | Kept, patched; not runnable on the dev machine |

All backends expose the same ROS 2 interface: `/camera/image_raw`, `/camera/depth_image`, `/camera/camera_info`,
`/stereo/disparity`, `/imu/data`, `/odom_ground_truth`, `/clock`, and take `/cmd_vel`.

## Gazebo
```bash
source /opt/ros/jazzy/setup.bash
colcon build --base-paths src sim && source install/setup.bash
ros2 launch netra_sim gazebo.launch.py                       # GUI
ros2 launch netra_sim gazebo.launch.py headless:=true stack:=true   # + NETRA stack, ground-truth localization
ros2 run netra_sim smoke_test                                # checks the sensor/actuator contract
```
`localization:=vio` uses the MSCKF instead of ground truth (see open issues).
After editing `scenario.py`: `python3 sim/netra_sim/gen_gazebo_assets.py` and `python3 sim/webots/gen_world.py`.

## CARLA
Status: **verified on a real CARLA 0.9.15 server** (docker `carlasim/carla:0.9.15`) with the sidecar backend:
sensor contract passes in sim time, ego drives by autopilot, dataset is recorded.

Why a sidecar: ROS 2 Jazzy runs Python 3.12 but the CARLA 0.9.15 client only exists up to Python 3.10 (a 0.9.16 client
against a 0.9.15 server segfaults). `carla_sidecar` runs the CARLA client in a uv-managed 3.10 env and the ROS bridge
talks to it over 127.0.0.1. With a 0.9.16 server use `backend:=inprocess` and `pip install carla==0.9.16` instead.

```bash
# one time: the CARLA client env (uv: https://astral.sh/uv)
uv venv --python 3.10 sim/carla_env && uv pip install --python sim/carla_env/bin/python carla==0.9.15 numpy opencv-python-headless
# terminal 1: server, windowed (the off-screen mode segfaulted on this RTX 5060). Needs ~9 GB RAM free.
xhost +local:docker
docker run -d --name netra_carla --gpus all --net=host -e DISPLAY=:0 -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix carlasim/carla:0.9.15 ./CarlaUE4.sh -nosound -quality-level=Low
# terminal 2: first-person view + autopilot + traffic + weather cycling + dataset recording
ros2 launch netra_sim carla.launch.py backend:=sidecar sidecar_python:=$PWD/sim/carla_env/bin/python \
  autopilot:=true npc_vehicles:=10 weather_cycle:=ClearNoon,ClearSunset,HardRainNoon,ClearNight \
  record_dir:=$PWD/exports/carla_dataset record_every_s:=0.5 image_width:=640 image_height:=360 fx:=250.0 mission:=false
```
The CARLA window shows the car's own camera (`spectator:=first_person`). Dataset: `record_dir/run_*/{rgb,depth,semantic}/NNNNNN.*`
(PNG, float16 metres `.npy`, class-id PNG) plus `meta.jsonl` (pose, speed, IMU, weather) and `config.json`.
Stop with Ctrl-C in terminal 2 (the sidecar restores the server; it also handles SIGTERM). Keep fx = 500 x width / 1280.
Without `autopilot`, `/cmd_vel` drives the car (bicycle-model stand-in for the skid-steer command).
Notes: only Unreal Towns are in this image (no `Mine_01`; CARLA cannot carve ditches, so it tests perception realism).
At this load sim time runs at ~0.3x wall time. If the server dies with exit 137 it was OOM-killed: close other apps or lower resolution.
The IMU check logs `IMU gravity check ok` / `gyro sign check ok`; a warning means a frame conversion needs fixing.

## Issues found and fixed
- Gazebo: no IMU plugin (no `/imu/data`); bridge topic names wrong; no `/clock`; ditch was flush with the ground
  (not an obstacle); 1 boulder vs 28; camera fx 863 vs 500 px expected by perception; no disparity source;
  wrong base inertia; wheel friction made skid-steer unable to turn; dead-reckoned odom; Humble-only, unused deps.
- Webots: no `WorldInfo` (32 ms step vs 16 ms controller); duplicated terrain/waypoints with a mismatch (-9 vs -9.5);
  cwd-dependent paths and a hardcoded `/mnt/user-data` path in `render.py`. `render.py` is a kinematic replay, not Webots.
- Stack: VIO gravity sign (+9.81 with a z-up IMU: 157 m drift in 4 s at rest); VIO IMU queue depth 1 dropped
  most of the 500 Hz samples.

## Open issues (found, NOT fixed)
1. **VIO still diverges when moving** (visual update): identity camera-IMU extrinsics, inconsistent orientation
   convention between `propagate` and `augment_state`, Jacobian fills only clone blocks, no outlier gating.
2. **Negative-obstacle detector gives false positives**: committed hazard on 28/30 frames of flat ground (same as a
   real ditch). Horizon guard alone does not fix it (tested). Likely integer-quantised disparity near the horizon.
3. **Positive obstacles are invisible to the stack in Gazebo**: only the segmentation network sees them and it
   calls untextured sim terrain "vegetation"; the robot wedges against boulders. CARLA/textures should help.
4. Planner drives 5 m forward when no goal is set; Webots lidar left/right index order unverified.
