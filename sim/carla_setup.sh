#!/usr/bin/env bash
# Prepare a (college) Ubuntu machine to run the NETRA CARLA backend. Safe to re-run.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "== GPU";    nvidia-smi -L 2>/dev/null || echo "no NVIDIA GPU visible: CARLA needs one"
echo "== ROS 2";  [ -f /opt/ros/jazzy/setup.bash ] && echo "jazzy found" || echo "ROS 2 Jazzy missing (use 0.9.16 client only with Python 3.12)"
echo "== Docker"; docker --version 2>/dev/null || echo "docker missing (or use a CARLA 0.9.16 tarball)"
echo "== uv env for the CARLA 0.9.15 client (Python 3.10 sidecar)"
if command -v uv >/dev/null; then
  [ -d "$HERE/carla_env" ] || uv venv --python 3.10 "$HERE/carla_env"
  uv pip install --python "$HERE/carla_env/bin/python" carla==0.9.15 numpy opencv-python-headless
else echo "install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"; fi
WHEEL=$(ls "$HERE"/../carla_wheel/carla-0.9.16-*.whl 2>/dev/null | head -1)
if [ -n "$WHEEL" ]; then pip install --user --no-deps "$WHEEL"; else pip install --user carla==0.9.16; fi
python3 -c "import carla; print('carla client OK', carla.__file__)"
echo "Server:  docker pull carlasim/carla:0.9.16   (needs internet once, ~20 GB)"
echo "         docker run --rm --gpus all --net=host carlasim/carla:0.9.16 ./CarlaUE4.sh -RenderOffScreen"
echo "Then:    ros2 launch netra_sim carla.launch.py map:=Mine_01"
