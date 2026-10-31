"""NETRA-UGV on a running CARLA 0.9.16 server (start the server first, see sim/README.md).

    ros2 launch netra_sim carla.launch.py map:=Mine_01 weather:=ClearSunset
    ros2 launch netra_sim carla.launch.py stack:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

import netra_sim
from netra_sim.launch_common import LOCALIZATION_ARGS, stack_actions


def generate_launch_description():
    pkg = get_package_share_directory('netra_sim')
    with open(os.path.join(pkg, 'models', 'netra_ugv', 'netra_ugv.urdf'), 'r') as f:
        robot_desc = f.read()

    args = [
        DeclareLaunchArgument('host', default_value='localhost'),
        DeclareLaunchArgument('port', default_value='2000'),
        DeclareLaunchArgument('map', default_value='', description="e.g. Mine_01 (off-road); '' keeps the loaded map"),
        DeclareLaunchArgument('weather', default_value='ClearNoon',
                              description='carla.WeatherParameters preset, e.g. ClearNight, HardRainNoon, DustStorm'),
        DeclareLaunchArgument('backend', default_value='inprocess',
                              description="'inprocess' = carla 0.9.16 client in ROS's Python; 'sidecar' = other client via carla_sidecar"),
        DeclareLaunchArgument('sidecar_python', default_value='',
                              description='python of the CARLA client env for the sidecar, e.g. sim/carla_env/bin/python (0.9.15)'),
        DeclareLaunchArgument('sidecar_port', default_value='5555'),
        DeclareLaunchArgument('autopilot', default_value='false', description='CARLA autopilot drives (ignores /cmd_vel)'),
        DeclareLaunchArgument('npc_vehicles', default_value='0'),
        DeclareLaunchArgument('weather_cycle', default_value='', description="e.g. 'ClearNoon,ClearSunset,HardRainNoon,ClearNight'"),
        DeclareLaunchArgument('record_dir', default_value='', description='save a dataset (rgb/depth/semantic/meta) under this folder'),
        DeclareLaunchArgument('record_every_s', default_value='0.5'),
        DeclareLaunchArgument('record_max', default_value='0'),
        DeclareLaunchArgument('image_width', default_value='1280'),
        DeclareLaunchArgument('image_height', default_value='720'),
        DeclareLaunchArgument('fx', default_value='500.0', description='focal length px; keep fx = 500*width/1280 for the same FOV'),
        DeclareLaunchArgument('mission', default_value='true'),
    ]

    bridge = Node(
        package='netra_sim', executable='carla_bridge', output='screen',
        parameters=[{
            'host': LaunchConfiguration('host'),
            'port': LaunchConfiguration('port'),
            'map': LaunchConfiguration('map'),
            'weather': LaunchConfiguration('weather'),
            'backend': LaunchConfiguration('backend'),
            'image_width': ParameterValue(LaunchConfiguration('image_width'), value_type=int),
            'image_height': ParameterValue(LaunchConfiguration('image_height'), value_type=int),
            'fx': ParameterValue(LaunchConfiguration('fx'), value_type=float),
            'autopilot': ParameterValue(LaunchConfiguration('autopilot'), value_type=bool),
            'npc_vehicles': ParameterValue(LaunchConfiguration('npc_vehicles'), value_type=int),
            'weather_cycle': ParameterValue(LaunchConfiguration('weather_cycle'), value_type=str),
            'record_dir': ParameterValue(LaunchConfiguration('record_dir'), value_type=str),
            'record_every_s': ParameterValue(LaunchConfiguration('record_every_s'), value_type=float),
            'record_max': ParameterValue(LaunchConfiguration('record_max'), value_type=int),
            'sidecar_port': ParameterValue(LaunchConfiguration('sidecar_port'), value_type=int),
        }])

    sidecar = ExecuteProcess(
        cmd=[LaunchConfiguration('sidecar_python'), '-m', 'netra_sim.carla_sidecar',
             '--port', LaunchConfiguration('sidecar_port')],
        additional_env={'PYTHONPATH': os.path.dirname(os.path.dirname(os.path.abspath(netra_sim.__file__)))},
        output='screen', condition=IfCondition(PythonExpression(["'", LaunchConfiguration('backend'), "' == 'sidecar'"])))

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])

    disparity = Node(package='netra_sim', executable='depth_to_disparity', output='screen',
                     parameters=[{'use_sim_time': True}])

    mission = Node(package='netra_sim', executable='mission_manager', output='screen',
                   parameters=[{'use_sim_time': True}], condition=IfCondition(LaunchConfiguration('mission')))

    return LaunchDescription(args + LOCALIZATION_ARGS + [sidecar, bridge, rsp, disparity, mission] + stack_actions())
