"""NETRA-UGV in Gazebo Harmonic: tactical field + skid-steer UGV + ROS 2 bridge.

    ros2 launch netra_sim gazebo.launch.py                 # GUI
    ros2 launch netra_sim gazebo.launch.py headless:=true  # server only (CI / SSH)
    ros2 launch netra_sim gazebo.launch.py stack:=true     # also start the NETRA autonomy stack
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from netra_sim import scenario
from netra_sim.launch_common import LOCALIZATION_ARGS, stack_actions


def generate_launch_description():
    pkg = get_package_share_directory('netra_sim')
    urdf_path = os.path.join(pkg, 'models', 'netra_ugv', 'netra_ugv.urdf')
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()

    sx, sy, syaw = scenario.START
    spawn_z = scenario.ground_z(sx, sy) + 0.3     # drop onto the wheels

    args = [
        DeclareLaunchArgument('headless', default_value='false',
                              description='run the Gazebo server only, no GUI'),
        DeclareLaunchArgument('world', default_value=os.path.join(pkg, 'worlds', 'tactical_field.sdf')),
        DeclareLaunchArgument('mission', default_value='true',
                              description='publish the waypoint mission on /goal_pose'),
    ]

    gz_flags = PythonExpression([
        "('-r -s --headless-rendering -v2 ' if '", LaunchConfiguration('headless'),
        "' == 'true' else '-r -v2 ') + '", LaunchConfiguration('world'), "'"])

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': gz_flags, 'on_exit_shutdown': 'true'}.items())

    spawn = Node(
        package='ros_gz_sim', executable='create', output='screen',
        arguments=['-topic', 'robot_description', '-name', 'netra_ugv',
                   '-x', str(sx), '-y', str(sy), '-z', '%.3f' % spawn_z, '-Y', str(syaw)])

    rsp = Node(
        package='robot_state_publisher', executable='robot_state_publisher', output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}])

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge', output='screen',
        parameters=[{'config_file': os.path.join(pkg, 'config', 'gz_bridge.yaml'),
                     'use_sim_time': True}])

    disparity = Node(
        package='netra_sim', executable='depth_to_disparity', output='screen',
        parameters=[{'use_sim_time': True}])

    mission = Node(
        package='netra_sim', executable='mission_manager', output='screen',
        parameters=[{'use_sim_time': True}], condition=IfCondition(LaunchConfiguration('mission')))

    return LaunchDescription(
        args + LOCALIZATION_ARGS + [
            AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(pkg, 'models')),
            gz_sim, rsp, spawn, bridge, disparity, mission] + stack_actions())
