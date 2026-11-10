"""Launch file for NETRA-UGV Planning & Safety Stack."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("netra_planning"), "config", "planner_params.yaml"])

    return LaunchDescription(
        [
            Node(
                package="netra_planning",
                executable="failsafe_manager",
                name="failsafe_manager",
                parameters=[config],
                output="screen",
            ),
            Node(
                package="netra_planning",
                executable="planner_node",
                name="netra_planner",
                parameters=[config],
                output="screen",
            ),
        ]
    )
