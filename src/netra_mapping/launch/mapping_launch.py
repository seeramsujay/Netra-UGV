"""Launch file for netra_mapping costmap node."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("netra_mapping"), "config", "costmap_params.yaml"])
    return LaunchDescription(
        [
            Node(
                package="netra_mapping",
                executable="costmap_node",
                name="costmap_node",
                parameters=[config],
                output="screen",
            ),
        ]
    )
