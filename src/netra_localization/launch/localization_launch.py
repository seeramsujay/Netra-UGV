"""Launch file for netra_localization VIO node."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("netra_localization"), "config", "vio_params.yaml"])

    return LaunchDescription(
        [
            Node(
                package="netra_localization",
                executable="vio_node",
                name="vio_node",
                parameters=[config],
                output="screen",
            ),
        ]
    )
