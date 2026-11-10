"""Launch file for netra_perception node."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("netra_perception"), "config", "perception_params.yaml"])

    return LaunchDescription(
        [
            Node(
                package="netra_perception",
                executable="perception_node",
                name="perception_node",
                parameters=[config],
                output="screen",
            ),
        ]
    )
