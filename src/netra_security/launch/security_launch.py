"""Launch file for netra_security nodes."""

from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare("netra_security"), "config", "security_params.yaml"])

    return LaunchDescription(
        [
            Node(
                package="netra_security",
                executable="secoc_can_driver",
                name="secoc_can_driver",
                parameters=[config],
                output="screen",
            ),
            Node(
                package="netra_security",
                executable="tamper_monitor",
                name="tamper_monitor",
                parameters=[config],
                output="screen",
            ),
        ]
    )
