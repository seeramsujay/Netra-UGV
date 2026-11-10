"""Pieces shared by the simulator launch files."""

from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare

# (package, launch file) of the NETRA autonomy stack, in start-up order.
STACK_LAUNCHES = [
    ("netra_localization", "localization_launch.py"),
    ("netra_perception", "perception_launch.py"),
    ("netra_mapping", "mapping_launch.py"),
    ("netra_planning", "planning_launch.py"),
    ("netra_security", "security_launch.py"),
]

LOCALIZATION_ARGS = [
    DeclareLaunchArgument(
        "stack",
        default_value="false",
        description="also launch the NETRA localization/perception/mapping/planning/security nodes",
    ),
    DeclareLaunchArgument(
        "localization",
        default_value="ground_truth",
        description="'vio' = netra_localization MSCKF, 'ground_truth' = /odom from simulator truth",
    ),
]


def stack_actions():
    """The NETRA stack on simulated time. Packages are resolved lazily, so they only
    need to be built when the `stack` launch argument is true."""
    stack_on = IfCondition(LaunchConfiguration("stack"))
    vio = IfCondition(
        PythonExpression(
            ["'", LaunchConfiguration("stack"), "' == 'true' and '", LaunchConfiguration("localization"), "' == 'vio'"]
        )
    )
    truth = IfCondition(
        PythonExpression(
            [
                "'",
                LaunchConfiguration("stack"),
                "' == 'true' and '",
                LaunchConfiguration("localization"),
                "' == 'ground_truth'",
            ]
        )
    )

    def include(pkg, launch_file):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([FindPackageShare(pkg), "launch", launch_file]))
        )

    (loc_pkg, loc_file), rest = STACK_LAUNCHES[0], STACK_LAUNCHES[1:]
    return [
        GroupAction(condition=vio, actions=[SetParameter(name="use_sim_time", value=True), include(loc_pkg, loc_file)]),
        Node(
            package="netra_sim",
            executable="odom_relay",
            output="screen",
            parameters=[{"use_sim_time": True}],
            condition=truth,
        ),
        GroupAction(
            condition=stack_on,
            actions=[SetParameter(name="use_sim_time", value=True)] + [include(p, f) for p, f in rest],
        ),
    ]
