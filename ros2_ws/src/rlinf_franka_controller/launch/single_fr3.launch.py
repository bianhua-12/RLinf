# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch one FR3 with a robot-driven ros2_control update loop."""

import xacro
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def build_control_parameters(controllers_yaml, robot_description, use_fake_hardware):
    """Build parameters for RLinf's controller manager executable."""
    return [
        controllers_yaml,
        {"robot_description": robot_description},
        {"load_gripper": False},
        {"robot_driven_loop": not use_fake_hardware},
    ]


def generate_robot_nodes(context):
    """Create standard Franka nodes with RLinf's control-loop executable."""
    use_fake_hardware_text = LaunchConfiguration("use_fake_hardware").perform(context)
    use_fake_hardware = use_fake_hardware_text.lower() == "true"
    namespace = LaunchConfiguration("namespace").perform(context)
    urdf_path = PathJoinSubstitution(
        [FindPackageShare("franka_description"), "robots", "fr3/fr3.urdf.xacro"]
    ).perform(context)
    robot_description = xacro.process_file(
        urdf_path,
        mappings={
            "ros2_control": "true",
            "arm_id": "fr3",
            "arm_prefix": LaunchConfiguration("arm_prefix").perform(context),
            "robot_ip": LaunchConfiguration("robot_ip").perform(context),
            "hand": "false",
            "use_fake_hardware": use_fake_hardware_text,
            "fake_sensor_commands": use_fake_hardware_text,
        },
    ).toprettyxml(indent="  ")
    controllers_yaml = PathJoinSubstitution(
        [FindPackageShare("rlinf_franka_controller"), "config", "controllers.yaml"]
    ).perform(context)

    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=namespace,
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="rlinf_franka_controller",
            executable="robot_driven_control_node",
            namespace=namespace,
            parameters=build_control_parameters(
                controllers_yaml, robot_description, use_fake_hardware
            ),
            remappings=[("joint_states", "franka/joint_states")],
            output="screen",
            on_exit=Shutdown(),
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            name="joint_state_publisher",
            namespace=namespace,
            parameters=[
                {
                    "source_list": ["franka/joint_states"],
                    "rate": 30,
                    "use_robot_description": False,
                }
            ],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=["joint_state_broadcaster"],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=[
                "joint_impedance_controller",
                "--controller-manager-timeout",
                "30",
            ],
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip"),
            DeclareLaunchArgument("namespace", default_value=""),
            DeclareLaunchArgument("arm_prefix", default_value=""),
            DeclareLaunchArgument("use_fake_hardware", default_value="false"),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
