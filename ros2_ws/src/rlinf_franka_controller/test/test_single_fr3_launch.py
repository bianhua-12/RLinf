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

import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext


@pytest.fixture(scope="module")
def launch_module():
    launch_path = Path(__file__).parents[1] / "launch" / "single_fr3.launch.py"
    spec = importlib.util.spec_from_file_location("single_fr3_launch", launch_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("use_fake_hardware", "robot_driven_loop"),
    [("false", True), ("true", False)],
)
def test_control_loop_mode_matches_hardware(
    launch_module, use_fake_hardware, robot_driven_loop
):
    context = LaunchContext()
    context.launch_configurations.update(
        robot_ip="127.0.0.1",
        namespace="test",
        arm_prefix="test",
        use_fake_hardware=use_fake_hardware,
    )

    nodes = launch_module.generate_robot_nodes(context)
    control_node = next(
        node for node in nodes if node.node_package == "rlinf_franka_controller"
    )

    assert control_node.node_executable == "robot_driven_control_node"
    parameters = launch_module.build_control_parameters(
        "controllers.yaml", "<robot/>", use_fake_hardware == "true"
    )
    assert {"robot_driven_loop": robot_driven_loop} in parameters
