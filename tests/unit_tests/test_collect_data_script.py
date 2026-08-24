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

import os
import subprocess
from pathlib import Path


def test_collect_data_uses_configured_python(tmp_path):
    repo_path = Path(__file__).resolve().parents[2]
    fake_python = tmp_path / "python"
    captured_args = tmp_path / "args"
    fake_python.write_text(
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > {captured_args!s}\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env["RLINF_PYTHON"] = str(fake_python)

    result = subprocess.run(
        [
            "bash",
            str(repo_path / "examples/embodiment/collect_data.sh"),
            "test_config.yaml",
            "runner.num_data_episodes=3",
        ],
        cwd=repo_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"Using Python at {fake_python}" in result.stdout
    args = captured_args.read_text(encoding="utf-8").splitlines()
    assert args[0] == str(repo_path / "examples/embodiment/collect_real_data.py")
    assert "--config-name" in args
    assert "test_config.yaml" in args
    assert "runner.num_data_episodes=3" in args


def test_run_collect_data_rejects_invalid_episode_count_before_cleanup():
    repo_path = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        ["bash", str(repo_path / "run_collect_data.sh"), "0"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "positive_episode_count" in result.stderr


def test_run_collect_data_sources_ros_before_launch(tmp_path):
    repo_path = Path(__file__).resolve().parents[2]
    setup_files = []
    for index in range(3):
        setup_file = tmp_path / f"setup_{index}.bash"
        setup_file.write_text(
            f"export TEST_ROS_SETUP_{index}=loaded\n", encoding="utf-8"
        )
        setup_files.append(setup_file)
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "[[ ${TEST_ROS_SETUP_0:-} == loaded ]] && "
        "[[ ${TEST_ROS_SETUP_1:-} == loaded ]] && "
        "[[ ${TEST_ROS_SETUP_2:-} == loaded ]]\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "RLINF_ROS_SETUP": str(setup_files[0]),
            "RLINF_FRANKA_ROS2_SETUP": str(setup_files[1]),
            "RLINF_ROS2_SETUP": str(setup_files[2]),
            "RLINF_PYTHON": str(fake_python),
            "RLINF_COLLECT_PREFLIGHT_ONLY": "1",
        }
    )

    result = subprocess.run(
        ["bash", str(repo_path / "run_collect_data.sh"), "3"],
        cwd=repo_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
