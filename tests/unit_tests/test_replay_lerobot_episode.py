# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest

from examples.embodiment.replay_lerobot_episode import _normalize_gripper_actions


def test_gello_gripper_actions_switch_at_collection_boundary():
    actions = np.zeros((5, 16), dtype=np.float32)
    values = np.array([-0.6, -0.1, 0.0, 0.1, 0.6], dtype=np.float32)
    actions[:, 7] = values
    actions[:, 15] = values[::-1]

    normalized = _normalize_gripper_actions(actions, "gello")

    np.testing.assert_array_equal(normalized[:, 7], [-1, -1, -1, 1, 1])
    np.testing.assert_array_equal(normalized[:, 15], [1, 1, -1, -1, -1])
    np.testing.assert_array_equal(normalized[:, :7], actions[:, :7])
    np.testing.assert_array_equal(normalized[:, 8:15], actions[:, 8:15])


def test_policy_gripper_actions_keep_policy_threshold_semantics():
    actions = np.zeros((3, 16), dtype=np.float32)
    actions[:, 7] = [-1.2, -0.1, 1.2]
    actions[:, 15] = [1.1, 0.2, -1.1]

    normalized = _normalize_gripper_actions(actions, "policy")

    np.testing.assert_allclose(normalized[:, 7], [-1.0, -0.1, 1.0])
    np.testing.assert_allclose(normalized[:, 15], [1.0, 0.2, -1.0])


def test_unknown_gripper_action_source_is_rejected():
    with pytest.raises(ValueError, match="Unknown gripper action source"):
        _normalize_gripper_actions(np.zeros((1, 16)), "auto")
