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
import sys

from omegaconf import OmegaConf
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rlinf.models.embodiment.reward.vlm_reward_model import HistoryVLMRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    HistoryVLMInputBuilder,
)
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.nested_dict_process import cat_list_of_dict_tensor, split_dict


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def eval(self):
        return self

    def generate(
        self, input_ids: torch.Tensor, reward_ids: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        del kwargs
        return torch.cat([input_ids, reward_ids.unsqueeze(-1)], dim=-1)


class _FakeProcessor:
    def batch_decode(self, output_ids: torch.Tensor, skip_special_tokens: bool = True):
        del skip_special_tokens
        return [str(int(token.item())) for token in output_ids[:, 0]]


class _FakeRewardParser:
    def parse_rewards(self, outputs: list[str]) -> torch.Tensor:
        return torch.tensor([float(output) for output in outputs], dtype=torch.float32)


class _FakeHistoryInputBuilder(HistoryVLMInputBuilder):
    def __init__(self, history_buffer_names: list[str]):
        super().__init__(
            _processor=None,
            history_buffer_names=history_buffer_names,
        )
        self.calls: list[list[int]] = []

    def get_valid_input_ids(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
    ) -> list[int]:
        del observations
        history_window = history_input["history_window"]["main_images"]
        return [
            env_idx
            for env_idx, frames in enumerate(history_window)
            if len(frames) > 0
        ]

    def prepare_inputs(
        self,
        observations: dict[str, object],
        history_input: dict[str, dict[str, list[list[object]]]],
        valid_input_ids: list[int],
    ) -> dict[str, torch.Tensor]:
        del history_input
        reward_ids = observations["slot_ids"][valid_input_ids].to(dtype=torch.long)
        self.calls.append(reward_ids.tolist())
        return {
            "input_ids": torch.zeros((len(valid_input_ids), 1), dtype=torch.long),
            "reward_ids": reward_ids,
        }

    def process_inputs(
        self, prepared_inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return prepared_inputs


class _TestHistoryVLMRewardModel(HistoryVLMRewardModel):
    def setup_processor(self) -> None:
        self._processor = _FakeProcessor()

    def setup_model(self) -> None:
        self._model = _FakeModel()

    def setup_input_builder(self) -> None:
        self.input_builder = _FakeHistoryInputBuilder(
            history_buffer_names=self.history_buffer_names
        )

    def setup_reward_parser(self) -> None:
        self.reward_parser = _FakeRewardParser()


def _make_cfg(infer_micro_batch_size: int) -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": "dummy",
            "precision": "bf16",
            "infer_micro_batch_size": infer_micro_batch_size,
            "input_builder_name": "history_vlm_input_builder",
            "reward_parser_name": "base_reward_parser",
            "history_buffers": {
                "history_window": {
                    "history_size": 10,
                    "input_interval": 10,
                    "history_keys": ["main_images"],
                    "input_on_done": False,
                }
            },
        }
    )


def _make_reward_input(
    slot_ids: list[int], valid_env_ids: list[int] | None = None
) -> dict[str, object]:
    valid_env_ids = valid_env_ids or list(range(len(slot_ids)))
    valid_env_id_set = set(valid_env_ids)

    return {
        "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
        "main_images": torch.zeros((len(slot_ids), 1, 1, 1), dtype=torch.uint8),
        "task_descriptions": [f"task-{slot_id}" for slot_id in slot_ids],
        "history_input": {
            "history_window": {
                "main_images": [
                    [f"frame-{slot_id}"] if env_idx in valid_env_id_set else []
                    for env_idx, slot_id in enumerate(slot_ids)
                ]
            }
        },
    }


def _make_rank_roundtrip_input(batch_size: int) -> dict[str, object]:
    return {
        "slot_ids": torch.arange(batch_size, dtype=torch.long),
        "main_images": torch.arange(batch_size, dtype=torch.uint8).view(
            batch_size, 1, 1, 1
        ),
        "task_descriptions": [f"task-{idx}" for idx in range(batch_size)],
        "history_input": {
            "history_window": {
                "main_images": [[f"frame-{idx}"] for idx in range(batch_size)]
            }
        },
    }


def _simulate_reward_roundtrip(
    batch_size: int, env_world_size: int, reward_world_size: int
) -> torch.Tensor:
    env_batch_size = batch_size // env_world_size
    reward_input = _make_rank_roundtrip_input(batch_size)
    env_local_batches = split_dict(
        reward_input, [env_batch_size for _ in range(env_world_size)]
    )

    incoming_reward_shards: dict[int, dict[int, dict[str, object]]] = {
        reward_rank: {} for reward_rank in range(reward_world_size)
    }
    for env_rank, env_batch in enumerate(env_local_batches):
        dst_ranks_and_sizes = CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=reward_world_size,
            src_rank=env_rank,
        )
        reward_input_shards = split_dict(
            env_batch, [size for _, size in dst_ranks_and_sizes]
        )
        for (reward_rank, _), shard in zip(dst_ranks_and_sizes, reward_input_shards):
            incoming_reward_shards[reward_rank][env_rank] = shard

    reward_outputs: dict[int, torch.Tensor] = {}
    for reward_rank in range(reward_world_size):
        src_ranks_and_sizes = CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=reward_world_size,
            dst_rank=reward_rank,
        )
        ordered_shards = [
            incoming_reward_shards[reward_rank][env_rank]
            for env_rank, _ in src_ranks_and_sizes
        ]
        merged = cat_list_of_dict_tensor(ordered_shards, dim=0)
        expected_slot_ids = [int(slot_id) for slot_id in merged["slot_ids"].tolist()]

        assert merged["task_descriptions"] == [
            f"task-{slot_id}" for slot_id in expected_slot_ids
        ]
        assert merged["history_input"]["history_window"]["main_images"] == [
            [f"frame-{slot_id}"] for slot_id in expected_slot_ids
        ]

        reward_outputs[reward_rank] = merged["slot_ids"].to(dtype=torch.float32)

    incoming_env_rewards: dict[int, dict[int, torch.Tensor]] = {
        env_rank: {} for env_rank in range(env_world_size)
    }
    for reward_rank, reward_tensor in reward_outputs.items():
        dst_ranks_and_sizes = CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=reward_world_size,
            dst_world_size=env_world_size,
            src_rank=reward_rank,
        )
        reward_shards = torch.split(
            reward_tensor, [size for _, size in dst_ranks_and_sizes], dim=0
        )
        for (env_rank, _), shard in zip(dst_ranks_and_sizes, reward_shards):
            incoming_env_rewards[env_rank][reward_rank] = shard

    env_results: list[torch.Tensor] = []
    for env_rank in range(env_world_size):
        src_ranks_and_sizes = CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=reward_world_size,
            dst_world_size=env_world_size,
            dst_rank=env_rank,
        )
        env_results.append(
            torch.cat(
                [
                    incoming_env_rewards[env_rank][reward_rank]
                    for reward_rank, _ in src_ranks_and_sizes
                ],
                dim=0,
            )
        )

    return torch.cat(env_results, dim=0)


def test_history_vlm_reward_model_keeps_micro_batch_order():
    model = _TestHistoryVLMRewardModel(_make_cfg(infer_micro_batch_size=2))

    rewards = model.compute_reward(_make_reward_input([10, 11, 12, 13]))

    assert torch.equal(rewards, torch.tensor([10.0, 11.0, 12.0, 13.0]))
    assert model.input_builder.calls == [[10, 11], [12, 13]]


def test_history_vlm_reward_model_writes_sparse_valid_envs_back_to_slots():
    model = _TestHistoryVLMRewardModel(_make_cfg(infer_micro_batch_size=2))

    rewards = model.compute_reward(
        _make_reward_input([20, 21, 22, 23], valid_env_ids=[1, 3])
    )

    assert torch.equal(rewards, torch.tensor([0.0, 21.0, 0.0, 23.0]))
    assert model.input_builder.calls == [[21], [23]]


@pytest.mark.parametrize(
    ("env_world_size", "reward_world_size"),
    [(1, 1), (2, 1), (1, 2), (3, 2), (2, 3), (4, 2), (2, 4)],
)
def test_reward_roundtrip_keeps_env_slot_alignment(
    env_world_size: int, reward_world_size: int
):
    rewards = _simulate_reward_roundtrip(
        batch_size=12,
        env_world_size=env_world_size,
        reward_world_size=reward_world_size,
    )

    assert torch.equal(rewards, torch.arange(12, dtype=torch.float32))
