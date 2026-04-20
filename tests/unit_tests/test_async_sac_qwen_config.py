from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from rlinf.config import validate_cfg


@pytest.fixture(autouse=True)
def clear_hydra_state():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _compose_async_sac_qwen_cfg(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    embodied_path = repo_root / "examples" / "embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(embodied_path))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(embodied_path / "config")
    ):
        return compose(
            config_name="maniskill_sac_mlp_qwen3vl4b_dualview_history_reward_async"
        )


def test_async_sac_qwen_config_validates(monkeypatch):
    cfg = _compose_async_sac_qwen_cfg(monkeypatch)

    validated_cfg = validate_cfg(cfg)

    assert validated_cfg.algorithm.loss_type == "embodied_sac"
    assert validated_cfg.reward.use_reward_model is True
    assert validated_cfg.reward.reward_mode == "history_buffer"
    assert validated_cfg.reward.pending_step_window == 1
    assert validated_cfg.reward.aggregate_request_count == 1
    assert validated_cfg.reward.use_output_step == 0
    assert validated_cfg.algorithm.replay_buffer.save_checkpoint is False
    assert validated_cfg.algorithm.replay_buffer.load_checkpoint is False
    assert validated_cfg.actor.model.policy_setup == "panda-ee-dpos"
    assert validated_cfg.actor.model.action_dim == 4
    assert validated_cfg.env.train.init_params.control_mode == "pd_ee_delta_pos"


def test_async_sac_qwen_config_rejects_delayed_reward(monkeypatch):
    cfg = _compose_async_sac_qwen_cfg(monkeypatch)
    cfg.reward.use_output_step = 1

    with pytest.raises(
        AssertionError, match="reward.use_output_step=0"
    ):
        validate_cfg(cfg)
