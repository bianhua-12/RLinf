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

"""Source-level checks for 800m value-model wiring.

These checks avoid importing heavyweight model modules during collection while
still verifying that the 800m configuration path is wired through the factory.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALUE_MODEL_DIR = REPO_ROOT / "rlinf/models/embodiment/vla_lib_value_model"
CONFIG_DIR = REPO_ROOT / "examples/vla_lib_sft/config"


def test_factory_forwards_800m_backbone_fields() -> None:
    """Hydra model factory should forward 800m-specific config values."""
    content = (VALUE_MODEL_DIR / "__init__.py").read_text()
    assert '_set("backbone_variant", "paligemma_2b")' in content
    assert '_set("siglip_path", None)' in content
    assert '_set("gemma3_path", None)' in content


def test_value_critic_uses_backbone_variant_routing() -> None:
    """Critic model should route 800m backbone setup into the shared module."""
    content = (VALUE_MODEL_DIR / "modeling_critic.py").read_text()
    assert (
        'self.backbone_variant = getattr(config, "backbone_variant", "paligemma_2b")'
        in content
    )
    assert "backbone_variant=self.backbone_variant" in content
    assert 'siglip_path=getattr(config, "siglip_path", None)' in content
    assert 'gemma3_path=getattr(config, "gemma3_path", None)' in content
    assert 'self.backbone_variant == "paligemma_800m"' in content


def test_validation_config_defines_local_800m_paths() -> None:
    """800m validation config should point at local Gemma3 and SigLIP paths."""
    content = (CONFIG_DIR / "libero_value_model_800m_test.yaml").read_text()
    assert 'backbone_variant: "paligemma_800m"' in content
    assert (
        'siglip_path: "/mnt/public/shchen/models/siglip2-so400m-patch14-224"' in content
    )
    assert 'gemma3_path: "/mnt/public/shchen/models/gemma-3-270m"' in content


def test_pi0_helper_is_backbone_aware() -> None:
    """Shared PI0 helper should no longer assume only the 2B paligemma path."""
    content = (VALUE_MODEL_DIR / "openpi/modeling_pi0.py").read_text()
    assert "def _get_language_model_dtype(self) -> torch.dtype:" in content
    assert 'if hasattr(self.paligemma_with_expert, "paligemma"):' in content
    assert "language_model = self.paligemma_with_expert.gemma3.model" in content


def test_800m_multi_expert_uses_uniform_bf16_under_fsdp() -> None:
    """800m multi-expert path should avoid mixed fp32/bf16 params under FSDP."""
    content = (VALUE_MODEL_DIR / "openpi05/paligemma_with_multi_expert.py").read_text()
    assert "def _requires_uniform_dtype() -> bool:" in content
    assert 'os.environ.get("WORLD_SIZE") is not None' in content
    assert 'os.environ.get("LOCAL_RANK") is not None' in content
    assert 'if self.backbone_variant == "paligemma_800m":' in content
    assert (
        'logger.info("Using uniform bfloat16 dtype for paligemma_800m backbone")'
        in content
    )


def test_800m_freeze_vlm_freezes_full_vlm_stack() -> None:
    """800m freeze_vlm should freeze SigLIP, projector, and Gemma3 together."""
    content = (VALUE_MODEL_DIR / "openpi05/paligemma_with_multi_expert.py").read_text()
    assert (
        "vlm_modules = [self.vision_tower, self.multi_modal_proj, self.gemma3]"
        in content
    )
