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

"""Source-level checks for SigLIP2 + Gemma3 value-model wiring."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALUE_MODEL_DIR = REPO_ROOT / "rlinf/models/embodiment/vla_lib_value_model"
CONFIG_DIR = REPO_ROOT / "examples/sft/config"


def test_factory_forwards_siglip_gemma3_backbone_fields() -> None:
    """Hydra model factory should forward SigLIP/Gemma3 config values."""
    content = (VALUE_MODEL_DIR / "__init__.py").read_text()
    assert '_set("backbone_variant", "paligemma")' in content
    assert '_set("siglip_path", None)' in content
    assert '_set("gemma3_path", None)' in content
    assert '_set("smolvlm_path", None)' in content
    assert '_set("smolvlm_attention_mode", "cross_attn")' in content


def test_value_critic_routes_siglip_gemma3_backbone() -> None:
    """Critic model should route the alternate backbone into the right module."""
    content = (VALUE_MODEL_DIR / "modeling_critic.py").read_text()
    assert 'self.backbone_variant = getattr(config, "backbone_variant", "paligemma")' in content
    assert 'self.backbone_variant == "siglip_gemma3"' in content
    assert "SiglipGemma3WithMultiExpert(" in content
    assert "siglip_path=siglip_path" in content
    assert "gemma3_path=gemma3_path" in content
    assert 'self.backbone_variant == "smolvlm"' in content
    assert "SmolVLMWithMultiExpert(" in content
    assert 'smolvlm_path = getattr(config, "smolvlm_path", "")' in content


def test_validation_config_defines_local_siglip_gemma3_paths() -> None:
    """Validation config should point at local Gemma3 and SigLIP paths."""
    content = (CONFIG_DIR / "libero_value_model_800m_test.yaml").read_text()
    assert 'backbone_variant: "siglip_gemma3"' in content
    assert (
        'siglip_path: "/mnt/public/shchen/models/siglip2-so400m-patch14-224"' in content
    )
    assert 'gemma3_path: "/mnt/public/shchen/models/gemma-3-270m"' in content


def test_gradient_checkpointing_helpers_are_backbone_aware() -> None:
    """Gradient-checkpointing helpers should handle both backbone families."""
    content = (VALUE_MODEL_DIR / "modeling_critic.py").read_text()
    assert "def _get_language_model(self) -> nn.Module:" in content
    assert "def _get_vision_tower(self) -> nn.Module:" in content
    assert "def _set_gradient_checkpointing_flag(" in content
    assert 'if self.backbone_variant == "siglip_gemma3":' in content
    assert 'if self.backbone_variant == "smolvlm":' in content


def test_siglip_gemma3_multi_expert_uses_uniform_bf16_under_fsdp() -> None:
    """SigLIP/Gemma3 path should use uniform bf16 under parameter sharding."""
    content = (VALUE_MODEL_DIR / "siglip_gemma3_with_multi_expert.py").read_text()
    assert (
        'def _apply_precision(self, precision: Literal["bfloat16", "float32"]):'
        in content
    )
    assert "_requires_uniform_dtype()" in content
    assert "using uniform bfloat16" in content


def test_siglip_gemma3_freeze_vlm_freezes_full_vlm_stack() -> None:
    """freeze_vlm should freeze SigLIP, projector, and Gemma3 together."""
    content = (VALUE_MODEL_DIR / "siglip_gemma3_with_multi_expert.py").read_text()
    assert "vlm_modules = [self.vision_tower, self.multi_modal_proj, self.gemma3]" in content


def test_processor_factory_supports_smolvlm() -> None:
    """Processor factory should route SmolVLM backbones to the adapter."""
    content = (VALUE_MODEL_DIR / "processing_smolvlm.py").read_text()
    assert "class SmolVLMProcessor(PI05Processor):" in content
    assert 'if backbone_variant == "smolvlm":' in content
    assert "return SmolVLMProcessor(" in content
