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

"""Checks for alternate value-model backbone wiring."""

from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.vla_lib_value_model.processing_smolvlm import (
    SmolVLMImageProcessorAdapter,
)
from rlinf.models.embodiment.vla_lib_value_model.smolvlm_with_multi_expert import (
    get_num_key_value_heads,
    get_real_image_slot_mask,
    normalize_attention_mask_for_eager_attention,
    resolve_torch_dtype,
    scatter_image_hidden_states_back_to_slots,
)

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
    assert (
        'self.backbone_variant = getattr(config, "backbone_variant", "paligemma")'
        in content
    )
    assert 'self.backbone_variant == "siglip_gemma3"' in content
    assert "SiglipGemma3WithMultiExpert(" in content
    assert "siglip_path=siglip_path" in content
    assert "gemma3_path=gemma3_path" in content
    assert 'self.backbone_variant == "smolvlm"' in content
    assert "SmolVLMWithMultiExpert(" in content
    assert 'smolvlm_path = getattr(config, "smolvlm_path", "")' in content


def test_validation_config_defines_local_siglip_gemma3_paths() -> None:
    """Validation config should point at local Gemma3 and SigLIP paths."""
    content = (CONFIG_DIR / "libero_value_model_siglip_gemma3_test.yaml").read_text()
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
    assert (
        "vlm_modules = [self.vision_tower, self.multi_modal_proj, self.gemma3]"
        in content
    )


def test_processor_factory_supports_smolvlm() -> None:
    """Processor factory should route SmolVLM backbones to the adapter."""
    content = (VALUE_MODEL_DIR / "processing_smolvlm.py").read_text()
    assert "class SmolVLMProcessor(PI05Processor):" in content
    assert 'if backbone_variant == "smolvlm":' in content
    assert "return SmolVLMProcessor(" in content


def test_smolvlm_adapter_preserves_camera_slots_for_missing_images() -> None:
    """Missing cameras should stay in-place instead of shifting later slots."""

    class _FakeImageProcessor:
        def __call__(self, nested_images, return_tensors="pt"):
            del return_tensors
            pixel_values = torch.from_numpy(np.asarray(nested_images, dtype=np.uint8))
            return {"pixel_values": pixel_values}

    adapter = SmolVLMImageProcessorAdapter(_FakeImageProcessor())
    images = {
        "base_0_rgb": torch.tensor([[[[1]]], [[[2]]]], dtype=torch.uint8),
        "left_wrist_0_rgb": torch.tensor([[[[3]]], [[[4]]]], dtype=torch.uint8),
        "right_wrist_0_rgb": torch.tensor([[[[5]]], [[[6]]]], dtype=torch.uint8),
    }
    image_masks = {
        "base_0_rgb": torch.tensor([True, True]),
        "left_wrist_0_rgb": torch.tensor([False, True]),
        "right_wrist_0_rgb": torch.tensor([True, False]),
    }

    encoded = adapter(images=images, image_masks=image_masks)

    assert encoded["pixel_values"].shape[:2] == (2, 3)
    assert encoded["image_masks"].tolist() == [
        [True, False, True],
        [True, True, False],
    ]
    assert encoded["pixel_values"][0, 0].item() == 1
    assert encoded["pixel_values"][0, 1].item() == 0
    assert encoded["pixel_values"][0, 2].item() == 5
    assert encoded["pixel_values"][1, 0].item() == 2
    assert encoded["pixel_values"][1, 1].item() == 4
    assert encoded["pixel_values"][1, 2].item() == 0


@pytest.mark.parametrize(
    ("precision", "expected_dtype"),
    [
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("float32", torch.float32),
    ],
)
def test_resolve_torch_dtype_matches_shared_precision_contract(
    precision: str, expected_dtype: torch.dtype
) -> None:
    """SmolVLM should honor the same dtype contract as the other backbones."""
    assert resolve_torch_dtype(precision) == expected_dtype


def test_resolve_torch_dtype_rejects_invalid_precision() -> None:
    """SmolVLM precision parsing should fail loudly on unknown values."""
    with pytest.raises(ValueError, match="Invalid precision"):
        resolve_torch_dtype("fp8")


def test_get_num_key_value_heads_falls_back_to_projection_shape() -> None:
    """SmolVLM should support attention modules without num_key_value_heads."""

    class _FakeAttention:
        head_dim = 64
        k_proj = torch.nn.Linear(960, 320, bias=False)

    assert get_num_key_value_heads(_FakeAttention()) == 5


def test_get_real_image_slot_mask_matches_zero_padded_slot_filtering() -> None:
    """SmolVLM should preserve split-image slots by marking all-zero slots as padding."""
    pixel_values = torch.tensor(
        [
            [
                [[[1.0]]],
                [[[0.0]]],
                [[[2.0]]],
            ],
            [
                [[[0.0]]],
                [[[3.0]]],
                [[[0.0]]],
            ],
        ]
    )

    assert get_real_image_slot_mask(pixel_values).tolist() == [
        [True, False, True],
        [False, True, False],
    ]


def test_scatter_image_hidden_states_back_to_slots_restores_dropped_positions() -> None:
    """SmolVLM image blocks should be scattered back to their original split-image slots."""
    real_image_slot_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, False],
        ]
    )
    image_hidden_states = torch.tensor(
        [
            [[1.0], [2.0]],
            [[3.0], [4.0]],
            [[5.0], [6.0]],
        ]
    )

    restored = scatter_image_hidden_states_back_to_slots(
        image_hidden_states=image_hidden_states,
        real_image_slot_mask=real_image_slot_mask,
    )

    assert restored.shape == (2, 6, 1)
    assert restored[:, :, 0].tolist() == [
        [1.0, 2.0, 0.0, 0.0, 3.0, 4.0],
        [0.0, 0.0, 5.0, 6.0, 0.0, 0.0],
    ]


def test_normalize_attention_mask_for_eager_attention_supports_additive_masks() -> None:
    """SmolVLM eager attention should accept RLinf's additive mask convention."""
    attention_mask = torch.tensor(
        [
            [
                [0.0, -2.3e38],
                [0.0, 0.0],
            ]
        ]
    )

    normalized = normalize_attention_mask_for_eager_attention(attention_mask)

    assert normalized.dtype == torch.bool
    assert normalized.tolist() == [[[True, False], [True, True]]]
