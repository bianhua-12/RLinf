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

"""Source-level checks for shared multi-expert backbones."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALUE_MODEL_DIR = REPO_ROOT / "rlinf/models/embodiment/vla_lib_value_model"


def test_paligemma_module_exports_shared_backbone_helpers() -> None:
    """PaliGemma shared module should keep the dtype helper used by both paths."""
    content = (VALUE_MODEL_DIR / "paligemma_with_multi_expert.py").read_text()
    assert "def _requires_uniform_dtype() -> bool:" in content
    assert "class PaliGemmaWithMultiExpertModel(nn.Module):" in content


def test_siglip_gemma3_module_defines_independent_backbone() -> None:
    """SigLIP/Gemma3 module should exist as its own implementation."""
    content = (VALUE_MODEL_DIR / "siglip_gemma3_with_multi_expert.py").read_text()
    assert "class SiglipGemma3WithMultiExpert(nn.Module):" in content
    assert (
        "self.multi_modal_proj = nn.Linear(siglip_hidden, gemma3_hidden, bias=True)"
        in content
    )


def test_smolvlm_module_defines_independent_backbone() -> None:
    """SmolVLM path should exist as its own implementation."""
    content = (VALUE_MODEL_DIR / "smolvlm_with_multi_expert.py").read_text()
    assert "class SmolVLMWithMultiExpert(nn.Module):" in content
    assert 'attention_mode: str = "cross_attn"' in content
    assert "self.experts = nn.ModuleDict()" in content
