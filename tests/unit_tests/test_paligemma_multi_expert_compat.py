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

"""Compatibility checks for unified multi-expert wrappers.

These tests intentionally inspect source text to avoid importing heavyweight
openpi05 package dependencies during unit-test collection.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OPENPI05_DIR = REPO_ROOT / "rlinf/models/embodiment/vla_lib_value_model/openpi05"


def test_800m_module_keeps_compat_import_path() -> None:
    """Legacy 800M module should re-export from unified implementation."""
    content = (OPENPI05_DIR / "paligemma_with_multi_expert_800m.py").read_text()
    assert (
        "from .paligemma_with_multi_expert import PaliGemmaWithMultiExpert800M"
        in content
    )


def test_unified_module_defines_800m_wrapper() -> None:
    """Unified module should contain the 800M compatibility wrapper class."""
    content = (OPENPI05_DIR / "paligemma_with_multi_expert.py").read_text()
    assert (
        "class PaliGemmaWithMultiExpert800M(PaliGemmaWithMultiExpertModel):" in content
    )
    assert 'backbone_variant="paligemma_800m"' in content
