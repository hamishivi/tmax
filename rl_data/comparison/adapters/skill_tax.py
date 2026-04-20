"""Identity adapter for our own dataset.

Our dataset is already in canonical layout, so there is no conversion to do.
This adapter exists so the CLI has a single uniform entry point for every
dataset it compares.
"""

from __future__ import annotations

from pathlib import Path

from rl_data.comparison.adapters import Adapter, register_adapter


class SkillTaxAdapter(Adapter):
    name = "skill_tax"
    hf_repo_id = None  # local-only
    default_dst = "rl_data/output/tasks_skill_tax_20260401_10k"

    def fetch(self, cache_dir: Path, *, revision=None) -> Path:  # noqa: D401
        return Path(self.default_dst)

    def convert_one(self, src: Path, dst_root: Path):  # noqa: D401
        # No-op; the data already lives in canonical form.
        return src.name


register_adapter(SkillTaxAdapter())
