"""Description loaders for the decontamination eval.

Two layouts are supported:

* Midtraining dataset: ``<dir>/task_*/task.json`` with a top-level
  ``description`` string (canonical skill-tax / adapter layout).
* Harbor benchmark: a directory tree produced by ``harbor download
  <name> --export -o <dir>`` containing ``instruction.md`` inside each
  task subdirectory. We recursively pick up every ``instruction.md``
  under the given root so both ``<root>/<task>/instruction.md`` and
  ``<root>/<dataset>/<task>/instruction.md`` work.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

DescPair = Tuple[str, str]


def load_dataset_descriptions(tasks_dir: Path) -> List[DescPair]:
    """Return ``[(task_name, description), ...]`` for every task dir under
    *tasks_dir*. A task dir is anything that contains a ``task.json`` (this
    matches ``_is_task_dir`` in :mod:`rl_data.analyze`, covering adapter
    prefixes ``otrl_task_*``, ``tg_*``, ``tt_task_*``, ``r2e_*``, ...)."""
    out: List[DescPair] = []
    if not tasks_dir.exists():
        logger.warning("dataset dir missing: %s", tasks_dir)
        return out
    for entry in sorted(tasks_dir.iterdir()):
        if not entry.is_dir():
            continue
        task_json = entry / "task.json"
        if not task_json.exists():
            continue
        try:
            data = json.loads(task_json.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skip %s: %s", task_json, e)
            continue
        desc = (data.get("description") or "").strip()
        if not desc:
            continue
        out.append((data.get("name") or entry.name, desc))
    return out


def load_benchmark_descriptions(bench_dir: Path) -> List[DescPair]:
    """Return ``[(task_name, instruction_text), ...]`` for every
    ``instruction.md`` found under *bench_dir* (any depth)."""
    out: List[DescPair] = []
    if not bench_dir.exists():
        logger.warning("benchmark dir missing: %s", bench_dir)
        return out
    for path in sorted(bench_dir.rglob("instruction.md")):
        try:
            text = path.read_text().strip()
        except OSError as e:
            logger.warning("skip %s: %s", path, e)
            continue
        if not text:
            continue
        out.append((path.parent.name, text))
    return out
