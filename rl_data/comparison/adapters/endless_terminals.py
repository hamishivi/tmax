"""Adapter for ``obiwan96/endless-terminals``.

Flattens the Harbor-style layout shipped on Hugging Face (per task:
``environment/``, ``tests/``, ``instruction.md``, ``task.toml``, ``solution/``)
into our canonical Apptainer layout.

CLI:

    python -m rl_data.comparison.adapters.endless_terminals \\
        --dst rl_data/output/tasks_endless_terminals

Options:
    --limit N       Convert only the first N source tasks.
    --workers K     Parallel conversion workers (default 16).
    --skip-download Skip the HF snapshot (reuse whatever is already cached).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rl_data.comparison.adapters import (
    Adapter,
    flatten_harbor_task,
    register_adapter,
)

logger = logging.getLogger(__name__)


class EndlessTerminalsAdapter(Adapter):
    name = "endless_terminals"
    hf_repo_id = "obiwan96/endless-terminals"
    default_dst = "rl_data/output/tasks_endless_terminals"

    def convert_one(self, src: Path, dst_root: Path):
        return flatten_harbor_task(
            src, dst_root,
            source_name="et",
            source_repo=self.hf_repo_id,
        )


register_adapter(EndlessTerminalsAdapter())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_et_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(EndlessTerminalsAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--revision", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = EndlessTerminalsAdapter()

    if args.skip_download:
        snapshot = args.cache_dir.resolve()
        logger.info("Skipping download; using %s", snapshot)
    else:
        snapshot = adapter.fetch(args.cache_dir.resolve(), revision=args.revision)

    converted, skipped = adapter.convert_all(
        snapshot, args.dst.resolve(),
        limit=args.limit, workers=args.workers,
    )
    logger.info("Done. converted=%d skipped=%d  dst=%s",
                converted, skipped, args.dst)


if __name__ == "__main__":
    main()
