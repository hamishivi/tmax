"""Adapter for ``open-thoughts/OpenThoughts-TB-dev``.

Similar Harbor-style layout as endless-terminals, but task dirs are slugged
(e.g. ``amuse-install``) rather than UUID-based.  Some tasks ship only a
Dockerfile (no apptainer ``container.def``); in that case the shared helper
derives one heuristically so our harness can still build a SIF.

CLI:

    python -m rl_data.comparison.adapters.openthoughts_tb \\
        --dst rl_data/output/tasks_openthoughts_tb
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


class OpenThoughtsTBAdapter(Adapter):
    name = "openthoughts_tb"
    hf_repo_id = "open-thoughts/OpenThoughts-TB-dev"
    default_dst = "rl_data/output/tasks_openthoughts_tb"

    def convert_one(self, src: Path, dst_root: Path):
        # Prefix to avoid clashing with ET's UUID-named dirs in case both
        # end up side-by-side during debugging.
        return flatten_harbor_task(
            src, dst_root,
            source_name="ot",
            source_repo=self.hf_repo_id,
            prefix="otb_",
        )


register_adapter(OpenThoughtsTBAdapter())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("rl_data/output/_otb_cache"))
    ap.add_argument("--dst", type=Path,
                    default=Path(OpenThoughtsTBAdapter.default_dst))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--revision", type=str, default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    adapter = OpenThoughtsTBAdapter()

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
