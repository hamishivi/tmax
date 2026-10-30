#!/usr/bin/env python3
"""Export portable recipes from checked build_images.py outputs.

This public interface replaces the historical helper's run-directory heuristics.
It requires current image/recipe checksums and retains explicit dependency imports.
Registry tags may move; consumers should also retain the build receipts and SIFs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

try:
    from .build_images import recipe
    from .materialise import checked_builds
except ImportError:
    from build_images import recipe
    from materialise import checked_builds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--builds', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='New recipe directory')
    args = parser.parse_args()
    checked = checked_builds(args.builds)
    rows = list(map(json.loads, (args.builds / 'manifest.jsonl').read_text().splitlines()))
    for row in rows:
        if not re.fullmatch(r'[A-Za-z0-9_./:@-]+', row['registry_image']):
            parser.error('Unsupported registry image reference')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'defs').mkdir()
    manifest = []
    for row in rows:
        tid = row['task_id']
        if row['needs_build']:
            build = checked[tid]
            body = recipe(row['registry_image'], build['removed_paths'],
                          build['resolved_dependencies'] or build['dependencies'])
            body = body.replace('Bootstrap: localimage', 'Bootstrap: docker', 1)
            (args.out / 'defs' / (tid + '.def')).write_text(body)
        manifest.append({k: row[k] for k in ('task_id', 'registry_image', 'derivation', 'needs_build')})
    (args.out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
