#!/usr/bin/env python3
"""Materialise accepted repairs as a fresh corpus, requiring checked image builds.

Instructions and original supporting tests are preserved. Oracle assets travel
only in tests/. Proposed setup.sh rewrites remain in fixes/ as provenance; image
changes come exclusively from build_images.py receipts. No source is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

try:
    from .image_safety import corpus_fingerprints, removal_paths, sha256_file, task_id
except ImportError:
    from image_safety import corpus_fingerprints, removal_paths, sha256_file, task_id


def checked_builds(root: Path) -> dict:
    manifest = {r['task_id']: r for r in map(json.loads, (root / 'manifest.jsonl').read_text().splitlines())}
    receipts = {r['task_id']: r for r in json.loads((root / 'builds.json').read_text())}
    result = {}
    for tid, row in manifest.items():
        task_id(tid)
        if not row['needs_build']:
            continue
        receipt = receipts.get(tid, {})
        image = root / (tid + '.sif')
        definition = root / 'defs' / (tid + '.def')
        if (receipt.get('status') != 'ok' or not image.is_file()
                or receipt.get('image_sha256') != sha256_file(image)
                or receipt.get('recipe_sha256') != sha256_file(definition)
                or receipt.get('removed_paths') != row['removed_paths']
                or receipt.get('dependencies') != row['dependencies']):
            raise ValueError(f'Missing, failed, or stale image build: {tid}')
        result[tid] = {**row, 'image': str(image.resolve()),
                       'resolved_dependencies': receipt.get('resolved_dependencies')}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--fixes', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True, help='JSON list of retained task IDs')
    parser.add_argument('--src-task-data', type=Path, required=True)
    parser.add_argument('--source-jsonl', type=Path, required=True)
    parser.add_argument('--image-builds', type=Path, help='build_images.py --apply output directory')
    parser.add_argument('--out', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    idx = {r['task_id']: r for r in map(json.loads, args.index.read_text().splitlines())}
    want = json.loads(args.tasks.read_text())
    if len(want) != len(set(want)):
        parser.error('Duplicate task IDs')
    originals = {}
    for row in map(json.loads, args.source_jsonl.read_text().splitlines()):
        env = row['env_config']
        if isinstance(env, str):
            env = json.loads(env)
        tid = task_id(env['task_id'])
        if tid in originals:
            parser.error(f'Duplicate source task ID: {tid}')
        originals[tid] = row
    builds = checked_builds(args.image_builds) if args.image_builds else {}
    prepared, artifact_rows = [], []
    # Fail before writing a partial corpus or silently falling back to a leaky base.
    for tid in want:
        task_id(tid)
        src, fix = args.src_task_data / tid, args.fixes / tid
        item = idx[tid]
        original = originals[tid]
        if not (src / 'tests').is_dir() or (src / 'tests').is_symlink():
            raise ValueError(f'Tests must be a real directory: {tid}')
        for name in ('instruction.md', 'setup.sh', 'tests/test.sh'):
            if not (src / name).is_file() or (src / name).is_symlink():
                raise ValueError(f'Missing source file: {tid}/{name}')
        p = fix / 'repair.json'
        repair = json.loads(p.read_text()) if p.exists() else {}
        accepted = repair.get('accepted', False)
        if accepted and repair.get('action') == 'rewrite_verifier' and not (fix / 'test.sh').is_file():
            raise ValueError(f'Missing accepted verifier: {tid}')
        paths = removal_paths(repair.get('leaked_paths', [])) if accepted else []
        built = builds.get(tid)
        if paths or (accepted and repair.get('needs_image_rebuild')):
            if not built or built['removed_paths'] != paths:
                raise ValueError(f'Missing checked leak-removal image: {tid}')
        if built and built['registry_image'] != item['image']:
            raise ValueError(f'Image build uses a different base: {tid}')
        image = built['image'] if built else item.get('sif_path', '')
        if not image or not Path(image).is_file():
            raise ValueError(f'Missing image: {tid}')
        assets = []
        for path in repair.get('oracle_to_tests', []) if accepted else []:
            removal_paths([path])
            asset = fix / 'tests_assets' / Path(path).name
            if not asset.is_file() or not asset.stat().st_size:
                raise ValueError(f'Missing relocated oracle: {tid}/{asset.name}')
            destination = src / 'tests' / asset.name
            if destination.is_dir() and not destination.is_symlink():
                raise ValueError(f'Oracle asset collides with a directory: {tid}/{asset.name}')
            assets.append(asset)
        names = [asset.name for asset in assets]
        if len(names) != len(set(names)) or 'test.sh' in names:
            raise ValueError(f'Oracle basename collision: {tid}')
        prepared.append((tid, src, fix, accepted, str(Path(image).resolve()), original, assets))
        artifact_rows.append({'task_id': tid, 'verifier_repaired': bool(accepted and (fix / 'test.sh').is_file()),
                              'image_rebuilt': bool(built), 'removed_paths': paths})
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for tid, src, fix, accepted, image, original, assets in prepared:
        dst = args.out / 'task-data' / tid
        shutil.copytree(src, dst, symlinks=True)
        if accepted and (fix / 'test.sh').is_file():
            shutil.copy2(fix / 'test.sh', dst / 'tests/test.sh')
        for asset in assets:
            destination = dst / 'tests' / asset.name
            # The original test tree may contain a link with this basename.
            # Replace that entry rather than following it into the input tree.
            if destination.is_symlink():
                destination.unlink()
            shutil.copy2(asset, destination)
        (dst / 'tests/test.sh').chmod(0o755)
        rec = dict(original)
        env = rec['env_config']
        env = json.loads(env) if isinstance(env, str) else dict(env)
        rec['ground_truth'] = tid
        rec['env_config'] = {**env, 'task_id': tid, 'image': image}
        records.append(rec)
    (args.out / 'tasks.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    (args.out / 'artifacts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in artifact_rows))
    (args.out / 'fingerprints.json').write_text(json.dumps(corpus_fingerprints(args.out), indent=2) + '\n')
    print(f'Materialised {len(records)} tasks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
