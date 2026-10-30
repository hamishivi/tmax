#!/usr/bin/env python3
"""Independently check the pinned public TMax v1.1 artifact provenance.

Downloads ~160 MB of task data/metadata, never images; performs no API model calls
or writes to external services. Existing cache files must match the checks below.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

import pyarrow.parquet as pq

TRAINING = ('TMaxxx/tmax-13k-v1.1', '84c6b85842e6ea108db97647ebbe48f9f1c8e7be')
METADATA = ('TMaxxx/tmax-13k-v1.1-metadata', '62fdc43e34e8c8e9895aa3126ec02e1c1233591c')
ORIGINAL = ('allenai/TMax-15K', 'e3ded940596cc611c9815684ba51a7e8ee0cffc9')
REFERENCE = ('allenai/tmax-15k-open-instruct', '7b090eca98bf351356bc1c64290c5c4a09f2f98c')
ARCHIVE_SHA256 = 'a654638ec448403823273b6510eeb457ae6d9914c7bf4f87ce5302d1fb0c7146'
CONVERSION_HASHES = {
    'prepare_hardened_open_instruct.py': '05923007f83afea44c310518fa490cb41d29c9b3f3a1f7da52cf3beaf148f7fc',
    'publish_hardened_images.py': '0b4986032c1dffe82e0515fe411a71d08b50f8bde6acdceca42221c80664a431',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def fetch(cache, source, filename):
    repo, rev = source
    path = cache / repo / rev / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + '.part')
        with urllib.request.urlopen(f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{filename}', timeout=120) as response:
            with pending.open('wb') as out:
                while chunk := response.read(4 * 1024 * 1024):
                    out.write(chunk)
        pending.replace(path)
    return path


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def audit(cache):
    hashes = {}
    def get(source, filename):
        path = fetch(cache, source, filename)
        hashes[f'{source[0]}@{source[1]}/{filename}'] = digest(path)
        return path
    training = pq.read_table(get(TRAINING, 'data/train-00000-of-00001.parquet'))
    metadata = pq.read_table(get(METADATA, 'data/train-00000-of-00001.parquet'))
    original = pq.read_table(get(ORIGINAL, 'data/train-00000-of-00001.parquet'))
    reference = pq.read_table(get(REFERENCE, 'data/train-00000-of-00001.parquet'))
    rows, meta = training.to_pylist(), metadata.to_pylist()
    tids = [r['ground_truth'] for r in rows]
    mids = [r['task_id'] for r in meta]
    oids = [r['task_id'] for r in original.to_pylist()]
    exclusions = jsonl(get(METADATA, 'exclusions.jsonl'))
    eids = [r['task_id'] for r in exclusions]
    require(len(tids) == len(set(tids)) == 13565, 'Training IDs/count differ')
    require(len(mids) == len(set(mids)) == 13565 and set(mids) == set(tids), 'Metadata IDs differ')
    require(len(eids) == len(set(eids)) == 1036, 'Exclusion IDs/count differ')
    require(not set(eids) & set(tids) and set(eids) | set(tids) == set(oids)
            and len(oids) == 14601, 'Original/included/excluded task accounting differs')
    require(training.schema.equals(reference.schema, check_metadata=True), 'Open Instruct schema differs')
    archive = get(TRAINING, 'task-data.tar.gz')
    require(digest(archive) == ARCHIVE_SHA256, 'Task archive SHA256 differs')
    expected, files = set(tids), {}
    with tarfile.open(archive, 'r:gz') as source:
        for member in source:
            name = member.name.removeprefix('./')
            if '/' not in name or not member.isfile():
                continue
            tid, rel = name.split('/', 1)
            require(tid in expected, 'Archive includes an unexpected task')
            if rel in {'instruction.md', 'setup.sh', 'tests/test.sh'}:
                require((tid, rel) not in files, 'Archive repeats a required file')
                files[tid, rel] = source.extractfile(member).read()
    bootstrap_count = 0
    for row in rows:
        tid = row['ground_truth']
        require(row['env_config']['task_id'] == tid, 'Environment ID differs')
        for name in ('instruction.md', 'setup.sh', 'tests/test.sh'):
            require((tid, name) in files, 'Missing required task file')
        instruction = files[tid, 'instruction.md'].decode().strip()
        require(any(instruction in m['content'] for m in row['messages'] if m['role'] == 'user'),
                'Archive instruction differs from training prompt')
        bootstrap_count += b'runtime-defect fix' in files[tid, 'tests/test.sh']
    plan = json.loads(get(TRAINING, 'image-publication-plan.json').read_text())
    published = jsonl(get(TRAINING, 'image-manifest.jsonl'))
    require(len(published) == len({r['task_id'] for r in published}) == 1713, 'Image receipt IDs differ')
    require(all(r['status'] == 'ok' and r['digest'].startswith('sha256:') for r in published), 'Image publication receipt failed')
    planned = {r['task_id']: r for r in plan['images']}
    require(len(planned) == len(plan['images']) == 1713, 'Image plan count differs')
    require(set(planned) == {r['task_id'] for r in published}, 'Planned/published image IDs differ')
    require(all(r['image'] == planned[r['task_id']]['image'] for r in published), 'Published image reference differs')
    by_tid = {r['ground_truth']: r for r in rows}
    require(all(by_tid[t]['env_config']['image'] == r['image'] for t, r in planned.items()), 'Training image differs')
    manifest = jsonl(get(METADATA, 'image-recipe/manifest.jsonl'))
    for name in ('README.md', 'rebuild.sh', 'build_derived_local.py'):
        get(METADATA, 'image-recipe/' + name)
    require(len(manifest) == 14601 and len({r['task_id'] for r in manifest}) == 14601, 'Recipe task count differs')
    require(sum(r['needs_build'] for r in manifest) == 1763, 'All-corpus derivation count differs')
    retained_derivations = {r['task_id'] for r in manifest if r['needs_build'] and r['task_id'] in expected}
    require(retained_derivations == set(planned), 'Retained recipe derivations differ from publication plan')
    for name, sha in CONVERSION_HASHES.items():
        require(digest(get(TRAINING, 'conversion/' + name)) == sha, 'Released conversion source changed')
    image_counts = Counter(r['env_config']['image'].rsplit(':', 1)[0] for r in rows)
    require(dict(image_counts) == {'hamishi740/swerl-tmax-v3': 11852, 'hamishivi/swerl-tmax-hardened': 1713}, 'Image mix differs')
    repairs = Counter(r['repair_status'] for r in meta)
    require(dict(repairs) == {'applied': 12440, 'no_repair_needed': 1125}, 'Repair counts differ')
    require(sum(r['in_filtered_subset'] for r in meta) == 7015, 'Solvable subset count differs')
    original_labels = Counter(r['rubric_label'] for r in meta + exclusions)
    require(dict(original_labels) == {'VERIFIER-TOO-WEAK': 11160, 'ANSWER-LEAK': 1592, 'CLEAN': 1125,
            'INSTR-VERIFIER-MISMATCH': 441, 'INSTR-ENV-MISMATCH': 177, 'OTHER': 83, 'INSTR-AMBIGUOUS': 23},
            'Original rubric counts differ')
    return {'training': dict(zip(('repository', 'revision'), TRAINING)),
            'metadata': dict(zip(('repository', 'revision'), METADATA)),
            'original_tasks': len(oids), 'retained_tasks': len(tids), 'excluded_tasks': len(eids),
            'repair_status': dict(repairs), 'rubric_labels': dict(Counter(r['rubric_label'] for r in meta)),
            'original_rubric_labels': dict(original_labels),
            'validation_status': dict(Counter(r['validation_status'] for r in meta)),
            'known_verifier_defects': sum(r['known_verifier_defect'] for r in meta),
            'image_repositories': dict(image_counts), 'published_image_kinds': dict(Counter(r['kind'] for r in published)),
            'all_corpus_derivations': 1763, 'archive_sha256': ARCHIVE_SHA256,
            'schema_matches_reference': True, 'task_id_accounting_matches_original': True,
            'archive_instructions_match_training': True,
            'verifiers_with_legacy_runtime_dependency_bootstrap': bootstrap_count,
            'artifact_sha256': hashes,
            'limits': ['No historical model calls or verdicts reproduced', 'No live image rebuilds or registry runtime checks performed']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.cache)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"Checked {report['retained_tasks']} tasks; report: {args.out}")


if __name__ == '__main__':
    main()
