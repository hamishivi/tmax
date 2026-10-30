#!/usr/bin/env python3
"""Export the patched-or-originally-clean corpus and separate exclusions.

--plan-only emits candidate IDs from accepted repair proposals. Final export
requires the materialised corpus; only files and images actually adopted count
as applied. Optional verified.jsonl attaches final-artifact validation evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

try:
    from .image_safety import corpus_fingerprints, task_id
except ImportError:
    from image_safety import corpus_fingerprints, task_id

KEEP_LABELS = {'CLEAN', 'TASK-TRIVIAL'}
DEFECTS = {'CRASH', 'BUILD_FAILED', 'EXEC_BLOCKED', 'NO_REWARD', 'PASSES_EMPTY'}


def sanity_results(folder: Path, fingerprints: dict) -> dict:
    results = {}
    for path in sorted(folder.glob('sanity*.jsonl')):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('artifact_id') == fingerprints.get(row.get('task_id')):
                results[row['task_id']] = row['verdict']
    return results


def keyed(path: Path, *, label_retries=False) -> dict:
    result = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        tid = task_id(row['task_id'])
        if tid in result:
            if label_retries:
                if row.get('label') and not row.get('error'):
                    result[tid] = row
                continue
            raise ValueError(f'Duplicate task ID in {path.name}: {tid}')
        result[tid] = row
    return result


def candidates(index: dict, labels: dict, fixes: Path) -> tuple[list[dict], list[dict]]:
    kept, excluded = [], []
    if set(labels) != set(index):
        raise ValueError('Every indexed task must have exactly one final label (resolve retries first)')
    for tid in index:
        label = labels[tid]
        if not label.get('label') or label.get('error'):
            raise ValueError(f'Task has no successful rubric label: {tid}')
        path = fixes / tid / 'repair.json'
        repair = json.loads(path.read_text()) if path.exists() else {}
        base = {'task_id': tid, 'rubric_label': label['label'], 'rubric_reason': label.get('reason', ''),
                'repair_refusal': repair.get('refusal') or ''}
        if repair.get('accepted'):
            missing = [p for p in repair.get('oracle_to_tests', [])
                       if not (fixes / tid / 'tests_assets' / Path(p).name).is_file()
                       or not (fixes / tid / 'tests_assets' / Path(p).name).stat().st_size]
            if missing:
                excluded.append({**base, 'exclusion_reason': 'oracle_extraction_incomplete'})
                continue
        if label['label'] in KEEP_LABELS or repair.get('accepted'):
            kept.append(base)
        else:
            excluded.append({**base, 'exclusion_reason': 'refused' if repair else 'not_repaired'})
    return kept, excluded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--fixes', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True, help='New output directory')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--verified', type=Path)
    parser.add_argument('--sanity', type=Path, help='Runtime sanity output directory for the final corpus')
    args = parser.parse_args()
    if not args.plan_only and not args.corpus:
        parser.error('--corpus is required for a final export')
    index, labels = keyed(args.index), keyed(args.labels, label_retries=True)
    kept, excluded = candidates(index, labels, args.fixes)
    rows = []
    if not args.plan_only:
        fingerprints = corpus_fingerprints(args.corpus)
        artifacts = keyed(args.corpus / 'artifacts.jsonl')
        if set(fingerprints) != {r['task_id'] for r in kept} or set(artifacts) != set(fingerprints):
            raise ValueError('Materialised corpus must contain exactly the retained candidates')
        verified = keyed(args.verified) if args.verified else {}
        sanity = sanity_results(args.sanity, fingerprints) if args.sanity else {}
        for row in kept:
            tid = row['task_id']
            artifact = artifacts[tid]
            applied = artifact['verifier_repaired'] or artifact['image_rebuilt']
            if not applied and row['rubric_label'] not in KEEP_LABELS:
                raise ValueError(f'Accepted proposal has no applied patch: {tid}')
            validation = verified.get(tid, {})
            if validation and validation.get('artifact_id') != fingerprints[tid]:
                raise ValueError(f'Stale validation: {tid}')
            status = validation.get('status', 'not_attempted')
            verdict = sanity.get(tid, 'not_checked')
            if artifact['verifier_repaired'] and verdict == 'not_checked':
                raise ValueError(f'Missing current runtime sanity result for rewritten verifier: {tid}; pass --sanity')
            defective = verdict in DEFECTS
            rows.append({**row, 'repair_status': 'applied' if applied else 'no_repair_needed',
                         'validation_status': {'verified': 'validated', 'unverified': 'not_validated'}.get(status, status),
                         'in_filtered_subset': status == 'verified' and not defective,
                         'known_verifier_defect': defective, 'runtime_sanity': verdict,
                         'verifier_repaired': artifact['verifier_repaired'], 'image_rebuilt': artifact['image_rebuilt'],
                         'attempts': validation.get('attempts', 0), 'n_solved': validation.get('n_solved', 0),
                         'best_reward': validation.get('best_reward', 0.0), 'artifact_id': fingerprints[tid]})
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / 'candidate-task-ids.json').write_text(json.dumps([r['task_id'] for r in kept], indent=2) + '\n')
    (args.out_dir / 'exclusions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in excluded))
    if not args.plan_only:
        (args.out_dir / 'prefilter.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (args.out_dir / 'verified-task-ids.json').write_text(json.dumps([r['task_id'] for r in rows if r['in_filtered_subset']]) + '\n')
    print(json.dumps({'original': len(index), 'retained': len(kept), 'excluded': len(excluded),
                      'plan_only': args.plan_only, 'labels': dict(Counter(r['rubric_label'] for r in kept))}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
