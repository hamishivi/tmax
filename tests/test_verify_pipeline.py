import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rl_data.verify import build_images, export_prefilter, export_verified, image_safety, materialise, repair
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('image_publication', ROOT / 'scripts/data/publish_hardened_images.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class ImageBuildTests(unittest.TestCase):
    def test_paths_and_dependencies_reject_shell_and_unpinned_inputs(self):
        for path in ['/', '/app/../oracle', '/app/./oracle', '/app/a;id', '/app/$(id)', '/app//oracle']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                image_safety.removal_paths([path])
        for req in ['requests', 'requests>=2', 'requests==2;id', 'https://example.test/a.whl', 'requests==2.*']:
            with self.subTest(req=req), self.assertRaises(ValueError):
                image_safety.dependency_spec({'requirements': [req], 'imports': ['requests']})
        self.assertEqual(image_safety.dependency_spec({'requirements': ['PyYAML==6.0.2'], 'imports': ['yaml']})['imports'], ['yaml'])

    def test_import_check_covers_every_module_and_version(self):
        with tempfile.TemporaryDirectory() as d:
            python = Path(d) / 'python3'
            python.symlink_to(sys.executable)
            script = build_images.check_script([], {'requirements': ['pip==0.0.0'], 'imports': ['json']})
            with patch.dict('os.environ', {'PATH': d}):
                self.assertNotEqual(subprocess.run(['/bin/sh', '-c', script], capture_output=True).returncode, 0)
            script = build_images.check_script([], {'requirements': [], 'imports': ['json', '_tmax_missing_module']})
            with patch.dict('os.environ', {'PATH': d}):
                self.assertNotEqual(subprocess.run(['/bin/sh', '-c', script], capture_output=True).returncode, 0)

    def test_any_remaining_leak_or_dangling_symlink_fails_final_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); leak = root / 'oracle'; absent = root / 'absent'
            leak.write_text('answer')
            script = build_images.check_script([str(leak), str(absent)], None)
            self.assertNotEqual(subprocess.run(['/bin/sh', '-c', script]).returncode, 0)
            leak.unlink(); leak.symlink_to(absent)
            self.assertNotEqual(subprocess.run(['/bin/sh', '-c', script]).returncode, 0)
            leak.unlink()
            self.assertEqual(subprocess.run(['/bin/sh', '-c', script]).returncode, 0)

    def test_failed_final_image_check_does_not_publish_sif(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'base.sif'; base.write_bytes(b'base')
            definition = root / 'recipe.def'; definition.write_text('recipe')
            row = {'task_id': 'task_a', 'base_sif': str(base), 'removed_paths': ['/app/oracle'], 'dependencies': None}
            calls = []
            def runner(args, **kwargs):
                calls.append(args)
                if args[1] == 'build':
                    Path(args[-2]).write_bytes(b'candidate')
                else:
                    raise subprocess.CalledProcessError(1, args)
            with self.assertRaises(subprocess.CalledProcessError):
                build_images.build_one(row, definition, root, runner=runner)
            self.assertFalse((root / 'task_a.sif').exists())
            self.assertEqual(len(calls), 2)

    def test_build_receipt_rejects_modified_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'defs').mkdir()
            image = root / 'task_a.sif'; image.write_bytes(b'image')
            definition = root / 'defs/task_a.def'; definition.write_text('recipe')
            row = {'task_id': 'task_a', 'needs_build': True, 'removed_paths': ['/app/oracle'], 'dependencies': None}
            (root / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
            receipt = {**row, 'status': 'ok', 'image_sha256': image_safety.sha256_file(image),
                       'recipe_sha256': image_safety.sha256_file(definition)}
            (root / 'builds.json').write_text(json.dumps([receipt]))
            self.assertIn('task_a', materialise.checked_builds(root))
            image.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                materialise.checked_builds(root)

    def test_build_records_resolved_transitive_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'base.sif'; base.write_bytes(b'base')
            definition = root / 'recipe.def'; definition.write_text('recipe')
            row = {'task_id': 'task_a', 'base_sif': str(base), 'removed_paths': [],
                   'dependencies': {'requirements': ['PyYAML==6.0.2'], 'imports': ['yaml']}}
            def runner(args, **kwargs):
                if args[1] == 'build':
                    Path(args[-2]).write_bytes(b'candidate')
                if kwargs.get('text'):
                    packages = {} if any(Path(a).name == 'base.sif' for a in args) else {'PyYAML': '6.0.2', 'New_Dep': '1.2.3'}
                    return SimpleNamespace(stdout=json.dumps(packages))
                return SimpleNamespace()
            receipt = build_images.build_one(row, definition, root, runner=runner)
            self.assertEqual(receipt['resolved_dependencies']['requirements'], ['new-dep==1.2.3', 'pyyaml==6.0.2'])

    def test_no_image_required_repair_can_silently_become_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); fixes = root / 'fixes/task_a'; fixes.mkdir(parents=True)
            (root / 'index.jsonl').write_text(json.dumps({'task_id': 'task_a', 'image': 'image:tag'}) + '\n')
            (fixes / 'repair.json').write_text(json.dumps({'accepted': True, 'needs_image_rebuild': True}))
            with self.assertRaises(ValueError):
                build_images.prepare(root / 'index.jsonl', root / 'fixes', {})


class CorpusTests(unittest.TestCase):
    def make_source(self, root):
        task = root / 'source/task_a'; (task / 'tests').mkdir(parents=True)
        (task / 'instruction.md').write_text('do a thing')
        (task / 'setup.sh').write_text('#!/bin/sh\n')
        (task / 'tests/test.sh').write_text('#!/bin/sh\nexit 1\n')
        (task / 'tests/truth.json').write_text('{"answer": 42}')
        image = root / 'original.sif'; image.write_bytes(b'original')
        idx = {'task_id': 'task_a', 'image': 'test:original', 'sif_path': str(image)}
        (root / 'index.jsonl').write_text(json.dumps(idx) + '\n')
        record = {'ground_truth': 'task_a', 'messages': [{'role': 'user', 'content': 'do a thing'}],
                  'env_config': {'task_id': 'task_a', 'image': 'test:original'}}
        (root / 'original.jsonl').write_text(json.dumps(record) + '\n')
        (root / 'ids.json').write_text('["task_a"]')
        (root / 'fixes/task_a').mkdir(parents=True)
        return task

    def invoke(self, root, out):
        argv = ['materialise.py', '--index', str(root / 'index.jsonl'), '--fixes', str(root / 'fixes'),
                '--tasks', str(root / 'ids.json'), '--src-task-data', str(root / 'source'),
                '--source-jsonl', str(root / 'original.jsonl'), '--out', str(out)]
        with patch.object(sys, 'argv', argv):
            return materialise.main()

    def test_materialisation_keeps_supporting_tests_and_binds_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_source(root)
            out = root / 'corpus'
            self.assertEqual(self.invoke(root, out), 0)
            self.assertEqual((out / 'task-data/task_a/tests/truth.json').read_text(), '{"answer": 42}')
            before = image_safety.corpus_fingerprints(out)
            (out / 'task-data/task_a/tests/truth.json').write_text('changed')
            self.assertNotEqual(image_safety.corpus_fingerprints(out), before)
            with self.assertRaises(FileExistsError):
                self.invoke(root, out)

    def test_unbuilt_leak_repair_fails_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_source(root)
            (root / 'fixes/task_a/repair.json').write_text(json.dumps({'accepted': True, 'needs_image_rebuild': True,
                                                                   'leaked_paths': ['/app/oracle']}))
            with self.assertRaises(ValueError):
                self.invoke(root, root / 'corpus')
            self.assertFalse((root / 'corpus').exists())

    def test_refused_repair_script_is_never_adopted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_source(root)
            (root / 'fixes/task_a/repair.json').write_text('{"accepted": false}')
            (root / 'fixes/task_a/test.sh').write_text('bad stale repair')
            self.invoke(root, root / 'corpus')
            self.assertNotIn('stale', (root / 'corpus/task-data/task_a/tests/test.sh').read_text())

    def test_label_retries_do_not_erase_success(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'labels.jsonl'
            path.write_text('\n'.join(map(json.dumps, [{'task_id': 'task_a', 'error': 'transient'},
                {'task_id': 'task_a', 'label': 'CLEAN'}, {'task_id': 'task_a', 'error': 'transient'}])))
            self.assertEqual(export_prefilter.keyed(path, label_retries=True)['task_a']['label'], 'CLEAN')

    def test_runtime_defect_evidence_must_match_final_artifact(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{'task_id': 'task_a', 'artifact_id': 'old', 'verdict': 'ok_rejects_empty'},
                    {'task_id': 'task_a', 'artifact_id': 'new', 'verdict': 'CRASH'}]
            (root / 'sanity.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            self.assertEqual(export_prefilter.sanity_results(root, {'task_a': 'new'}), {'task_a': 'CRASH'})

    def test_prefilter_keeps_accepted_and_clean_but_excludes_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            fixes = Path(folder)
            for tid, accepted in [('task_a', True), ('task_b', False)]:
                (fixes / tid).mkdir()
                (fixes / tid / 'repair.json').write_text(json.dumps({'accepted': accepted}))
            index = {tid: {} for tid in ['task_a', 'task_b', 'task_c']}
            labels = {tid: {'label': 'CLEAN' if tid == 'task_c' else 'VERIFIER-TOO-WEAK'} for tid in index}
            kept, excluded = export_prefilter.candidates(index, labels, fixes)
            self.assertEqual({r['task_id'] for r in kept}, {'task_a', 'task_c'})
            self.assertEqual([r['task_id'] for r in excluded], ['task_b'])

    def test_accepted_proposal_with_missing_oracle_is_separately_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            fixes = Path(folder); (fixes / 'task_a').mkdir()
            (fixes / 'task_a/repair.json').write_text(json.dumps({'accepted': True, 'oracle_to_tests': ['/app/oracle']}))
            kept, excluded = export_prefilter.candidates({'task_a': {}}, {'task_a': {'label': 'ANSWER-LEAK'}}, fixes)
            self.assertEqual(kept, [])
            self.assertEqual(excluded[0]['exclusion_reason'], 'oracle_extraction_incomplete')

    def test_stale_solved_evidence_cannot_enter_filtered_subset(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [{'task_id': 'task_a', 'attempt': 0, 'artifact_id': 'old', 'outcome': 'solved'},
                    {'task_id': 'task_a', 'attempt': 0, 'artifact_id': 'new', 'outcome': 'infra'}]
            (root / 'rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            attempts = export_verified.load_rollouts([str(root), str(root)], expected={'task_a': 'new'})['task_a']
            self.assertEqual(len(attempts), 1)
            self.assertEqual(export_verified.classify(attempts)[0], 'not_evaluable')

    def test_oracle_parent_removal_and_collisions_are_rejected(self):
        proposal = {'action': 'remove_leak', 'setup_sh': 'rm -rf /app/input', 'leaked_paths': ['/app/input'],
                    'oracle_to_tests': [], 'test_sh': ''}
        reason = repair.check_repair(proposal, {'instruction': 'read /app/input/data.csv', 'tests': ''})
        self.assertIn('instruction references', reason)

    def test_public_pipeline_exports_only_current_artifact_success(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_source(root)
            (root / 'labels.jsonl').write_text('{"task_id":"task_a","label":"CLEAN"}\n')
            corpus = root / 'corpus'
            self.invoke(root, corpus)
            fingerprint = image_safety.corpus_fingerprints(corpus)['task_a']
            rollouts = root / 'rollouts'; rollouts.mkdir()
            (rollouts / 'rollouts.jsonl').write_text(json.dumps({'task_id': 'task_a', 'attempt': 0,
                'artifact_id': fingerprint, 'outcome': 'solved', 'reward': 1, 'steps': 1, 'model': 'test'}) + '\n')
            argv = ['export_verified.py', '--rollouts', str(rollouts), '--labels', str(root / 'labels.jsonl'),
                '--fixes', str(root / 'fixes'), '--index', str(root / 'index.jsonl'), '--corpus', str(corpus),
                '--out-dir', str(root / 'verified')]
            with patch.object(sys, 'argv', argv):
                self.assertEqual(export_verified.main(), 0)
            argv = ['export_prefilter.py', '--index', str(root / 'index.jsonl'), '--labels', str(root / 'labels.jsonl'),
                '--fixes', str(root / 'fixes'), '--corpus', str(corpus), '--verified', str(root / 'verified/verified.jsonl'),
                '--out-dir', str(root / 'export')]
            with patch.object(sys, 'argv', argv):
                self.assertEqual(export_prefilter.main(), 0)
            self.assertEqual(json.loads((root / 'export/verified-task-ids.json').read_text()), ['task_a'])
            (corpus / 'task-data/task_a/tests/test.sh').write_text('changed final verifier')
            argv[-1] = str(root / 'stale-export')
            with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                export_prefilter.main()
            self.assertFalse((root / 'stale-export').exists())


class PublicationTests(unittest.TestCase):
    def test_whiteout_layer_is_deterministic_and_removes_target_name(self):
        import tarfile
        first, diff = publication.whiteout_layer(['/app/oracle'])
        second, _ = publication.whiteout_layer(['/app/oracle'])
        self.assertEqual(first, second)
        raw = gzip.decompress(first)
        self.assertEqual(publication.sha256(raw), diff)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            self.assertEqual(archive.getnames(), ['app/.wh.oracle'])

    def test_leak_recipe_refuses_shell_expressions(self):
        for recipe in ['rm -rf /', 'rm -rf /app/../oracle', 'rm -rf /app/$SECRET']:
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                publication.leaked_paths(recipe)


if __name__ == '__main__':
    unittest.main()
