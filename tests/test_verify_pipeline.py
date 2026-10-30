import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import asyncio
from unittest.mock import patch

from rl_data.verify import build_images, export_prefilter, export_verified, image_safety, materialise, repair, gate, rollout, sanity, fix_tasks
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

    def invoke(self, root, out, image_builds=None):
        argv = ['materialise.py', '--index', str(root / 'index.jsonl'), '--fixes', str(root / 'fixes'),
                '--tasks', str(root / 'ids.json'), '--src-task-data', str(root / 'source'),
                '--source-jsonl', str(root / 'original.jsonl'), '--out', str(out)]
        if image_builds is not None:
            argv.extend(['--image-builds', str(image_builds)])
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

    def test_modes_links_and_empty_directories_invalidate_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_source(root)
            corpus = root / 'corpus'; self.invoke(root, corpus)
            tests = corpus / 'task-data/task_a/tests'
            truth = tests / 'truth.json'
            before = image_safety.corpus_fingerprints(corpus)
            truth.chmod(0o600)
            self.assertNotEqual(image_safety.corpus_fingerprints(corpus), before)
            (tests / 'same-truth.json').write_bytes(truth.read_bytes())
            link = tests / 'oracle'; link.symlink_to('truth.json')
            before = image_safety.corpus_fingerprints(corpus)
            link.unlink(); link.symlink_to('same-truth.json')
            self.assertNotEqual(image_safety.corpus_fingerprints(corpus), before)
            before = image_safety.corpus_fingerprints(corpus)
            (tests / 'empty').mkdir()
            self.assertNotEqual(image_safety.corpus_fingerprints(corpus), before)

    def test_sanity_recheck_executes_only_final_task_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); self.make_source(root)
            (root / 'source/task_a/tests/oracle').symlink_to('truth.json')
            corpus = root / 'corpus'; self.invoke(root, corpus)
            self.assertTrue((corpus / 'task-data/task_a/tests/oracle').is_symlink())
            (corpus / 'artifacts.jsonl').write_text('{"task_id":"task_a","verifier_repaired":true}\n')
            seen = []
            async def fake_arm(tid, task_data, image, commands):
                self.assertTrue((Path(task_data) / tid / 'tests/oracle').is_symlink())
                seen.append((Path(task_data) / tid / 'tests/truth.json').exists())
                return {'reward': 0, 'observation': '', 'error': None}
            argv = ['sanity.py', '--corpus', str(corpus), '--out', str(root / 'sanity')]
            with patch.object(sys, 'argv', argv), patch.object(sanity, 'run_arm', fake_arm):
                sanity.main()
                (corpus / 'task-data/task_a/tests/truth.json').unlink()
                sanity.main()
            self.assertEqual(seen, [True, False])

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

    def test_linked_tests_directory_cannot_overwrite_original_verifier(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = self.make_source(root)
            protected = root / 'original-tests'
            (source / 'tests').rename(protected)
            (source / 'tests').symlink_to(protected, target_is_directory=True)
            original = (protected / 'test.sh').read_bytes()
            (root / 'fixes/task_a/repair.json').write_text('{"accepted":true,"action":"rewrite_verifier"}')
            (root / 'fixes/task_a/test.sh').write_text('replacement verifier')
            with self.assertRaises(ValueError):
                self.invoke(root, root / 'corpus')
            self.assertFalse((root / 'corpus').exists())
            self.assertEqual((protected / 'test.sh').read_bytes(), original)

    def test_oracle_overlay_replaces_link_without_changing_its_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = self.make_source(root)
            protected = root / 'original-truth'; protected.write_text('original oracle')
            (source / 'tests/truth.json').unlink()
            (source / 'tests/truth.json').symlink_to(protected)
            fix = root / 'fixes/task_a'
            (fix / 'repair.json').write_text(json.dumps({'accepted': True, 'action': 'rewrite_verifier',
                'leaked_paths': ['/app/truth.json'], 'oracle_to_tests': ['/app/truth.json']}))
            (fix / 'test.sh').write_text('echo 0 > /logs/verifier/reward.txt')
            (fix / 'tests_assets').mkdir()
            (fix / 'tests_assets/truth.json').write_text('extracted oracle')
            (fix / 'tests_assets/obsolete').write_text('stale asset')
            image = root / 'derived.sif'; image.write_bytes(b'derived')
            builds = {'task_a': {'image': str(image), 'registry_image': 'test:original',
                                 'removed_paths': ['/app/truth.json']}}
            with patch.object(materialise, 'checked_builds', return_value=builds):
                self.invoke(root, root / 'corpus', image_builds=root / 'builds')
            self.assertEqual(protected.read_text(), 'original oracle')
            adopted = root / 'corpus/task-data/task_a/tests/truth.json'
            self.assertFalse(adopted.is_symlink())
            self.assertEqual(adopted.read_text(), 'extracted oracle')
            self.assertFalse((adopted.parent / 'obsolete').exists())

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

    def test_hybrid_repairs_cannot_bypass_content_guards(self):
        proposal = {'action': 'rewrite_verifier', 'setup_sh': 'rm -rf /app/input',
                    'leaked_paths': ['/app/input'], 'oracle_to_tests': [],
                    'test_sh': 'echo 0 > /logs/verifier/reward.txt'}
        task = {'instruction': 'read /app/input/data.csv', 'tests': ''}
        self.assertIn('instruction references', repair.check_repair(proposal, task))
        proposal.update(leaked_paths=['/app/oracle'], setup_sh='rm -rf /app/oracle',
                        test_sh='cat /app/oracle; echo 0 > /logs/verifier/reward.txt')
        self.assertIn('verifier still reads', repair.check_repair(proposal, task))
        proposal.update(action='remove_leak', test_sh='echo no reward')
        self.assertIn('never writes', repair.check_repair(proposal, task))
        proposal.update(test_sh='echo 0 > /logs/verifier/reward.txt')
        self.assertIsNone(repair.check_repair(proposal, task))

    def test_hybrid_removal_without_setup_rewrite_requests_image_build(self):
        proposal = {'action': 'rewrite_verifier', 'setup_sh': '', 'test_sh': 'echo 0 > /logs/verifier/reward.txt',
                    'leaked_paths': ['/app/oracle'], 'oracle_to_tests': [], 'notes': 'Remove the leaked oracle'}
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(proposal)))])
        task = {'task_id': 'task_a', 'instruction': 'Produce the result', 'setup': '', 'tests': 'old verifier'}
        with patch.object(fix_tasks, 'complete', return_value=response):
            result = fix_tasks.repair_one(task, {'label': 'VERIFIER-TOO-WEAK'}, 'test', 'test', None, 8192)
        self.assertTrue(result['accepted'])
        self.assertTrue(result['needs_image_rebuild'])

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


class VerifierEvidenceTests(unittest.TestCase):
    def run_result(self, reward, metadata):
        result = SimpleNamespace(reward=reward, metadata=metadata, done=True, result='verifier response')
        class FakeEnv:
            def __init__(self, **kwargs):
                pass
            async def setup(self):
                pass
            async def reset(self, **kwargs):
                return SimpleNamespace(result='task'), []
            async def step(self, call):
                return result
            async def close(self):
                pass
        env_call = lambda **kwargs: SimpleNamespace(**kwargs)
        with patch.object(gate, '_import_env', return_value=(FakeEnv, env_call)):
            arm = asyncio.run(gate.run_arm('task_a', '/unused', 'fake.sif', []))
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content=f'```bash\n{rollout.SUBMIT_HINT}\n```', tool_calls=None))])
        with patch.object(rollout, '_import_env', return_value=(FakeEnv, env_call)), patch.object(rollout, 'complete', return_value=response):
            episode = asyncio.run(rollout.run_one('task_a', '/unused', 'fake.sif', 'test', 'test', None, 2, 0))
        return arm, episode

    def test_missing_reward_is_no_evidence_in_sanity_and_rollouts(self):
        arm, episode = self.run_result(0, {'invalid_reward': True})
        self.assertIsNone(arm['reward'])
        self.assertEqual(sanity.classify(arm), 'NO_REWARD')
        self.assertEqual(episode['outcome'], 'invalid_reward')
        status, details = export_verified.classify([episode])
        self.assertEqual(status, 'not_evaluable')
        self.assertEqual(details['n_no_evidence'], 1)

    def test_timeout_and_infrastructure_are_never_scored_zeroes(self):
        for metadata, sanity_status, outcome, export_status in [
                ({'timeout': True}, 'TIMEOUT', 'timeout', 'budget_limited'),
                ({'infrastructure_failure': True}, 'INFRA', 'infra', 'not_evaluable')]:
            with self.subTest(metadata=metadata):
                arm, episode = self.run_result(0, metadata)
                self.assertEqual(sanity.classify(arm), sanity_status)
                self.assertEqual(episode['outcome'], outcome)
                self.assertEqual(export_verified.classify([episode])[0], export_status)

    def test_actual_scored_rejection_remains_negative_evidence(self):
        arm, episode = self.run_result(0, {})
        self.assertEqual(sanity.classify(arm), 'ok_rejects_empty')
        self.assertEqual(episode['outcome'], 'unsolved')
        self.assertEqual(export_verified.classify([episode])[0], 'unverified')

    def test_missing_or_nonfinite_reward_is_not_a_score_without_metadata(self):
        for reward in [None, float('nan'), float('inf'), -1, 2]:
            with self.subTest(reward=reward):
                arm, episode = self.run_result(reward, {})
                self.assertEqual(sanity.classify(arm), 'NO_REWARD')
                self.assertEqual(episode['outcome'], 'invalid_reward')


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
