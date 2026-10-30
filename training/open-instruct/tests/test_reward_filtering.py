"""Exercise the production validity/filter block without the GPU data loader."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


def production_functions():
    path = Path(__file__).parents[1] / "open_instruct/data_loader.py"
    tree = ast.parse(path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"rollout_reward_is_valid", "compute_group_advantages"}
    ]
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class RewardFilteringTest(unittest.TestCase):
    def test_invalid_rollouts_excluded_from_population_but_siblings_retained(self):
        path = Path(__file__).parents[1] / "open_instruct/data_loader.py"
        tree = ast.parse(path.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "accumulate_inference_batches"
        )
        loop = next(node for node in function.body if isinstance(node, ast.While))
        start = next(
            i
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "states" for target in node.targets)
        )
        end = next(
            i
            for i in range(start, len(loop.body))
            if isinstance(loop.body[i], ast.If)
            and "len(valid_rewards) < min_valid_group_size" in ast.unparse(loop.body[i].test)
        )
        body = loop.body[start : end + 1] + ast.parse("accepted.append(result)").body
        replay = ast.For(
            target=ast.Name(id="result", ctx=ast.Store()),
            iter=ast.Name(id="results", ctx=ast.Load()),
            body=body,
            orelse=[],
        )
        metrics = Mock()
        group = SimpleNamespace(
            prompt_id="p",
            responses=[[1], [2], [3]],
            finish_reasons=["stop"] * 3,
            reward_scores=[0.5, 0, 0],
            request_info=SimpleNamespace(
                rollout_states=[{"info": {}}, {"info": {"invalid_reward": True}}, {"timeout": True, "info": {}}]
            ),
        )
        namespace = {
            **production_functions(),
            "results": [group],
            "accepted": [],
            "population_metrics": metrics,
            "logger": Mock(),
            "min_valid_group_size": 2,
            "replenish_prompts": True,
            "replenish_accepted_prompts": False,
            "enqueue_next_prompt": Mock(),
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[replay], type_ignores=[])), str(path), "exec"),
            namespace,
        )
        metrics.add_group.assert_called_once_with([0.5, 0], ["stop", "stop"], [1, 1])
        self.assertEqual(namespace["accepted"], [group])
        namespace["enqueue_next_prompt"].assert_not_called()

        # With only one valid sibling there is no group-relative signal.
        group.request_info.rollout_states[0]["info"]["infrastructure_failure"] = True
        namespace["accepted"] = []
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[replay], type_ignores=[])), str(path), "exec"),
            namespace,
        )
        self.assertEqual(namespace["accepted"], [])
        namespace["enqueue_next_prompt"].assert_called_once()

    def test_smaller_groups_match_independent_reference(self):
        compute = production_functions()["compute_group_advantages"]
        scores = np.array([0.0, np.nan, 0.5, 1.0, 1.0, 0.0, np.inf, 0.25])
        valid = np.isfinite(scores)
        for mode in ("standard", "centered", "maxrl"):
            actual = compute(scores, 4, mode, valid)
            expected = np.zeros(8)
            for start in (0, 4):
                idx = np.arange(start, start + 4)[valid[start : start + 4]]
                group = scores[idx]
                centered = group - group.mean()
                expected[idx] = (
                    centered / (group.std() + 1e-8)
                    if mode == "standard"
                    else centered / group.mean()
                    if mode == "maxrl"
                    else centered
                )
            np.testing.assert_allclose(actual, expected)
            self.assertTrue(np.isfinite(actual).all())

    def test_empty_singleton_and_constant_groups(self):
        compute = production_functions()["compute_group_advantages"]
        for mode in ("standard", "centered", "maxrl"):
            for valid in ([False] * 4, [True, False, False, False], [True, True, False, True]):
                np.testing.assert_array_equal(compute(np.ones(4), 4, mode, valid), np.zeros(4))

    def test_unchanged_complete_groups(self):
        compute = production_functions()["compute_group_advantages"]
        scores = np.array([0, 0.5, 1, 0, 1, 1, 0.25, 0.75], dtype=np.float32)
        groups = scores.reshape(-1, 4)
        for mode in ("standard", "centered", "maxrl"):
            centered = groups - groups.mean(axis=1, keepdims=True)
            expected = (
                centered / (groups.std(axis=1, keepdims=True) + 1e-8)
                if mode == "standard"
                else centered / groups.mean(axis=1, keepdims=True)
                if mode == "maxrl"
                else centered
            )
            np.testing.assert_allclose(compute(scores, 4, mode), expected.ravel(), atol=1e-7)

    def test_invalid_mask_shape_fails(self):
        with self.assertRaises(ValueError):
            production_functions()["compute_group_advantages"](np.ones(4), 4, "standard", [True])

    def test_packing_filter_preserves_original_prompt_ids(self):
        path = Path(__file__).parents[1] / "open_instruct/data_loader.py"
        tree = ast.parse(path.read_text())
        nodes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in {"invalid_reward_idxes", "do_mask_filter"} for t in node.targets
            ):
                nodes.append(node)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "do_mask_filter":
                nodes.append(node)
        nodes.sort(key=lambda n: n.lineno)
        config = SimpleNamespace(
            mask_truncated_completions=False,
            mask_non_submitting_completions=False,
            mask_non_submitting_completions_percent=0,
        )
        result = SimpleNamespace(
            **{key: list(range(8)) for key in ("responses", "masks", "finish_reasons", "logprobs")}
        )
        namespace = dict(
            np=np,
            self=SimpleNamespace(config=config),
            logger=Mock(),
            valid_rewards=np.array([True, False, True, True, True, True, False, True]),
            truncated_idxes=[],
            non_submitting_idxes=[],
            num_before_filter=8,
            num_unmasked_non_submitting_completion=0,
            scores=np.arange(8),
            raw_scores=np.arange(8),
            advantages=np.arange(8),
            batch=np.arange(8),
            result=result,
            rollout_sample_ids=list(range(8)),
            rollout_model_steps=list(range(8)),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        kept = [0, 2, 3, 4, 5, 7]
        for key in ("scores", "raw_scores", "advantages", "batch", "rollout_sample_ids", "rollout_model_steps"):
            np.testing.assert_array_equal(namespace[key], kept)
        for value in vars(result).values():
            self.assertEqual(value, kept)
        # The second prompt remains prompt 1 despite the missing first sibling.
        self.assertEqual([i // 4 for i in namespace["rollout_sample_ids"]], [0, 0, 0, 1, 1, 1])

    def test_reward_diagnostics_ignore_invalid_placeholders(self):
        path = Path(__file__).parents[1] / "open_instruct/data_loader.py"
        tree = ast.parse(path.read_text())
        block = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "self.config.add_concave_length_penalty and len(scores) > 0"
        )
        start = next(
            i
            for i, node in enumerate(block.body)
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "solved_mask_raw"
        )
        namespace = dict(
            np=np,
            self=SimpleNamespace(config=SimpleNamespace(max_possible_score=1, num_samples_per_prompt_rollout=3)),
            valid_rewards=np.array([True, False, True]),
            raw_scores=np.array([1.0, 0.0, 0.5]),
            scores=np.array([0.8, -999.0, 0.4]),
            concave_length_penalties=np.array([0.2, 999.0, 0.1]),
            concave_length_x=np.ones(3),
        )
        exec(compile(ast.Module(body=block.body[start:], type_ignores=[]), str(path), "exec"), namespace)
        metrics = namespace["concave_length_metrics"]
        self.assertAlmostEqual(metrics["concave_length_penalty/raw_score_mean"], 0.75)
        self.assertAlmostEqual(metrics["concave_length_penalty/shaped_score_mean"], 0.6)
        self.assertAlmostEqual(metrics["concave_length_penalty/penalty_unsolved_mean"], 0.1)

        group_count = next(
            value
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "val/total_reward_groups"
        )
        namespace["rollout_sample_ids"] = [0, 2, 3, 4]
        self.assertEqual(eval(compile(ast.Expression(group_count), str(path), "eval"), namespace), 2)
