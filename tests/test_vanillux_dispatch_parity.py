"""Exercise the actual rollout selector/parser without importing the GPU stack."""

from __future__ import annotations

import ast
import json
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training/open-instruct/open_instruct"
FORMAT_FEEDBACK = "Invalid bash tool call"


def load_definition(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@dataclass
class Call:
    id: str
    name: str
    args: dict


PARSER_NAMESPACE = {"dataclass": dataclass, "field": field, "EnvCall": Call, "json": json, "logger": Mock()}
ParseResult = load_definition(TRAINING / "environments/tools/parsers.py", "ToolCallParseResult", PARSER_NAMESPACE)
parse_calls = load_definition(
    TRAINING / "environments/tools/parsers.py", "parse_tool_calls", PARSER_NAMESPACE, "VllmToolParser"
)


def load_dispatch():
    """Compile the production call selection and feedback accounting blocks."""
    path = TRAINING / "vllm_utils.py"
    tree = ast.parse(path.read_text())
    process_request = next(
        node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "process_request"
    )
    loop = next(node for node in ast.walk(process_request) if isinstance(node, ast.While))

    def initializer(name):
        return next(
            node
            for node in loop.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name
        )

    selection = next(
        node
        for node in loop.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.values[0], ast.Name)
        and node.test.values[0].id == "tool_call_format_error_message"
    )
    accounting = next(
        node
        for node in loop.body
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id == "format_error_feedback"
    )
    start = next(
        i
        for i, node in enumerate(loop.body)
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "tool_calls" for target in node.targets)
    )
    end = loop.body.index(initializer("format_error_feedback"))
    calls = compile(ast.Module(body=loop.body[start:end], type_ignores=[]), str(path), "exec")
    feedback = compile(
        ast.Module(
            body=[initializer("format_error_feedback"), selection, initializer("observations"), accounting],
            type_ignores=[],
        ),
        str(path),
        "exec",
    )
    return calls, feedback


CALL_DISPATCH, FEEDBACK_DISPATCH = load_dispatch()


def select_calls(parse_result, allowed_tools, active_env_names):
    namespace = {
        "parse_result": parse_result,
        "allowed_tools": allowed_tools,
        "pool_setup": SimpleNamespace(active_env_names=active_env_names),
    }
    exec(CALL_DISPATCH, namespace)
    return namespace["tool_calls"]


class DispatchParityTests(unittest.TestCase):
    def select(self, calls, *, invalid_first=False, vanillux=True):
        return select_calls(
            ParseResult(tool_calls=calls, had_tool_call=bool(calls), first_tool_call_invalid=invalid_first),
            {"bash", "python"},
            ["swerl_vanillux_sandbox" if vanillux else "generic_sandbox"],
        )

    def test_only_first_command_runs(self):
        calls = [Call("1", "bash", {"command": "pwd"}), Call("2", "bash", {"command": "false"})]
        self.assertEqual(self.select(calls), calls[:1])
        self.assertEqual(self.select(calls, vanillux=False), calls)

    def test_invalid_first_call_never_promotes_later_bash(self):
        bash = Call("2", "bash", {"command": "pwd"})
        for first in [Call("1", "python", {}), Call("1", "bash", {"command": 42})]:
            with self.subTest(first=first):
                self.assertEqual(self.select([first, bash]), [])
        self.assertEqual(self.select([bash], invalid_first=True), [])

    def test_empty_commands_and_no_calls(self):
        self.assertEqual(self.select([]), [])
        for args in [{}, {"command": ""}, {"command": "  \n "}]:
            call = Call("1", "bash", args)
            self.assertEqual(self.select([call]), [call])

    def test_native_parser_keeps_malformed_first_position(self):
        for malformed in ["{", "[]", "null", "42", None]:
            with self.subTest(malformed=malformed):
                native_result = SimpleNamespace(
                    tools_called=True,
                    tool_calls=[
                        SimpleNamespace(id="1", function=SimpleNamespace(name="bash", arguments=malformed)),
                        SimpleNamespace(id="2", function=SimpleNamespace(name="bash", arguments='{"command":"pwd"}')),
                    ],
                )
                fake = SimpleNamespace(
                    _make_request=lambda: None,
                    tool_parser=SimpleNamespace(extract_tool_calls=lambda _result=native_result, **kwargs: _result),
                )
                parsed = parse_calls(fake, "model output")
                self.assertTrue(parsed.first_tool_call_invalid)
                self.assertEqual(len(parsed.tool_calls), 1)
                self.assertEqual(select_calls(parsed, {"bash"}, ["swerl_vanillux_sandbox"]), [])
                # Other environments retain their previous valid-call policy.
                self.assertEqual(select_calls(parsed, {"bash"}, ["generic_sandbox"]), parsed.tool_calls)


class DispatchFeedbackTests(unittest.TestCase):
    def dispatch_feedback(self, *, vanillux, enabled, had_tool_call=True, step_count=0):
        rollout = SimpleNamespace(step_count=step_count, tool_error="", rewards=[], tool_call_stats=[])
        namespace = {
            "tool_call_format_error_message": FORMAT_FEEDBACK if enabled else None,
            "text_env_names": [],
            "pool_setup": SimpleNamespace(
                active_env_names=["swerl_vanillux_sandbox" if vanillux else "generic_sandbox"]
            ),
            "parse_result": ParseResult(had_tool_call=had_tool_call),
            "tool_calls": [],
            "allowed_tools": {"bash"},
            "rollout": rollout,
            "max_steps": 64,
            "ToolCallStats": SimpleNamespace,
        }
        exec(FEEDBACK_DISPATCH, namespace)
        return rollout, namespace["observations"]

    def assert_feedback(self, rollout, observations, role, *, previous_steps=0):
        self.assertEqual(observations, [(FORMAT_FEEDBACK, role)])
        self.assertEqual(rollout.step_count, previous_steps + 1)
        self.assertEqual(rollout.rewards, [0.0])
        self.assertEqual(rollout.tool_error, FORMAT_FEEDBACK)
        self.assertEqual(len(rollout.tool_call_stats), 1)
        self.assertEqual(rollout.tool_call_stats[0].tool_name, "tool_call_format_error")
        self.assertFalse(rollout.tool_call_stats[0].success)

    def assert_no_feedback(self, rollout, observations, *, previous_steps=0):
        self.assertEqual(observations, [])
        self.assertEqual(rollout.step_count, previous_steps)
        self.assertEqual(rollout.rewards, [])
        self.assertEqual(rollout.tool_error, "")
        self.assertEqual(rollout.tool_call_stats, [])

    def test_vanillux_recovery_uses_user_role_and_consumes_step_budget(self):
        for had_tool_call in (False, True):
            for step_count in (0, 63, 64):
                with self.subTest(had_tool_call=had_tool_call, step_count=step_count):
                    rollout, observations = self.dispatch_feedback(
                        vanillux=True, enabled=True, had_tool_call=had_tool_call, step_count=step_count
                    )
                    if step_count == 64:
                        self.assert_no_feedback(rollout, observations, previous_steps=step_count)
                    else:
                        self.assert_feedback(rollout, observations, "user", previous_steps=step_count)

    def test_generic_recovery_keeps_existing_roles_and_budget_accounting(self):
        for had_tool_call, role in ((False, "user"), (True, "tool")):
            with self.subTest(had_tool_call=had_tool_call):
                rollout, observations = self.dispatch_feedback(
                    vanillux=False, enabled=True, had_tool_call=had_tool_call
                )
                self.assert_feedback(rollout, observations, role)

    def test_disabling_generic_recovery_does_not_inject_or_count_feedback(self):
        rollout, observations = self.dispatch_feedback(vanillux=False, enabled=False)
        self.assert_no_feedback(rollout, observations)


if __name__ == "__main__":
    unittest.main()
