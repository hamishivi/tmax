"""CPU-only tests of the actual evaluation and training protocol methods.

Run with ``python -m unittest discover -s tests -p 'test_vanillux_protocol_parity.py'``.
Only PyYAML is needed: AST extraction avoids importing the GPU training stack,
Harbor, or LiteLLM while executing the production methods unchanged.
"""

import ast
import asyncio
import contextlib
import copy
import json
import os
import re
import shlex
import tempfile
import time
import unittest
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "rl_data/generator"
TRAIN_DIR = ROOT / "training/open-instruct/open_instruct/environments"
AGENT_PATH = ROOT / "Vanillux2Agent/agent.py"
ENV_PATH = TRAIN_DIR / "swerl_vanillux_sandbox.py"


def load_definitions(path, names, namespace, class_name=None):
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    selected = []
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(getattr(target, "id", None) in names for target in node.targets):
            selected.append(node)
    if len(selected) != len(names):
        raise AssertionError(f"Missing definitions {names} in {path}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)


def tool_call(arguments, name="bash", call_id="first"):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


@dataclass
class ProtocolStepResult:
    result: str = ""
    reward: float = 0.0
    done: bool = False
    metadata: dict = field(default_factory=dict)


class TestVanilluxProtocolParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eval_prompts = yaml.safe_load((EVAL_DIR / "vanillux_prompts.yaml").read_text())
        cls.train_prompts = yaml.safe_load((TRAIN_DIR / "vanillux_prompts.yaml").read_text())
        eval_namespace = {
            "_OBS_CFG": cls.eval_prompts["observation"],
            "_INSTANCE_TEMPLATE": cls.eval_prompts["instance_template"],
            "_FORMAT_ERROR_TEMPLATE": cls.eval_prompts["format_error_template"],
            "_SYSTEM_TEMPLATE": cls.eval_prompts["system_template"],
        }
        load_definitions(
            EVAL_DIR / "vanillux_solver.py",
            {"_truncate_observation", "_render_instance", "_format_error_message"},
            eval_namespace,
        )
        eval_namespace.update(
            Any=Any,
            Dict=dict,
            Optional=Optional,
            Path=Path,
            deepcopy=copy.deepcopy,
            os=os,
            re=re,
            BaseEnvironment=Any,
            AgentContext=Any,
            asyncio=asyncio,
            contextlib=contextlib,
            time=time,
            json=json,
            logger=Mock(),
            litellm=SimpleNamespace(
                completion_cost=lambda *args, **kwargs: 0.0,
                exceptions=SimpleNamespace(
                    ContextWindowExceededError=type("ContextWindowExceededError", (Exception,), {})
                ),
            ),
        )
        eval_namespace["HARNESS_CONFIG_DIR"] = ROOT / "sft/preprocessing/config"
        load_definitions(
            EVAL_DIR / "sample_solutions.py",
            {
                "SUBMIT_MARKER",
                "_DEFAULT_TOOL_SCHEMAS",
                "_DEFAULT_SYSTEM_PROMPT",
                "_load_harness_config",
                "_extract_tool_call",
            },
            eval_namespace,
        )
        eval_namespace["TOOL_SCHEMAS"] = eval_namespace["_load_harness_config"]()[1]
        load_definitions(
            AGENT_PATH,
            {
                "_COMPOSE_PROVIDER_RE",
                "_DOCKER_EXEC_ERROR_RE",
                "MAX_RETRIES",
                "RETRY_BASE_DELAY",
                "LLM_TIMEOUT_SECONDS",
                "LLM_OUTER_TIMEOUT_BUFFER_SECONDS",
            },
            eval_namespace,
        )
        load_definitions(
            AGENT_PATH,
            {"_format_tool_result", "_append_format_error", "_query_with_retry", "run"},
            eval_namespace,
            "Vanillux2Agent",
        )
        cls.eval_namespace = eval_namespace

        observation = cls.train_prompts["observation"]
        train_namespace = {
            "_OBS_MAX_CHARS": observation["max_chars"],
            "_OBS_HEAD_CHARS": observation["head_chars"],
            "_OBS_TAIL_CHARS": observation["tail_chars"],
            "_OBS_TOO_LONG_HINT": observation["too_long_hint"],
            "INSTANCE_TEMPLATE": cls.train_prompts["instance_template"],
            "FORMAT_ERROR_TEMPLATE": cls.train_prompts["format_error_template"],
            "StepResult": ProtocolStepResult,
            "shlex": shlex,
            "uuid": uuid,
            "_BASH_WRAPPER_PATH_QUOTED": shlex.quote("/tmp/.swerl_vanillux_bash_wrapper.sh"),
            "SUBMIT_MARKER": eval_namespace["SUBMIT_MARKER"],
            "re": re,
        }
        load_definitions(
            ENV_PATH,
            {"_BASH_TOOL", "TOOL_CALL_FORMAT_ERROR_MESSAGE", "_COMPOSE_PROVIDER_RE", "_DOCKER_EXEC_ERROR_RE"},
            train_namespace,
        )
        load_definitions(
            ENV_PATH, {"truncate_observation", "render_instance", "format_error_message"}, train_namespace
        )
        load_definitions(ENV_PATH, {"_execute_bash"}, train_namespace, "SWERLVanilluxSandboxEnv")
        cls.train_namespace = train_namespace

    def training_result(self, command, stdout="", stderr="", code=0, completion_stdout="124\n", completion_code=0):
        backend = Mock()
        execution = SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=code, timed_out=None)
        if code == 124:
            backend.run_command.side_effect = [
                execution,
                SimpleNamespace(stdout=completion_stdout, stderr="", exit_code=completion_code),
            ]
        else:
            backend.run_command.return_value = execution
        env = SimpleNamespace(
            _backend=backend,
            _run_tests=Mock(return_value=ProtocolStepResult("verified", reward=1.0, done=True)),
            _turns_remaining_message=lambda: None,
            _timeout=120,
            _task_id="parity-test",
        )
        result = self.train_namespace["_execute_bash"](env, {"command": command})
        return result, env

    def eval_result(self, stdout, stderr, code):
        return self.eval_namespace["_format_tool_result"](
            SimpleNamespace(stdout=stdout, stderr=stderr, return_code=code)
        )

    def eval_run(self, calls_per_turn, stdout="", stderr="", code=0, max_steps=None, timed_out=False):
        def response(calls):
            message = {"role": "assistant", "content": "Working on it.", "tool_calls": calls}
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(model_dump=lambda: message))])

        with tempfile.TemporaryDirectory() as directory:
            agent = SimpleNamespace(
                model_name="fake-model",
                max_steps=len(calls_per_turn) if max_steps is None else max_steps,
                cost_limit=0.0,
                cost=0.0,
                command_timeout=120,
                logs_dir=Path(directory),
                _query_with_retry=AsyncMock(side_effect=[response(calls) for calls in calls_per_turn]),
                _accumulate_usage=lambda *args: None,
                _append_format_error=self.eval_namespace["_append_format_error"],
                _execute_bash=AsyncMock(
                    return_value=SimpleNamespace(stdout=stdout, stderr=stderr, return_code=code, timed_out=timed_out)
                ),
                _format_tool_result=self.eval_namespace["_format_tool_result"],
            )
            asyncio.run(self.eval_namespace["run"](agent, "Task", object(), SimpleNamespace()))
            messages = json.loads((Path(directory) / "trajectory.json").read_text())
            agent.timing = json.loads((Path(directory) / "timing.json").read_text())
        return agent, messages

    def test_prompts_and_rendering_match(self):
        self.assertEqual(self.eval_prompts, self.train_prompts)
        system_prompt = (
            ROOT / "training/open-instruct/scripts/train/debug/envs/swerl_vanillux_sandbox_system_prompt.txt"
        )
        self.assertEqual(system_prompt.read_text(), self.eval_prompts["system_template"])
        task = "Task with {{task}}, spaces, and\nmultiple lines."
        self.assertEqual(self.eval_namespace["_render_instance"](task), self.train_namespace["render_instance"](task))
        self.assertEqual(
            self.train_namespace["TOOL_CALL_FORMAT_ERROR_MESSAGE"],
            self.eval_namespace["_format_error_message"](
                "Your last response did not include a valid `bash` tool call."
            ),
        )

    def test_observations_match_for_streams_whitespace_and_truncation(self):
        provider_noise = (
            '\x1b[4m>>>> Executing external compose provider "/usr/bin/docker-compose". '
            "Please see podman-compose(1) for how to disable this message. <<<<\n\n\x1b[0m"
        )
        cases = [
            ("", "", 0),
            (None, None, 0),
            ("hello\n\n", "", 0),
            (" \n\t", "", 0),
            ("out\n", "err\n", 3),
            ("", "error\n", 1),
            ("payload\n", provider_noise, 0),
            ("payload\n", "Error: executing docker-compose exec task: exit status 1\n", 1),
            ("a" * 10000, "", 0),
            ("a" * 10001, "", 0),
            ("α" * 6000 + "🙂" * 6000, "", 0),
        ]
        for stdout, stderr, code in cases:
            with self.subTest(length=len(stdout or ""), code=code):
                actual, _ = self.training_result("printf payload", stdout, stderr, code)
                self.assertEqual(actual.result, self.eval_result(stdout, stderr, code))
                self.assertFalse(actual.done)
        self.assertEqual(self.eval_result("hello\n", "", 0), "hello\n\n(exit_code=0)")
        self.assertEqual(self.eval_result(" \n", "", 0), "(no output)\n\n(exit_code=0)")

    def test_submission_matches_reference_including_marker_edge_cases(self):
        marker = self.eval_namespace["SUBMIT_MARKER"]
        cases = [
            (f"echo {marker}", f"{marker}\n", "", True),
            (f"printf %s {marker} >/tmp/answer", "", "", True),
            (f"false # {marker}", "", "", True),
            ("cat result.txt", f"{marker}\n", "", True),
            ("cat result.txt >&2", "", marker, True),
            ("cat large.txt", "a" * 6000 + marker + "b" * 6000, "", False),
            ("cat small.txt", "unfinished", "", False),
        ]
        for command, stdout, stderr, expected in cases:
            with self.subTest(command=command):
                agent, _ = self.eval_run(
                    [[tool_call(json.dumps({"command": command}))], [tool_call('{"command":"next"}')]], stdout, stderr
                )
                self.assertEqual(agent._execute_bash.call_count, 1 if expected else 2)
                actual, env = self.training_result(command, stdout, stderr)
                self.assertEqual(actual.done, expected)
                self.assertEqual(env._run_tests.call_count, int(expected))

    def test_exit_124_does_not_end_episode_and_cleans_up_provenance(self):
        user_stderr = "timeout 1 was intentional\n"
        result, env = self.training_result(
            "timeout 1 sleep 2", "", f"Command timed out after 120s.\n{user_stderr}", 124
        )
        self.assertFalse(result.done)
        self.assertNotIn("timeout", result.metadata)
        self.assertEqual(result.result, self.eval_result("", user_stderr, 124))
        env._run_tests.assert_not_called()
        calls = env._backend.run_command.call_args_list
        status_path = shlex.split(calls[0].args[0])[-1]
        self.assertIn(f"cat {status_path}; rm -f {status_path}", calls[1].args[0])

    def test_actual_backend_deadline_is_terminal(self):
        result, env = self.training_result("sleep 500", "", "Command timed out after 120s.\n", 124, "")
        self.assertTrue(result.done)
        self.assertTrue(result.metadata["timeout"])
        self.assertEqual(result.reward, 0.0)
        self.assertIn("Command timed out after 120s.", result.result)
        env._run_tests.assert_not_called()

    def test_evaluation_deadline_continues_after_recording_output_even_with_submit_marker(self):
        marker = self.eval_namespace["SUBMIT_MARKER"]
        calls = [[tool_call(json.dumps({"command": f"sleep 500 # {marker}"}))], [tool_call('{"command":"next"}')]]
        agent, messages = self.eval_run(calls, stdout="partial output", code=124, timed_out=True)
        self.assertEqual(agent._execute_bash.await_count, 2)
        self.assertEqual(agent._query_with_retry.await_count, 2)
        self.assertEqual(messages[-1]["content"], "partial output\n\n(exit_code=124)")
        self.assertTrue(agent.timing[-1]["timed_out"])

    def test_evaluation_ordinary_exit_124_continues(self):
        calls = [[tool_call('{"command":"exit 124"}')], [tool_call('{"command":"next"}')]]
        agent, _ = self.eval_run(calls, code=124)
        self.assertEqual(agent._execute_bash.await_count, 2)
        self.assertFalse(agent.timing[-1]["timed_out"])

    def test_failed_provenance_probe_is_infrastructure_failure(self):
        result, env = self.training_result("exit 124", "", "", 124, "124\n", 1)
        self.assertTrue(result.done)
        self.assertTrue(result.metadata["infrastructure_failure"])
        self.assertNotIn("timeout", result.metadata)
        self.assertIn("probe exit_code=1", result.metadata["error"])
        env._run_tests.assert_not_called()

    def test_empty_and_whitespace_commands_are_noops(self):
        for command in ["", "  \n ", "  pwd\n"]:
            with self.subTest(command=command):
                result, env = self.training_result(command)
                self.assertEqual(result.result, "(no output)\n\n(exit_code=0)")
                self.assertEqual(shlex.split(env._backend.run_command.call_args.args[0])[2], command.strip())
                agent, _ = self.eval_run([[tool_call(json.dumps({"command": command}))]])
                self.assertEqual(agent._execute_bash.call_args.args[0], command.strip())

    def test_invalid_command_types_are_recovered_without_execution(self):
        for command in [None, 123, False, [], {}]:
            with self.subTest(command=command):
                result, env = self.training_result(command)
                env._backend.run_command.assert_not_called()
                self.assertEqual(result.result, self.train_namespace["TOOL_CALL_FORMAT_ERROR_MESSAGE"])
                self.assertFalse(result.done)
        for args in (None, 42, [], "text"):
            with self.subTest(args=args):
                env = SimpleNamespace(_backend=Mock())
                result = self.train_namespace["_execute_bash"](env, args)
                env._backend.run_command.assert_not_called()
                self.assertEqual(result.result, self.train_namespace["TOOL_CALL_FORMAT_ERROR_MESSAGE"])

    def test_format_feedback_uses_valid_user_message(self):
        messages = [{"role": "assistant", "content": "invalid tool request"}]
        self.eval_namespace["_append_format_error"](messages)
        self.assertEqual(
            messages[-1], {"role": "user", "content": self.train_namespace["TOOL_CALL_FORMAT_ERROR_MESSAGE"]}
        )

    def test_format_errors_use_only_the_normal_turn_budget(self):
        calls = [[tool_call("{")]] * 65 + [[tool_call('{"command":"pwd"}')]]
        for max_steps, commands in [(64, []), (66, ["pwd"])]:
            with self.subTest(max_steps=max_steps):
                agent, _ = self.eval_run(calls, max_steps=max_steps)
                self.assertEqual(agent._query_with_retry.await_count, max_steps)
                self.assertEqual([call.args[0] for call in agent._execute_bash.call_args_list], commands)

    def test_agent_run_recovers_and_keeps_first_call_transcript_consistent(self):
        submit = "echo " + self.eval_namespace["SUBMIT_MARKER"]
        agent, messages = self.eval_run(
            [
                [tool_call("{}", name="python", call_id="bad")],
                [tool_call('{"command":"  pwd\\n"}'), tool_call('{"command":"ignored"}', call_id="ignored")],
                [tool_call(json.dumps({"command": submit}), call_id="submit")],
            ]
        )
        self.assertEqual([entry.args[0] for entry in agent._execute_bash.call_args_list], ["pwd", submit])
        self.assertEqual(
            messages[3], {"role": "user", "content": self.train_namespace["TOOL_CALL_FORMAT_ERROR_MESSAGE"]}
        )
        assistant_calls = [call["id"] for message in messages for call in message.get("tool_calls", [])]
        tool_responses = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
        self.assertEqual(assistant_calls, ["first", "submit"])
        self.assertEqual(tool_responses, assistant_calls)

    def test_agent_run_recovers_from_malformed_json_and_nonstring_commands(self):
        invalid_args = ["{", "[]", "null", '"text"', "123", None]
        invalid_args += [json.dumps({"command": command}) for command in (None, 42, False, [], {})]
        for arguments in invalid_args:
            with self.subTest(arguments=arguments):
                agent, messages = self.eval_run(
                    [
                        [tool_call(arguments, call_id="bad"), tool_call('{"command":"ignored"}', call_id="ignored")],
                        [tool_call('{"command":"pwd"}')],
                    ]
                )
                agent._execute_bash.assert_awaited_once()
                self.assertEqual(agent._execute_bash.call_args.args[0], "pwd")
                self.assertNotIn("tool_calls", messages[2])
                self.assertEqual(
                    messages[3], {"role": "user", "content": self.train_namespace["TOOL_CALL_FORMAT_ERROR_MESSAGE"]}
                )

    def test_tool_schema_matches_training_without_mutating_original(self):
        namespace = self.eval_namespace
        original = copy.deepcopy(namespace["TOOL_SCHEMAS"])
        completion = Mock(return_value=object())
        namespace["litellm"].completion = completion
        namespace["ABORT_EXCEPTIONS"] = ()
        for persistent in (True, False):
            with self.subTest(persistent=persistent):
                agent = SimpleNamespace(
                    api_base=None, persistent_bash=persistent, max_tokens=100, temperature=None, top_p=None, top_k=None
                )
                asyncio.run(namespace["_query_with_retry"](agent, "fake-model", []))
                schema = completion.call_args.kwargs["tools"][0]
                if persistent:
                    self.assertEqual(schema, self.train_namespace["_BASH_TOOL"])
                    self.assertIn("preserved between calls", schema["function"]["description"])
                else:
                    self.assertEqual(schema, original[0])
                self.assertEqual(namespace["TOOL_SCHEMAS"], original)


if __name__ == "__main__":
    unittest.main()
