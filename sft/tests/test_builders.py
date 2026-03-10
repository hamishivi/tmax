"""Unit tests for preprocessing.builders."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.builders import (
    build_reasoning_content,
    build_submit_messages,
    build_tool_calls,
    build_tool_result,
)


# ── build_reasoning_content ───────────────────────────────────────────

class TestBuildReasoningContent:
    def test_both_fields(self):
        parsed = {"analysis": "State is clean.", "plan": "Run ls."}
        result = build_reasoning_content(parsed, "")
        assert "State is clean." in result
        assert "Run ls." in result

    def test_analysis_only(self):
        parsed = {"analysis": "Checking...", "plan": ""}
        result = build_reasoning_content(parsed, "")
        assert result == "Checking..."

    def test_with_surrounding_prose(self):
        parsed = {"analysis": "A", "plan": "P"}
        result = build_reasoning_content(parsed, "Thinking aloud...")
        assert result.startswith("Thinking aloud...")
        assert "A" in result
        assert "P" in result

    def test_empty_everything(self):
        result = build_reasoning_content({"analysis": "", "plan": ""}, "")
        assert result == ""


# ── build_tool_calls ──────────────────────────────────────────────────

class TestBuildToolCalls:
    def test_single_command(self):
        parsed = {"commands": [{"keystrokes": "ls -la\n", "duration": 0.1}]}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "bash"
        assert calls[0]["function"]["arguments"] == {"command": "ls -la"}
        assert calls[0]["id"].startswith("call_")
        assert calls[0]["type"] == "function"

    def test_multiple_commands(self):
        parsed = {
            "commands": [
                {"keystrokes": "cd /testbed\n", "duration": 0.1},
                {"keystrokes": "grep -r 'bug' .\n", "duration": 1.0},
            ]
        }
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert len(calls) == 1
        assert calls[0]["function"]["arguments"]["command"] == "cd /testbed\ngrep -r 'bug' ."

    def test_special_keystroke_ctrl_c(self):
        parsed = {"commands": [{"keystrokes": "C-c", "duration": 0.1}]}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert calls[0]["function"]["arguments"]["command"] == "C-c"

    def test_special_keystroke_ctrl_d(self):
        parsed = {"commands": [{"keystrokes": "C-d", "duration": 0.1}]}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert calls[0]["function"]["arguments"]["command"] == "C-d"

    def test_mixed_normal_and_special(self):
        parsed = {
            "commands": [
                {"keystrokes": "python train.py\n", "duration": 0.1},
                {"keystrokes": "C-c", "duration": 0.5},
            ]
        }
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert calls[0]["function"]["arguments"]["command"] == "python train.py\nC-c"

    def test_wait_only_dropped(self):
        parsed = {"commands": [{"keystrokes": "", "duration": 5.0}]}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert calls == []

    def test_empty_commands(self):
        parsed = {"commands": []}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert calls == []

    def test_no_commands_key(self):
        parsed = {"analysis": "just thinking"}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert calls == []

    def test_deterministic_ids(self):
        parsed = {"commands": [{"keystrokes": "ls\n"}]}
        calls_a = build_tool_calls(parsed, "trace_1", 0)
        calls_b = build_tool_calls(parsed, "trace_1", 0)
        assert calls_a[0]["id"] == calls_b[0]["id"]

    def test_different_turns_different_ids(self):
        parsed = {"commands": [{"keystrokes": "ls\n"}]}
        calls_0 = build_tool_calls(parsed, "trace_1", 0)
        calls_1 = build_tool_calls(parsed, "trace_1", 1)
        assert calls_0[0]["id"] != calls_1[0]["id"]

    def test_arguments_is_dict_not_string(self):
        parsed = {"commands": [{"keystrokes": "echo hi\n"}]}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert isinstance(calls[0]["function"]["arguments"], dict)

    def test_command_with_quotes(self):
        parsed = {"commands": [{"keystrokes": "echo \"hello world\"\n"}]}
        calls = build_tool_calls(parsed, "trace_1", 0)
        assert calls[0]["function"]["arguments"]["command"] == 'echo "hello world"'


# ── build_tool_result ─────────────────────────────────────────────────

class TestBuildToolResult:
    def test_standard_output(self):
        content = (
            "Current terminal state:\n"
            "New Terminal Output:\n"
            "root@host:/workspace# ls\n"
            "file.txt\n"
            "root@host:/workspace#\n"
            "\n\n\n"
        )
        result = build_tool_result(content, "call_abc123")
        assert result["role"] == "tool"
        assert result["tool_call_ids"] == ["call_abc123"]
        assert "root@host:/workspace# ls" in result["content"]
        assert result["content"].endswith("root@host:/workspace#")

    def test_confirmation_prompt_stripped(self):
        content = (
            "Current terminal state:\n"
            "New Terminal Output:\n"
            "root@host:/workspace#\n"
            "\n"
            "Are you sure you want to mark the task as complete? This will trigger...\n"
        )
        result = build_tool_result(content, "call_abc123")
        assert "Are you sure" not in result["content"]

    def test_content_is_flat_string(self):
        content = "Current terminal state:\nNew Terminal Output:\noutput here"
        result = build_tool_result(content, "call_x")
        assert isinstance(result["content"], str)

    def test_preserves_cwd_prompts(self):
        content = (
            "Current terminal state:\n"
            "New Terminal Output:\n"
            "root@abc:/testbed/src# grep -r 'pattern' .\n"
            "./file.py:pattern found\n"
            "root@abc:/testbed/src#\n\n\n"
        )
        result = build_tool_result(content, "call_x")
        assert "root@abc:/testbed/src#" in result["content"]


# ── build_submit_messages ─────────────────────────────────────────────

class TestBuildSubmitMessages:
    def test_task_complete_true_no_commands(self):
        parsed = {"task_complete": True, "analysis": "Done.", "plan": "Submit."}
        msgs = build_submit_messages(parsed, "trace_1", 2, "Done reasoning.")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "bash"
        assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in msgs[0]["tool_calls"][0]["function"]["arguments"]["command"]
        assert msgs[0]["reasoning_content"] == "Done reasoning."

    def test_task_complete_true_with_commands(self):
        parsed = {
            "task_complete": True,
            "commands": [{"keystrokes": "cat result.txt\n"}],
        }
        msgs = build_submit_messages(parsed, "trace_1", 2, "reasoning")
        assert len(msgs) == 1
        assert "reasoning_content" not in msgs[0]

    def test_task_complete_false(self):
        parsed = {"task_complete": False}
        msgs = build_submit_messages(parsed, "trace_1", 0, "")
        assert msgs == []

    def test_task_complete_absent(self):
        parsed = {"analysis": "working"}
        msgs = build_submit_messages(parsed, "trace_1", 0, "")
        assert msgs == []

    def test_submit_id_is_deterministic(self):
        parsed = {"task_complete": True}
        msgs_a = build_submit_messages(parsed, "trace_1", 2, "")
        msgs_b = build_submit_messages(parsed, "trace_1", 2, "")
        assert msgs_a[0]["tool_calls"][0]["id"] == msgs_b[0]["tool_calls"][0]["id"]
