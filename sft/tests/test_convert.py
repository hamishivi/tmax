"""Integration tests for preprocessing.convert (full trace conversion)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.convert import convert_trace
from preprocessing.builders import SUBMIT_COMMAND


# ── Fixture: a minimal valid Terminus-2 trace ─────────────────────────

def _make_trace(
    conversations: list[dict],
    trial_name: str = "task_0__abc123",
    **extra,
) -> dict:
    return {"conversations": conversations, "trial_name": trial_name, **extra}


SYSTEM_SECTION = (
    "You are an AI assistant...\nFormat your response as JSON...\n"
)
TASK_SECTION = (
    "Task Description:\n"
    "# Terminal Automation Request\n\n"
    "## Goal\nSearch for files.\n\n"
    "## Requirements\n- Use find.\n\n"
)
STATE_SECTION = (
    "Current terminal state:\n"
    "Current Terminal Screen:\n"
    "root@host:/workspace#\n\n\n"
)

MSG0 = {"role": "user", "content": SYSTEM_SECTION + TASK_SECTION + STATE_SECTION}

ASSISTANT_1 = {
    "role": "assistant",
    "content": (
        '{"analysis": "I need to find files.", '
        '"plan": "Run find command.", '
        '"commands": [{"keystrokes": "find . -name \'*.txt\'\\n", "duration": 1.0}], '
        '"task_complete": false}'
    ),
}

USER_1 = {
    "role": "user",
    "content": (
        "Current terminal state:\n"
        "New Terminal Output:\n"
        "root@host:/workspace# find . -name '*.txt'\n"
        "./a.txt\n"
        "root@host:/workspace#\n\n\n"
    ),
}

ASSISTANT_2 = {
    "role": "assistant",
    "content": (
        '{"analysis": "Found a.txt.", '
        '"plan": "Submit.", '
        '"commands": [{"keystrokes": "cat a.txt\\n", "duration": 0.1}], '
        '"task_complete": true}'
    ),
}


# ── Test: complete valid trace ────────────────────────────────────────

class TestConvertValidTrace:
    def test_basic_structure(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row, source_label="test/source")

        assert result["_conversion_ok"] is True
        assert result["source"] == "test/source"
        msgs = result["messages"]

        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "tool"

    def test_system_prompt_replaced(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        sys_msg = result["messages"][0]
        assert "Format your response as JSON" not in sys_msg["content"]
        assert "bash" in sys_msg["content"]

    def test_task_description_preserved(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        user_msg = result["messages"][1]
        assert "Terminal Automation Request" in user_msg["content"]
        assert "Search for files" in user_msg["content"]

    def test_reasoning_content_present(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        asst = result["messages"][2]
        assert "reasoning_content" in asst
        assert "I need to find files" in asst["reasoning_content"]

    def test_tool_call_structure(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        asst = result["messages"][2]
        assert "tool_calls" in asst
        tc = asst["tool_calls"][0]
        assert tc["function"]["name"] == "bash"
        assert isinstance(tc["function"]["arguments"], dict)
        assert "find . -name '*.txt'" in tc["function"]["arguments"]["command"]
        assert tc["id"].startswith("call_")

    def test_tool_result_structure(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        tool_msg = result["messages"][3]
        assert tool_msg["role"] == "tool"
        assert isinstance(tool_msg["content"], str)
        assert "./a.txt" in tool_msg["content"]
        assert tool_msg["tool_call_ids"][0] == result["messages"][2]["tool_calls"][0]["id"]

    def test_submit_present_at_end(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        msgs = result["messages"]
        submit_msgs = [m for m in msgs if m.get("tool_calls") and
                       "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in
                       m["tool_calls"][0]["function"]["arguments"].get("command", "")]
        assert len(submit_msgs) == 1

    def test_metadata(self):
        row = _make_trace(
            [MSG0, ASSISTANT_1, USER_1, ASSISTANT_2],
            model="test-model",
            task="task_0",
            episode="episode-1",
        )
        result = convert_trace(row, source_label="test/src")
        meta = result["metadata"]
        assert meta["source_model"] == "test-model"
        assert meta["task"] == "task_0"
        assert meta["has_task_complete"] is True
        assert meta["num_turns"] == 2

    def test_source_label_stamped(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row, source_label="nvidia/Nemotron/easy")
        assert result["source"] == "nvidia/Nemotron/easy"


# ── Test: edge cases ─────────────────────────────────────────────────

class TestConvertEdgeCases:
    def test_empty_conversations(self):
        row = _make_trace([])
        result = convert_trace(row)
        assert result["_conversion_ok"] is False

    def test_not_starting_with_user(self):
        row = _make_trace([{"role": "assistant", "content": "hi"}])
        result = convert_trace(row)
        assert result["_conversion_ok"] is False

    def test_missing_task_delim(self):
        msg0 = {"role": "user", "content": "Just some task.\nCurrent terminal state:\nroot@x:#\n"}
        row = _make_trace([
            msg0,
            ASSISTANT_2,
        ])
        result = convert_trace(row)
        assert any("TASK_DELIM" in w for w in result["warnings"])

    def test_malformed_json_fallback(self):
        bad_assistant = {
            "role": "assistant",
            "content": "I will just type some stuff with no JSON at all.",
        }
        row = _make_trace([MSG0, bad_assistant])
        result = convert_trace(row)
        assert any("JSON extraction failed" in w for w in result["warnings"])

    def test_special_keystrokes_preserved(self):
        asst = {
            "role": "assistant",
            "content": (
                '{"analysis": "Interrupt.", "plan": "Send C-c.", '
                '"commands": [{"keystrokes": "C-c", "duration": 0.5}], '
                '"task_complete": false}'
            ),
        }
        user = {
            "role": "user",
            "content": "Current terminal state:\nNew Terminal Output:\n^C\nroot@h:/w#\n",
        }
        row = _make_trace([MSG0, asst, user])
        result = convert_trace(row)
        tc = result["messages"][2]["tool_calls"][0]
        assert tc["function"]["arguments"]["command"] == "C-c"

    def test_confirmation_turn_consumed(self):
        """When the harness asks 'Are you sure?', the confirmation text should
        be stripped from the tool result content."""
        confirm_user = {
            "role": "user",
            "content": (
                "Current terminal state:\n"
                "New Terminal Output:\n"
                "root@host:#\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger your solution to be graded.\n"
            ),
        }
        final_asst = {
            "role": "assistant",
            "content": '{"analysis": "Yes.", "plan": "Confirm.", "commands": [], "task_complete": true}',
        }
        row = _make_trace([MSG0, ASSISTANT_1, confirm_user, final_asst])
        result = convert_trace(row)
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        for tm in tool_msgs:
            assert "Are you sure" not in tm["content"]


# ── Test: structural integrity ────────────────────────────────────────

class TestStructuralIntegrity:
    def test_every_tool_call_has_matching_result(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        msgs = result["messages"]

        call_ids = set()
        for m in msgs:
            for tc in m.get("tool_calls", []):
                if tc["function"]["name"] == "bash":
                    call_ids.add(tc["id"])

        result_ids = set()
        for m in msgs:
            if m["role"] == "tool":
                result_ids.update(m["tool_call_ids"])

        # Every bash tool_call with a following user message should have a result
        assert result_ids.issubset(call_ids)

    def test_no_duplicate_tool_call_ids(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        msgs = result["messages"]
        ids = [tc["id"] for m in msgs for tc in m.get("tool_calls", [])]
        assert len(ids) == len(set(ids))

    def test_role_order(self):
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row)
        msgs = result["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        for m in msgs[2:]:
            assert m["role"] in ("assistant", "tool")


# ── Test: premature submit echo handling ─────────────────────────────

class TestDuplicateSubmitIntegration:
    """End-to-end test through convert_trace with a trace that has a
    duplicate submit flow (model echoes submit with task_complete: false,
    harness confirms, model re-submits with task_complete: true)."""

    def test_duplicate_submit_cleaned_in_conversion(self):
        first_submit = {
            "role": "assistant",
            "content": (
                '{"analysis": "Done, submitting.", "plan": "Submit.", '
                '"commands": [{"keystrokes": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n"}], '
                '"task_complete": false}'
            ),
        }
        confirmation = {
            "role": "user",
            "content": (
                "Current terminal state:\n"
                "New Terminal Output:\n"
                "root@host:/workspace#\n\n"
                "Are you sure you want to mark the task as complete? "
                "This will trigger grading.\n"
            ),
        }
        final_submit = {
            "role": "assistant",
            "content": (
                '{"analysis": "Yes.", "plan": "Confirm.", '
                '"commands": [], "task_complete": true}'
            ),
        }
        row = _make_trace([
            MSG0, ASSISTANT_1, USER_1,
            first_submit, confirmation, final_submit,
        ])
        result = convert_trace(row, source_label="test/dup")

        assert result["_conversion_ok"] is True
        msgs = result["messages"]

        submit_echos = [
            m for m in msgs
            if m.get("tool_calls") and
            SUBMIT_COMMAND in m["tool_calls"][0]["function"]["arguments"].get("command", "")
        ]
        assert len(submit_echos) == 1, (
            f"Expected exactly 1 submit echo, got {len(submit_echos)}"
        )

        assert any("premature submit" in w.lower() for w in result["warnings"])

    def test_premature_submit_reasoning_preserved(self):
        """Reasoning from the premature submit turn should be merged into the
        final submit message rather than discarded."""
        premature = {
            "role": "assistant",
            "content": (
                '{"analysis": "I believe the task is done.", "plan": "Submit now.", '
                '"commands": [{"keystrokes": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n"}], '
                '"task_complete": false}'
            ),
        }
        confirmation = {
            "role": "user",
            "content": (
                "Current terminal state:\n"
                "New Terminal Output:\n"
                "root@host:/workspace#\n\n"
                "Are you sure you want to mark the task as complete?\n"
            ),
        }
        final_submit = {
            "role": "assistant",
            "content": (
                '{"analysis": "Confirmed.", "plan": "Done.", '
                '"commands": [], "task_complete": true}'
            ),
        }
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, premature, confirmation, final_submit])
        result = convert_trace(row)
        msgs = result["messages"]
        submit = msgs[-1]
        assert "reasoning_content" in submit
        assert "I believe the task is done" in submit["reasoning_content"]
        assert "Confirmed" in submit["reasoning_content"]

    def test_normal_trace_unaffected(self):
        """A trace with a single clean submit (task_complete: true) should be
        completely unaffected by the premature-submit logic."""
        row = _make_trace([MSG0, ASSISTANT_1, USER_1, ASSISTANT_2])
        result = convert_trace(row, source_label="test/normal")

        assert result["_conversion_ok"] is True
        assert not any("premature" in w.lower() for w in result["warnings"])
        msgs = result["messages"]
        submit_echos = [
            m for m in msgs
            if m.get("tool_calls") and
            SUBMIT_COMMAND in m["tool_calls"][0]["function"]["arguments"].get("command", "")
        ]
        assert len(submit_echos) == 1
