"""Unit tests for ContextCompactor (0.8.11.5)."""

from __future__ import annotations

import pytest

from osa.context import (
    ConservativeTokenEstimator,
    ContextCompactor,
)
from osa.models import ChatMessage, ToolCall


def test_context_compactor_validation() -> None:
    with pytest.raises(ValueError, match="min_recent_messages must be at least 1"):
        ContextCompactor(min_recent_messages=0)

    with pytest.raises(ValueError, match="max_tool_content_tokens must be at least 10"):
        ContextCompactor(max_tool_content_tokens=5)

    compactor = ContextCompactor()
    with pytest.raises(ValueError, match="budget must be greater than zero"):
        compactor.compact([], budget=0)


def test_context_compactor_empty_and_system_only() -> None:
    compactor = ContextCompactor()

    res_empty = compactor.compact((), budget=100)
    assert res_empty.retained_messages == ()
    assert res_empty.evicted_messages == ()
    assert res_empty.estimated_tokens == 0

    sys_msg = ChatMessage(role="system", content="You are OSA.")
    res_sys = compactor.compact((sys_msg,), budget=100)
    assert res_sys.retained_messages == (sys_msg,)
    assert res_sys.evicted_messages == ()
    assert res_sys.estimated_tokens > 0


def test_context_compactor_within_budget_no_eviction() -> None:
    compactor = ContextCompactor()

    messages = (
        ChatMessage(role="system", content="System"),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Hi there!"),
    )

    result = compactor.compact(messages, budget=1000)
    assert result.retained_messages == messages
    assert result.evicted_messages == ()
    assert result.compacted_tool_count == 0


def test_context_compactor_truncates_older_tool_outputs() -> None:
    compactor = ContextCompactor(
        max_tool_content_tokens=20,
        preserve_recent_tool_rounds=1,
    )

    huge_output = "Line of output data from directory listing.\n" * 50

    call1 = ToolCall(id="call_1", name="list_dir", arguments={"path": "."})
    call2 = ToolCall(id="call_2", name="read_file", arguments={"path": "a.txt"})

    messages = (
        ChatMessage(role="system", content="System prompt"),
        ChatMessage(role="user", content="List directory"),
        ChatMessage(role="assistant", content="Running", tool_calls=(call1,)),
        ChatMessage(role="tool", content=huge_output, tool_call_id="call_1"),
        ChatMessage(role="assistant", content="Done listing"),
        ChatMessage(role="user", content="Read file"),
        ChatMessage(role="assistant", content="Reading", tool_calls=(call2,)),
        ChatMessage(role="tool", content=huge_output, tool_call_id="call_2"),
        ChatMessage(role="assistant", content="Done reading"),
    )

    # Budget is sufficient to keep all messages if the older tool output is compacted
    result = compactor.compact(messages, budget=3000)

    assert result.compacted_tool_count == 1
    # Round 1 tool output was compacted
    tool1_msg = result.retained_messages[3]
    assert tool1_msg.role == "tool"
    assert "...[output truncated for context budget]" in tool1_msg.content

    # Round 2 (recent) tool output was protected
    tool2_msg = result.retained_messages[7]
    assert tool2_msg.role == "tool"
    assert "...[output truncated for context budget]" not in tool2_msg.content


def test_context_compactor_atomic_unit_preservation() -> None:
    """Tool calls and their responses must never be separated during sliding window eviction."""
    compactor = ContextCompactor(
        min_recent_messages=2,
        preserve_recent_tool_rounds=0,
    )

    call1 = ToolCall(id="c1", name="tool1", arguments={})
    call2 = ToolCall(id="c2", name="tool2", arguments={})

    messages = (
        ChatMessage(role="system", content="Sys"),
        ChatMessage(role="user", content="Task 1"),
        ChatMessage(role="assistant", content="Calling 1", tool_calls=(call1,)),
        ChatMessage(role="tool", content="Result 1", tool_call_id="c1"),
        ChatMessage(role="assistant", content="Finished 1"),
        ChatMessage(role="user", content="Task 2"),
        ChatMessage(role="assistant", content="Calling 2", tool_calls=(call2,)),
        ChatMessage(role="tool", content="Result 2", tool_call_id="c2"),
        ChatMessage(role="assistant", content="Finished 2"),
    )

    estimator = ConservativeTokenEstimator()
    sys_tokens = estimator.estimate_message(messages[0])
    recent_task_tokens = estimator.estimate_messages(messages[5:])

    # Budget fits System + Task 2 turn, but not Task 1 turn
    budget = sys_tokens + recent_task_tokens + 10

    result = compactor.compact(messages, budget=budget)

    # System is retained at index 0
    assert result.retained_messages[0].role == "system"

    # Evicted messages must contain the entirety of Task 1 (user, assistant+call, tool, assistant response)
    evicted_roles = [m.role for m in result.evicted_messages]
    assert evicted_roles == ["user", "assistant", "tool", "assistant"]

    # In retained messages, the tool call and tool message of Task 2 are intact together
    retained_roles = [m.role for m in result.retained_messages]
    assert retained_roles == ["system", "user", "assistant", "tool", "assistant"]
    assert result.retained_messages[2].tool_calls[0].id == "c2"
    assert result.retained_messages[3].tool_call_id == "c2"


def test_context_compactor_slides_window_order() -> None:
    compactor = ContextCompactor(min_recent_messages=2)

    messages = (
        ChatMessage(role="system", content="System"),
        ChatMessage(role="user", content="Message 1"),
        ChatMessage(role="assistant", content="Reply 1"),
        ChatMessage(role="user", content="Message 2"),
        ChatMessage(role="assistant", content="Reply 2"),
        ChatMessage(role="user", content="Message 3"),
        ChatMessage(role="assistant", content="Reply 3"),
    )

    estimator = ConservativeTokenEstimator()
    tokens_per_pair = estimator.estimate_messages(messages[1:3])
    sys_tokens = estimator.estimate_message(messages[0])

    # Allow System + 2 pairs (4 messages)
    budget = sys_tokens + (tokens_per_pair * 2) + 5

    result = compactor.compact(messages, budget=budget)

    assert len(result.evicted_messages) == 2
    assert result.evicted_messages[0].content == "Message 1"
    assert result.evicted_messages[1].content == "Reply 1"

    assert len(result.retained_messages) == 5
    assert result.retained_messages[0].role == "system"
    assert result.retained_messages[1].content == "Message 2"
    assert result.retained_messages[-1].content == "Reply 3"
