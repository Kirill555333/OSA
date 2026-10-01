"""Unit tests for ContextManager (0.8.11.6)."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from osa.context import (
    BudgetedMemoryResult,
    BudgetedMemoryRetriever,
    ContextBudget,
    ContextManager,
    ContextManagerError,
    ConversationSummary,
)
from osa.memory.long_term import MemoryRecord
from osa.memory.retrieval import MemorySearchResult
from osa.models import ChatMessage, ToolDefinition


def test_context_manager_defaults_and_short_dialogue() -> None:
    manager = ContextManager(default_system_prompt="You are OSA.")
    assert manager.current_summary.is_empty

    messages = [
        ChatMessage(role="user", content="Hello OSA"),
    ]

    assembled = manager.assemble(messages)
    assert len(assembled.messages) == 2
    assert assembled.messages[0].role == "system"
    assert assembled.messages[0].content == "You are OSA."
    assert assembled.messages[1].role == "user"
    assert assembled.messages[1].content == "Hello OSA"
    assert assembled.evicted_messages_count == 0
    assert assembled.summary.is_empty


def test_context_manager_long_dialogue_triggers_compaction_and_summary() -> None:
    # Tight history budget of 80 tokens to trigger sliding window eviction
    budget = ContextBudget.create(
        total_context_limit=1000,
        reserved_output_tokens=100,
        history_budget=80,
    )
    manager = ContextManager(budget=budget, default_system_prompt="System prompt")

    messages = [
        ChatMessage(role="user", content="Goal: Build the autonomous space probe"),
        ChatMessage(role="assistant", content="Decided: Use Python for onboard control"),
        ChatMessage(role="user", content="Message 2: Discuss telemetry protocols"),
        ChatMessage(role="assistant", content="Reply 2: Telemetry protocol is MQTT"),
        ChatMessage(role="user", content="Message 3: What about battery life?"),
        ChatMessage(role="assistant", content="Reply 3: Battery life is 10 days"),
        ChatMessage(role="user", content="Current active question: What sensors are ready?"),
    ]

    assembled = manager.assemble(messages)

    assert assembled.evicted_messages_count > 0
    assert not assembled.summary.is_empty
    # Summary block is injected into the messages
    roles = [m.role for m in assembled.messages]
    assert roles.count("system") >= 2  # base system + summary block
    summary_msg = assembled.messages[1]
    assert "[Prior Conversation Summary]" in summary_msg.content


def test_context_manager_with_budgeted_memory_retrieval() -> None:
    mock_retriever = MagicMock(spec=BudgetedMemoryRetriever)
    record = MemoryRecord(
        id=1,
        content="User lives in Berlin",
        category="user",
        importance=8,
        created_at="2026-10-01T00:00:00Z",
        updated_at="2026-10-01T00:00:00Z",
    )
    mock_retriever.retrieve.return_value = BudgetedMemoryResult(
        memories=(MemorySearchResult(memory=record, score=3.0),),
        prompt="[Memory #1] (user): User lives in Berlin",
        estimated_tokens=15,
    )

    manager = ContextManager(
        memory_retriever=mock_retriever,
        default_system_prompt="Base sys",
    )

    messages = [ChatMessage(role="user", content="Where do I live?")]
    assembled = manager.assemble(messages, query="Where do I live?")

    assert assembled.memory_result is not None
    assert len(assembled.memory_result.memories) == 1
    # Check that memory context is in assembled messages
    contents = [m.content for m in assembled.messages]
    assert any("User lives in Berlin" in c for c in contents)


def test_context_manager_with_task_context_and_tools() -> None:
    manager = ContextManager(default_system_prompt="Sys")
    tool = ToolDefinition(
        name="test_tool",
        description="A tool description",
        parameters={"type": "object"},
    )

    messages = [ChatMessage(role="user", content="Work on task")]
    assembled = manager.assemble(
        messages,
        task_context="Step 3 of 5: downloading data",
        tools=(tool,),
    )

    contents = [m.content for m in assembled.messages]
    assert any("Current Task State" in c and "Step 3 of 5" in c for c in contents)
    assert assembled.estimated_input_tokens > 0


def test_context_manager_override_memory_context() -> None:
    manager = ContextManager(default_system_prompt="Sys")
    messages = [ChatMessage(role="user", content="Hello")]

    assembled = manager.assemble(
        messages,
        override_memory_context="Custom injected memory prompt line",
    )

    contents = [m.content for m in assembled.messages]
    assert any("Custom injected memory prompt line" in c for c in contents)


def test_context_manager_reset_and_set_summary() -> None:
    manager = ContextManager()
    custom_summary = ConversationSummary(goal="Custom Goal", current_state="Done")

    manager.set_summary(custom_summary)
    assert manager.current_summary.goal == "Custom Goal"

    manager.reset()
    assert manager.current_summary.is_empty


def test_context_manager_hard_overflow_fails_closed() -> None:
    # Extremely small total context limit that cannot fit system prompt + messages
    tiny_budget = ContextBudget(
        total_context_limit=30,
        reserved_output_tokens=10,
        system_budget=5,
        memory_budget=0,
        summary_budget=0,
        task_budget=0,
        history_budget=15,
        tool_budget=0,
    )

    manager = ContextManager(
        budget=tiny_budget,
        default_system_prompt="A system prompt that is much larger than twenty tokens to trigger budget overflow.",
    )

    messages = [ChatMessage(role="user", content="A message that also has words.")]

    with pytest.raises(ContextManagerError, match="Assembled context exceeds budget"):
        manager.assemble(messages)
