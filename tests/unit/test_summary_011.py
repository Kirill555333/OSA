"""Unit tests for ConversationSummary and SummaryBuilder (0.8.11.3)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from osa.context import (
    ConversationSummary,
    SummaryBuilder,
)
from osa.models import ChatMessage, ModelError, ModelResponse


def test_conversation_summary_empty_and_properties() -> None:
    empty = ConversationSummary.empty()
    assert empty.is_empty
    assert empty.format_for_prompt() == ""
    assert empty.messages_summarized_count == 0


def test_conversation_summary_format_for_prompt() -> None:
    summary = ConversationSummary(
        goal="Develop OSA context manager",
        current_state="0.8.11.3 in progress",
        decisions=("Use structured summary", "Strict budget enforcement"),
        facts=("Model limit is 13568",),
        constraints=("Do not break ActionRequest",),
        unresolved=("Implement compaction algorithm",),
        messages_summarized_count=20,
    )

    assert not summary.is_empty
    formatted = summary.format_for_prompt()

    assert "[Prior Conversation Summary]" in formatted
    assert "Goal: Develop OSA context manager" in formatted
    assert "Current State: 0.8.11.3 in progress" in formatted
    assert "- Use structured summary" in formatted
    assert "- Model limit is 13568" in formatted
    assert "- Do not break ActionRequest" in formatted
    assert "- Implement compaction algorithm" in formatted


def test_conversation_summary_serialization_round_trip() -> None:
    summary = ConversationSummary(
        goal="Goal",
        current_state="State",
        decisions=("Dec 1",),
        facts=("Fact 1",),
        constraints=("Constraint 1",),
        unresolved=("Task 1",),
        messages_summarized_count=5,
        updated_at="2026-10-01T12:00:00Z",
    )

    data = summary.to_dict()
    restored = ConversationSummary.from_dict(data)

    assert restored == summary
    assert restored.decisions == ("Dec 1",)
    assert restored.messages_summarized_count == 5


def test_summary_builder_fallback_deterministic() -> None:
    builder = SummaryBuilder(model=None)

    messages = [
        ChatMessage(role="user", content="Build the context manager for OSA."),
        ChatMessage(role="assistant", content="Plan agreed and step 1 started."),
        ChatMessage(role="user", content="Constraint: только не ломай backward compatibility?"),
    ]

    summary = builder.build_summary(messages)

    assert not summary.is_empty
    assert summary.goal == "Build the context manager for OSA."
    assert summary.messages_summarized_count == 3
    assert len(summary.constraints) >= 1
    assert len(summary.unresolved) >= 1


def test_summary_builder_with_mock_model_json() -> None:
    mock_model = MagicMock()
    json_payload = {
        "goal": "Refactor context subsystem",
        "current_state": "Completed step 2",
        "decisions": ["Adopt Session model"],
        "facts": ["Using local Qwen inference"],
        "constraints": ["Keep zero external dependencies"],
        "unresolved": ["Implement compactor"],
    }
    mock_model.generate.return_value = ModelResponse(
        content=f"```json\n{json.dumps(json_payload)}\n```",
        model_name="mock-llm",
    )

    builder = SummaryBuilder(model=mock_model)
    messages = [
        ChatMessage(role="user", content="Step 2 done"),
        ChatMessage(role="assistant", content="Confirmed"),
    ]

    summary = builder.build_summary(messages)

    assert summary.goal == "Refactor context subsystem"
    assert summary.current_state == "Completed step 2"
    assert "Adopt Session model" in summary.decisions
    assert "Using local Qwen inference" in summary.facts
    assert summary.messages_summarized_count == 2


def test_summary_builder_model_failure_triggers_fallback() -> None:
    mock_model = MagicMock()
    mock_model.generate.side_effect = ModelError("Backend unreachable")

    builder = SummaryBuilder(model=mock_model)
    messages = [
        ChatMessage(role="user", content="Perform system diagnostic."),
        ChatMessage(role="assistant", content="Diagnostic completed."),
    ]

    # Must not raise ModelError; must fall back to deterministic extraction
    summary = builder.build_summary(messages)

    assert not summary.is_empty
    assert summary.goal == "Perform system diagnostic."
    assert summary.messages_summarized_count == 2


def test_summary_builder_cumulative_count() -> None:
    builder = SummaryBuilder(model=None)
    prev = ConversationSummary(
        goal="Initial goal",
        messages_summarized_count=10,
    )

    new_messages = [
        ChatMessage(role="user", content="Next task"),
        ChatMessage(role="assistant", content="Done"),
    ]

    updated = builder.build_summary(new_messages, previous_summary=prev)
    assert updated.messages_summarized_count == 12
