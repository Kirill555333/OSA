"""Unit tests for Agent Core and ContextManager integration (0.8.11.7)."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from osa.context import ContextBudget, ContextManager
from osa.core import Agent
from osa.memory.integration import MemoryIntegration, MemoryIntegrationResult
from osa.memory.long_term import MemoryRecord
from osa.memory.retrieval import MemorySearchResult
from osa.models import ChatMessage, ModelInterface, ModelRequest, ModelResponse


def _create_mock_model(reply: str = "Assistant reply") -> MagicMock:
    mock_model = MagicMock(spec=ModelInterface)
    mock_model.model_name = "mock-model"
    mock_model.generate.return_value = ModelResponse(
        content=reply,
        model_name="mock-model",
    )
    return mock_model


def test_agent_initialization_with_context_manager() -> None:
    model = _create_mock_model()
    agent = Agent(model=model, system_prompt="System prompt")

    assert agent.context_manager is not None
    assert agent.context_manager.budget is not None
    assert agent.context_manager.current_summary.is_empty


def test_agent_custom_context_budget_injection() -> None:
    model = _create_mock_model()
    custom_budget = ContextBudget.create(
        total_context_limit=4096,
        reserved_output_tokens=256,
    )

    agent = Agent(
        model=model,
        system_prompt="System prompt",
        context_budget=custom_budget,
    )

    assert agent.context_manager.budget.total_context_limit == 4096
    assert agent.context_manager.budget.reserved_output_tokens == 256


def test_agent_chat_invokes_context_manager_and_model() -> None:
    model = _create_mock_model(reply="Understood.")
    agent = Agent(model=model, system_prompt="You are OSA.")

    response = agent.chat("Hello there!")

    assert response.content == "Understood."
    # Context preserves full messages
    assert len(agent.context.messages()) == 3
    assert agent.context.messages()[0].role == "system"
    assert agent.context.messages()[1].role == "user"
    assert agent.context.messages()[2].role == "assistant"

    # Verify model received messages assembled via context manager
    last_call_req: ModelRequest = model.generate.call_args[0][0]
    sent_roles = [m.role for m in last_call_req.messages]
    assert sent_roles == ["system", "user"]


def test_agent_long_conversation_summary_compaction() -> None:
    model = _create_mock_model(reply="OK")
    # Set a tight history budget (80 tokens) so older messages get compacted
    tight_budget = ContextBudget.create(
        total_context_limit=1000,
        reserved_output_tokens=100,
        history_budget=80,
    )

    agent = Agent(
        model=model,
        system_prompt="Base system prompt",
        context_budget=tight_budget,
    )

    # Simulate accumulating dialogue turns
    for i in range(1, 8):
        agent.chat(f"User message number {i}")

    # Full session archive stores all turns
    assert len(agent.context.messages()) >= 15

    # ContextManager should have generated a summary for evicted turns
    assert not agent.context_manager.current_summary.is_empty

    # Model request must include the summary and not exceed budget
    last_call_req: ModelRequest = model.generate.call_args[0][0]
    sent_contents = [m.content for m in last_call_req.messages]
    assert any("[Prior Conversation Summary]" in c for c in sent_contents)


def test_agent_reset_clears_context_and_summary_preserving_long_term_memory() -> None:
    model = _create_mock_model()
    agent = Agent(model=model, system_prompt="System prompt")

    agent.chat("Remember this task")
    assert len(agent.context.messages()) == 3

    # Manually populate summary on manager to verify reset
    agent.context_manager.current_summary.to_dict()

    agent.reset()

    # Raw context preserves system prompt
    assert len(agent.context.messages()) == 1
    assert agent.context.messages()[0].role == "system"
    # Context manager summary is cleared
    assert agent.context_manager.current_summary.is_empty


def test_agent_memory_integration_retrieval_passed_to_model() -> None:
    model = _create_mock_model()
    mock_memory_integration = MagicMock(spec=MemoryIntegration)
    record = MemoryRecord(
        id=42,
        content="User loves terminal workflows",
        category="preference",
        importance=9,
        created_at="2026-10-01T00:00:00Z",
        updated_at="2026-10-01T00:00:00Z",
    )
    mock_memory_integration.retrieve.return_value = MemoryIntegrationResult(
        memories=(MemorySearchResult(memory=record, score=5.0),),
        prompt="[Memory #42]: User loves terminal workflows",
    )

    agent = Agent(
        model=model,
        system_prompt="System prompt",
        memory_integration=mock_memory_integration,
    )

    agent.chat("What do I like?")

    last_call_req: ModelRequest = model.generate.call_args[0][0]
    sent_contents = [m.content for m in last_call_req.messages]
    assert any("User loves terminal workflows" in c for c in sent_contents)
