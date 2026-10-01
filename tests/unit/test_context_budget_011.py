"""Unit tests for ContextBudget and ConservativeTokenEstimator (0.8.11.1)."""

from __future__ import annotations

import pytest

from osa.context import (
    ConservativeTokenEstimator,
    ContextBudget,
    ContextBudgetError,
    ContextOverflowError,
)
from osa.models import ChatMessage, ToolCall, ToolDefinition


def test_context_budget_initialization_and_properties() -> None:
    budget = ContextBudget(
        total_context_limit=1000,
        reserved_output_tokens=200,
        system_budget=100,
        memory_budget=100,
        summary_budget=100,
        task_budget=100,
        history_budget=300,
        tool_budget=100,
    )

    assert budget.total_context_limit == 1000
    assert budget.reserved_output_tokens == 200
    assert budget.max_input_tokens == 800
    assert budget.allocated_input_tokens == 800
    assert budget.unallocated_headroom == 0


def test_context_budget_allows_headroom() -> None:
    budget = ContextBudget(
        total_context_limit=1000,
        reserved_output_tokens=200,
        system_budget=100,
        memory_budget=50,
        summary_budget=50,
        task_budget=50,
        history_budget=200,
        tool_budget=50,
    )

    assert budget.max_input_tokens == 800
    assert budget.allocated_input_tokens == 500
    assert budget.unallocated_headroom == 300


def test_context_budget_rejects_oversubscription() -> None:
    with pytest.raises(ContextBudgetError, match="exceeds max_input_tokens"):
        ContextBudget(
            total_context_limit=1000,
            reserved_output_tokens=200,
            system_budget=300,
            memory_budget=200,
            summary_budget=100,
            task_budget=100,
            history_budget=200,
            tool_budget=100,
        )


def test_context_budget_rejects_invalid_limits() -> None:
    with pytest.raises(ContextBudgetError, match="total_context_limit must be greater than zero"):
        ContextBudget(
            total_context_limit=0,
            reserved_output_tokens=100,
            system_budget=10,
            memory_budget=10,
            summary_budget=10,
            task_budget=10,
            history_budget=10,
            tool_budget=10,
        )

    with pytest.raises(ContextBudgetError, match="reserved_output_tokens must be strictly less"):
        ContextBudget(
            total_context_limit=100,
            reserved_output_tokens=100,
            system_budget=10,
            memory_budget=10,
            summary_budget=10,
            task_budget=10,
            history_budget=10,
            tool_budget=10,
        )

    with pytest.raises(ContextBudgetError, match="cannot be negative"):
        ContextBudget(
            total_context_limit=1000,
            reserved_output_tokens=100,
            system_budget=-10,
            memory_budget=10,
            summary_budget=10,
            task_budget=10,
            history_budget=10,
            tool_budget=10,
        )


def test_context_budget_factory_defaults() -> None:
    budget = ContextBudget.create(
        total_context_limit=13568,
        reserved_output_tokens=512,
    )

    assert budget.total_context_limit == 13568
    assert budget.reserved_output_tokens == 512
    assert budget.max_input_tokens == 13056
    assert budget.allocated_input_tokens <= budget.max_input_tokens
    assert budget.history_budget > 0
    assert budget.system_budget > 0
    assert budget.tool_budget > 0


def test_context_budget_factory_custom_allocations() -> None:
    budget = ContextBudget.create(
        total_context_limit=4000,
        reserved_output_tokens=500,
        system_budget=300,
        tool_budget=200,
    )

    assert budget.system_budget == 300
    assert budget.tool_budget == 200
    assert budget.max_input_tokens == 3500
    assert budget.allocated_input_tokens <= 3500


def test_validate_input_tokens_success_and_overflow() -> None:
    budget = ContextBudget.create(
        total_context_limit=1000,
        reserved_output_tokens=200,
    )

    # Within budget: 800 tokens max input
    budget.validate_input_tokens(500)
    budget.validate_input_tokens(800)

    # Exceeding budget:
    with pytest.raises(ContextOverflowError, match="exceed max_input_tokens"):
        budget.validate_input_tokens(801)


def test_conservative_token_estimator_text() -> None:
    estimator = ConservativeTokenEstimator()

    assert estimator.estimate_text("") == 0
    assert estimator.estimate_text("Hello world") >= 2

    # Russian text estimation
    ru_text = "Привет, как твои дела? Запомни эту важную инструкцию."
    ru_tokens = estimator.estimate_text(ru_text)
    assert ru_tokens > 5
    # Must be conservative (at least ~1 token per 2 chars floor)
    assert ru_tokens >= len(ru_text) // 3


def test_conservative_token_estimator_messages() -> None:
    estimator = ConservativeTokenEstimator()

    message = ChatMessage(
        role="assistant",
        content="Here is a planned execution step.",
        tool_calls=(
            ToolCall(
                id="call_123",
                name="browser_open",
                arguments={"url": "https://example.com"},
            ),
        ),
    )

    estimated = estimator.estimate_message(message)
    assert estimated > 15

    messages = (
        ChatMessage(role="system", content="You are OSA."),
        ChatMessage(role="user", content="Open browser."),
        message,
    )

    total_est = estimator.estimate_messages(messages)
    assert total_est > estimated


def test_conservative_token_estimator_tools() -> None:
    estimator = ConservativeTokenEstimator()

    tool = ToolDefinition(
        name="calculator",
        description="Perform basic math computations.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )

    cost = estimator.estimate_tool_definition(tool)
    assert cost > 10

    tools_cost = estimator.estimate_tools((tool, tool))
    assert tools_cost == cost * 2
