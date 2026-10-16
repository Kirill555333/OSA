"""Tests for AgentMaxRoundsError and no-rollback behavior (1.5.1.5.1)."""

from __future__ import annotations

import pytest

from osa.core import Agent, AgentError, AgentMaxRoundsError
from osa.models import (
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry


class EndlessToolModel(ModelInterface):
    """Fake model that always requests one tool and never completes."""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "endless-tool"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        return ModelResponse(
            content="",
            model_name=self.model_name,
            finish_reason="tool_calls",
            tool_calls=(
                ToolCall(
                    id=f"call_{len(self.calls)}",
                    name="calculator",
                    arguments={"expression": "1 + 1"},
                ),
            ),
        )

    def health_check(self) -> bool:
        return True


class FailingModel(ModelInterface):
    """Fake model that fails with ModelError on every call."""

    @property
    def model_name(self) -> str:
        return "failing"

    def generate(self, request: ModelRequest) -> ModelResponse:
        raise ModelError("simulated failure")

    def health_check(self) -> bool:
        return True


def _build_agent(
    model: ModelInterface,
    *,
    max_tool_rounds: int = 3,
) -> Agent:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    return Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {"calculator": PermissionLevel.ALLOW}
        ),
        max_tool_rounds=max_tool_rounds,
    )


def test_max_rounds_error_is_agent_error_subclass() -> None:
    assert issubclass(AgentMaxRoundsError, AgentError)


def test_max_rounds_error_exposes_rounds() -> None:
    exc = AgentMaxRoundsError(5)
    assert exc.rounds == 5
    assert "5" in str(exc)


def test_chat_returns_response_on_max_rounds() -> None:
    model = EndlessToolModel()
    agent = _build_agent(model, max_tool_rounds=3)

    response = agent.chat("Посчитай что-нибудь.")

    assert response.finish_reason == "max_rounds_exceeded"
    assert "лимит раундов" in response.content
    assert "3" in response.content
    assert len(model.calls) == 3


def test_chat_does_not_roll_back_on_max_rounds() -> None:
    model = EndlessToolModel()
    agent = _build_agent(model, max_tool_rounds=2)

    agent.chat("Посчитай что-нибудь.")

    messages = agent.context.messages()
    user_messages = [m for m in messages if m.role == "user"]
    tool_messages = [m for m in messages if m.role == "tool"]

    assert len(user_messages) == 1
    assert user_messages[0].content == "Посчитай что-нибудь."
    assert len(tool_messages) == 2


def test_chat_still_rolls_back_on_other_agent_errors() -> None:
    model = FailingModel()
    agent = Agent(model=model)

    with pytest.raises(AgentError) as excinfo:
        agent.chat("Hello")

    assert not isinstance(excinfo.value, AgentMaxRoundsError)
    assert len(agent.context) == 0


def test_chat_stream_yields_warning_on_max_rounds() -> None:
    model = EndlessToolModel()
    agent = _build_agent(model, max_tool_rounds=2)

    chunks = list(agent.chat_stream("Посчитай."))

    assert len(chunks) == 1
    assert "лимит раундов" in chunks[0]

    tool_messages = [
        m for m in agent.context.messages() if m.role == "tool"
    ]
    assert len(tool_messages) == 2
