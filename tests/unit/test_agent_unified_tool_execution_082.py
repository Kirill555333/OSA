from __future__ import annotations

import pytest

from osa.core.agent import Agent, PermissionDeniedError
from osa.core.modes import AgentMode
from osa.models import ModelResponse
from osa.permissions import (
    ConfirmationHandler,
    PermissionLevel,
    PermissionPolicy,
)
from osa.tools import ToolRegistry, ToolResult


class FakeModel:
    def generate(self, request):
        return ModelResponse(
            content="ok",
            model_name="fake",
        )

    def generate_stream_events(self, request):
        raise NotImplementedError


class SuccessTool:
    name = "demo"
    description = "Demo tool"
    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "integer"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, arguments):
        return ToolResult(
            success=True,
            output=f"processed:{arguments['value']}",
        )


class FailingTool:
    name = "failing"
    description = "Failing tool"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments):
        return ToolResult(
            success=False,
            error="tool failure",
        )


def make_agent(
    *,
    registry: ToolRegistry,
    permission_policy: PermissionPolicy,
    confirmation_handler: ConfirmationHandler | None = None,
    mode: AgentMode = AgentMode.CHAT,
) -> Agent:
    return Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=permission_policy,
        confirmation_handler=(
            confirmation_handler or ConfirmationHandler()
        ),
        mode=mode,
    )


def test_execute_tool_uses_unified_pipeline() -> None:
    registry = ToolRegistry()
    registry.register(SuccessTool())

    agent = make_agent(
        registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
    )

    result = agent.execute_tool(
        "demo",
        {"value": 42},
        request_id="agent-call-1",
    )

    assert result.success is True
    assert result.output == "processed:42"


def test_execute_tool_preserves_tool_failure() -> None:
    registry = ToolRegistry()
    registry.register(FailingTool())

    agent = make_agent(
        registry=registry,
        permission_policy=PermissionPolicy(
            {
                "failing": PermissionLevel.ALLOW,
            }
        ),
    )

    result = agent.execute_tool(
        "failing",
        {},
        request_id="agent-call-2",
    )

    assert result.success is False
    assert result.error == "tool failure"


def test_execute_tool_denies_permission_through_pipeline() -> None:
    registry = ToolRegistry()
    registry.register(SuccessTool())

    agent = make_agent(
        registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.DENY,
            }
        ),
    )

    with pytest.raises(
        PermissionDeniedError,
        match="Permission denied",
    ):
        agent.execute_tool(
            "demo",
            {"value": 42},
            request_id="agent-call-3",
        )


def test_execute_tool_requests_confirmation_through_pipeline() -> None:
    registry = ToolRegistry()
    registry.register(SuccessTool())

    calls: list[str] = []

    def confirm(description: str) -> bool:
        calls.append(description)
        return True

    agent = make_agent(
        registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.CONFIRM,
            }
        ),
        confirmation_handler=ConfirmationHandler(
            callback=confirm,
        ),
    )

    result = agent.execute_tool(
        "demo",
        {"value": 7},
        request_id="agent-call-4",
    )

    assert result.success is True
    assert result.output == "processed:7"
    assert len(calls) == 1
    assert "demo" in calls[0]


def test_execute_tool_rejects_unavailable_mode_tool() -> None:
    registry = ToolRegistry()
    registry.register(SuccessTool())

    agent = make_agent(
        registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
    )

    agent.set_mode(AgentMode.CHAT)

    result = agent.execute_tool(
        "demo",
        {"value": 1},
        request_id="agent-call-5",
    )

    assert result.success is True
