from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind
from osa.core.agent_action import (
    AgentActionBridge,
    AgentActionBridgeError,
)
from osa.models import ToolCall


def test_tool_request_builds_unified_tool_action() -> None:
    request = AgentActionBridge.tool_request(
        "calculator",
        {
            "expression": "25 * 4",
        },
    )

    assert request.kind is ActionKind.TOOL
    assert request.name == "calculator"
    assert request.arguments == {
        "expression": "25 * 4",
    }
    assert request.metadata["source"] == "agent"
    assert request.request_id


def test_tool_request_copies_arguments_and_metadata() -> None:
    arguments = {
        "value": 42,
    }
    metadata = {
        "round": 2,
    }

    request = AgentActionBridge.tool_request(
        "demo",
        arguments,
        metadata=metadata,
    )

    arguments["value"] = 99
    metadata["round"] = 7

    assert request.arguments == {
        "value": 42,
    }
    assert request.metadata["round"] == 2
    assert request.metadata["source"] == "agent"


def test_action_request_supports_non_tool_actions() -> None:
    request = AgentActionBridge.action_request(
        kind=ActionKind.DESKTOP,
        name="desktop_launch_application",
        arguments={
            "command": "Google Chrome",
        },
        metadata={
            "origin": "agent-chat",
        },
    )

    assert request.kind is ActionKind.DESKTOP
    assert request.name == "desktop_launch_application"
    assert request.arguments == {
        "command": "Google Chrome",
    }
    assert request.metadata["origin"] == "agent-chat"
    assert request.metadata["source"] == "agent"


def test_from_tool_call_preserves_call_id() -> None:
    tool_call = ToolCall(
        id="call_123",
        name="calculator",
        arguments={
            "expression": "347 * 29",
        },
    )

    request = AgentActionBridge.from_tool_call(tool_call)

    assert request.request_id == "call_123"
    assert request.kind is ActionKind.TOOL
    assert request.name == "calculator"
    assert request.arguments == {
        "expression": "347 * 29",
    }
    assert request.metadata["source"] == "agent"


def test_from_tool_call_preserves_extra_metadata() -> None:
    tool_call = ToolCall(
        id="call_456",
        name="calculator",
        arguments={},
    )

    request = AgentActionBridge.from_tool_call(
        tool_call,
        metadata={
            "round": 3,
        },
    )

    assert request.metadata["round"] == 3
    assert request.metadata["source"] == "agent"


@pytest.mark.parametrize(
    "name",
    [
        "",
        "   ",
        None,
    ],
)
def test_action_request_rejects_invalid_name(
    name: str | None,
) -> None:
    with pytest.raises(
        AgentActionBridgeError,
        match="Action name",
    ):
        AgentActionBridge.action_request(
            kind=ActionKind.TOOL,
            name=name,  # type: ignore[arg-type]
        )


def test_action_request_rejects_invalid_arguments() -> None:
    with pytest.raises(
        AgentActionBridgeError,
        match="arguments",
    ):
        AgentActionBridge.action_request(
            kind=ActionKind.TOOL,
            name="calculator",
            arguments=["invalid"],  # type: ignore[arg-type]
        )


def test_action_request_rejects_invalid_metadata() -> None:
    with pytest.raises(
        AgentActionBridgeError,
        match="metadata",
    ):
        AgentActionBridge.action_request(
            kind=ActionKind.TOOL,
            name="calculator",
            metadata=["invalid"],  # type: ignore[arg-type]
        )


def test_action_request_rejects_invalid_request_id() -> None:
    with pytest.raises(
        AgentActionBridgeError,
        match="request_id",
    ):
        AgentActionBridge.action_request(
            kind=ActionKind.TOOL,
            name="calculator",
            request_id="   ",
        )


def test_from_tool_call_rejects_invalid_type() -> None:
    with pytest.raises(
        AgentActionBridgeError,
        match="ToolCall",
    ):
        AgentActionBridge.from_tool_call(
            object(),  # type: ignore[arg-type]
        )


def test_action_request_rejects_invalid_kind() -> None:
    with pytest.raises(
        AgentActionBridgeError,
        match="ActionKind",
    ):
        AgentActionBridge.action_request(
            kind="tool",  # type: ignore[arg-type]
            name="calculator",
        )
