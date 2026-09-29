from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind
from osa.core.agent_tool_round import (
    AgentToolRound,
    AgentToolRoundError,
)
from osa.models import ToolCall


def test_request_from_tool_call_preserves_unified_fields() -> None:
    request = AgentToolRound.request_from_tool_call(
        ToolCall(
            id="call_084",
            name="calculator",
            arguments={
                "expression": "10 + 5",
            },
        )
    )

    assert request.request_id == "call_084"
    assert request.kind is ActionKind.TOOL
    assert request.name == "calculator"
    assert request.arguments == {
        "expression": "10 + 5",
    }
    assert request.metadata["source"] == "agent"


def test_request_from_tool_call_preserves_metadata() -> None:
    request = AgentToolRound.request_from_tool_call(
        ToolCall(
            id="call_metadata",
            name="calculator",
            arguments={},
        ),
        metadata={
            "round": 4,
            "streaming": True,
        },
    )

    assert request.metadata["source"] == "agent"
    assert request.metadata["round"] == 4
    assert request.metadata["streaming"] is True


def test_request_from_tool_call_rejects_invalid_call() -> None:
    with pytest.raises(
        AgentToolRoundError,
        match="Invalid model tool call",
    ):
        AgentToolRound.request_from_tool_call(
            object(),  # type: ignore[arg-type]
        )


def test_request_rejects_invalid_metadata() -> None:
    with pytest.raises(
        AgentToolRoundError,
        match="metadata",
    ):
        AgentToolRound.request(
            "calculator",
            {},
            metadata=[],  # type: ignore[arg-type]
        )
