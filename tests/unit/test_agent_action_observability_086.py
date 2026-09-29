from __future__ import annotations

import json

from osa.core import Agent
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry
from osa.utils import EventLogger


class ObservedToolModel(ModelInterface):
    """Model that requests one calculator tool."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "observed-tool-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="obs-call-1",
                        name="calculator",
                        arguments={
                            "expression": "12 + 30",
                            "secret": "must-not-be-logged",
                        },
                    ),
                ),
            )

        return ModelResponse(
            content="Ответ: 42.",
            model_name=self.model_name,
            finish_reason="stop",
        )

    def generate_stream_events(
        self,
        request: ModelRequest,
    ):
        raise NotImplementedError

    def health_check(self) -> bool:
        return True


def test_agent_action_events_are_correlated_and_safe(
    tmp_path,
) -> None:
    log_path = tmp_path / "osa-events.jsonl"

    logger = EventLogger(log_path)

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        ObservedToolModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.ALLOW,
            }
        ),
        event_logger=logger,
    )

    response = agent.chat(
        "Calculate the value."
    )

    assert response.content == "Ответ: 42."

    records = [
        json.loads(line)
        for line in log_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    action_events = [
        record
        for record in records
        if record["event"].startswith("action.")
    ]

    assert [
        record["event"]
        for record in action_events
    ] == [
        "action.requested",
        "action.completed",
    ]

    requested = action_events[0]["data"]
    completed = action_events[1]["data"]

    assert requested["request_id"] == "obs-call-1"
    assert completed["request_id"] == "obs-call-1"

    assert requested["action_kind"] == "tool"
    assert completed["action_kind"] == "tool"

    assert requested["action_name"] == "calculator"
    assert completed["action_name"] == "calculator"

    assert requested["round"] == 1
    assert completed["round"] == 1

    assert completed["success"] is True
    assert completed["output_length"] == 2

    serialized = log_path.read_text(
        encoding="utf-8"
    )

    assert "12 + 30" not in serialized
    assert "must-not-be-logged" not in serialized
    assert "expression" not in serialized
    assert "secret" not in serialized
    assert "arguments" not in serialized
    assert "argument_keys" not in serialized


def test_agent_action_requested_contains_no_argument_metadata(
    tmp_path,
) -> None:
    log_path = tmp_path / "osa-events.jsonl"

    logger = EventLogger(log_path)

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        ObservedToolModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.ALLOW,
            }
        ),
        event_logger=logger,
    )

    agent.chat("Calculate the value.")

    records = [
        json.loads(line)
        for line in log_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    requested = next(
        record
        for record in records
        if record["event"] == "action.requested"
    )

    data = requested["data"]

    assert "arguments" not in data
    assert "argument_keys" not in data
    assert "expression" not in str(data)
    assert "secret" not in str(data)
    assert "12 + 30" not in str(data)
