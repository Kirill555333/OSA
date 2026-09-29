from __future__ import annotations

import json

from osa.core import Agent, PermissionDeniedError
from osa.models import ModelInterface, ModelRequest, ModelResponse, ToolCall
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry
from osa.utils import EventLogger


class DeniedModel(ModelInterface):
    """Model that requests one denied tool call."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "denied-observability-model"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="deny-obs-1",
                        name="calculator",
                        arguments={"expression": "2 + 2"},
                    ),
                ),
            )

        return ModelResponse(
            content="Операция отклонена.",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def test_permission_denial_emits_action_denied(
    tmp_path,
) -> None:
    log_path = tmp_path / "osa-events.jsonl"

    logger = EventLogger(log_path)

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        DeniedModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.DENY,
            }
        ),
        event_logger=logger,
    )

    response = agent.chat(
        "calculate"
    )

    assert response.content == "Операция отклонена."

    records = [
        json.loads(line)
        for line in log_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    denied = [
        record
        for record in records
        if record["event"] == "action.denied"
    ]

    assert len(denied) == 1

    data = denied[0]["data"]

    assert data["request_id"] == "deny-obs-1"
    assert data["action_kind"] == "tool"
    assert data["action_name"] == "calculator"
    assert data["status"] == "denied"
    assert data["round"] == 1
    assert data["decision"] == "denied"
    assert "2 + 2" not in str(data)


def test_permission_denial_does_not_emit_completed_action(
    tmp_path,
) -> None:
    log_path = tmp_path / "osa-events.jsonl"

    logger = EventLogger(log_path)

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        DeniedModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.DENY,
            }
        ),
        event_logger=logger,
    )

    agent.chat(
        "calculate"
    )

    records = [
        json.loads(line)
        for line in log_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    action_events = [
        record["event"]
        for record in records
        if record["event"].startswith("action.")
    ]

    assert action_events == [
        "action.requested",
        "action.denied",
    ]
