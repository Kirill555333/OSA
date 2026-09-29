from __future__ import annotations

import json

from osa.actions.contracts import ActionKind, ActionResult
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry, ToolResult
from osa.utils import EventLogger


class FakeModel:
    def generate(self, request):
        return ModelResponse(
            content="ok",
            model_name="fake",
        )

    def generate_stream_events(self, request):
        raise NotImplementedError


class DemoTool:
    name = "demo"
    description = "Demo tool"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments):
        return ToolResult(
            success=True,
            output="direct",
        )


class RecordingRecoveryExecutor:
    max_attempts = 3

    def __init__(self) -> None:
        self.calls = []

    def execute(
        self,
        request,
        *,
        run_id=None,
        task_id=None,
    ):
        self.calls.append(
            (request, run_id, task_id)
        )

        return RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                request.request_id,
                output="recovered",
            )
        )


def test_agent_emits_recovery_lifecycle_events(
    tmp_path,
) -> None:
    log_path = tmp_path / "osa-events.jsonl"
    logger = EventLogger(log_path)

    registry = ToolRegistry()
    registry.register(DemoTool())

    executor = RecordingRecoveryExecutor()

    agent = Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            executor
        ),
        event_logger=logger,
    )

    result = agent.execute_tool(
        "demo",
        {},
        request_id="recovery-obs-1",
    )

    assert result.success is True
    assert result.output == "recovered"

    records = [
        json.loads(line)
        for line in log_path.read_text(
            encoding="utf-8",
        ).splitlines()
    ]

    events = [
        record
        for record in records
        if record["event"].startswith("action.")
    ]

    assert [
        record["event"]
        for record in events
    ] == [
        "action.requested",
        "action.recovery.started",
        "action.recovery.completed",
        "action.completed",
    ]

    requested = events[0]["data"]
    recovery_started = events[1]["data"]
    recovery_completed = events[2]["data"]
    completed = events[3]["data"]

    assert requested["request_id"] == "recovery-obs-1"
    assert recovery_started["request_id"] == "recovery-obs-1"
    assert recovery_completed["request_id"] == "recovery-obs-1"
    assert completed["request_id"] == "recovery-obs-1"

    assert recovery_started["max_attempts"] == 3
    assert recovery_completed["status"] == "completed"
    assert "round" not in recovery_completed
    assert completed["success"] is True


def test_recovery_executor_receives_original_request() -> None:
    registry = ToolRegistry()
    registry.register(DemoTool())

    executor = RecordingRecoveryExecutor()

    agent = Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            executor
        ),
    )

    agent.execute_tool(
        "demo",
        {"value": 42},
        request_id="recovery-request-1",
    )

    assert len(executor.calls) == 1

    request, run_id, task_id = executor.calls[0]

    assert request.request_id == "recovery-request-1"
    assert request.kind is ActionKind.TOOL
    assert request.name == "demo"
    assert request.arguments == {"value": 42}
    assert request.metadata["source"] == "agent"

    assert run_id is None
    assert task_id is None
