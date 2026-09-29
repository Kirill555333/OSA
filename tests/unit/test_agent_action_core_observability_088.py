"""0.8.8.6 Observability invariants for the unified action core."""

from __future__ import annotations

import json

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry, ToolResult
from osa.utils import EventLogger


class FakeModel:
    """Minimal model required by Agent."""

    def generate(self, request):
        return ModelResponse(
            content="ok",
            model_name="fake",
        )

    def generate_stream_events(self, request):
        raise NotImplementedError


class DemoTool:
    """Deterministic tool with sensitive-looking argument data."""

    name = "demo"
    description = "Demo tool"
    parameters = {
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
            },
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        arguments,
    ) -> ToolResult:
        self.calls.append(
            dict(arguments)
        )

        return ToolResult(
            success=True,
            output="completed",
        )


class RecoveryExecutor:
    """Return a successful unified recovery result."""

    max_attempts = 2

    def __init__(self) -> None:
        self.calls: list[ActionRequest] = []

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.calls.append(request)

        return RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                request.request_id,
                output="completed",
            ),
        )


def _build_agent(
    *,
    logger: EventLogger | None,
    tool: DemoTool,
    recovery: RecoveryExecutor,
) -> Agent:
    registry = ToolRegistry()
    registry.register(tool)

    return Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            recovery
        ),
        event_logger=logger,
    )


def test_observability_does_not_change_successful_action_result(
    tmp_path,
) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path)

    tool = DemoTool()
    recovery = RecoveryExecutor()

    agent = _build_agent(
        logger=logger,
        tool=tool,
        recovery=recovery,
    )

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": "sensitive-value",
            },
            request_id="obs-core-1",
        )
    )

    assert result.success is True
    assert isinstance(
        result.result,
        ActionResult,
    )
    assert result.result.success is True
    assert result.result.output == "completed"

    assert len(recovery.calls) == 1
    assert recovery.calls[0].request_id == "obs-core-1"


def test_observability_does_not_change_legacy_tool_projection(
    tmp_path,
) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path)

    tool = DemoTool()
    recovery = RecoveryExecutor()

    agent = _build_agent(
        logger=logger,
        tool=tool,
        recovery=recovery,
    )

    result = agent.execute_tool(
        "demo",
        {
            "value": "sensitive-value",
        },
        request_id="obs-core-2",
    )

    assert result.success is True
    assert result.output == "completed"

    assert len(recovery.calls) == 1
    assert recovery.calls[0].request_id == "obs-core-2"


def test_action_observability_contains_no_action_arguments(
    tmp_path,
) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path)

    tool = DemoTool()
    recovery = RecoveryExecutor()

    agent = _build_agent(
        logger=logger,
        tool=tool,
        recovery=recovery,
    )

    agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": "secret-value-088",
                "hidden": "also-secret",
            },
            request_id="obs-core-3",
        )
    )

    serialized = log_path.read_text(
        encoding="utf-8",
    )

    assert "secret-value-088" not in serialized
    assert "also-secret" not in serialized
    assert '"arguments"' not in serialized
    assert "argument_keys" not in serialized
    assert '"hidden"' not in serialized


def test_action_lifecycle_events_are_correlated(
    tmp_path,
) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = EventLogger(log_path)

    tool = DemoTool()
    recovery = RecoveryExecutor()

    agent = _build_agent(
        logger=logger,
        tool=tool,
        recovery=recovery,
    )

    agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={},
            request_id="obs-core-4",
        ),
        run_id="run-088",
        task_id="task-088",
    )

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
        "action.recovery.started",
        "action.recovery.completed",
        "action.completed",
    ]

    assert {
        record["data"]["request_id"]
        for record in action_events
    } == {"obs-core-4"}

    assert action_events[0]["data"]["action_kind"] == "tool"
    assert action_events[0]["data"]["action_name"] == "demo"
    assert action_events[-1]["data"]["success"] is True


def test_event_logger_is_fail_safe_for_invalid_write_target(
    tmp_path,
) -> None:
    logger = EventLogger(
        tmp_path / "events.jsonl"
    )

    logger._path = tmp_path / "events-as-directory"
    logger._path.mkdir()

    logger.log(
        "test.failure",
        value="ignored",
    )

    assert logger.enabled is True


def test_agent_without_event_logger_executes_normally() -> None:
    tool = DemoTool()

    registry = ToolRegistry()
    registry.register(tool)

    agent = Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
    )

    result = agent.execute_tool(
        "demo",
        {
            "value": "normal",
        },
        request_id="obs-core-6",
    )

    assert result.success is True
    assert result.output == "completed"
    assert tool.calls == [
        {
            "value": "normal",
        }
    ]
