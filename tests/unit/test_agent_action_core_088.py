"""0.8.8.1 pre-refactor unified action execution contract tests."""

from __future__ import annotations

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry, ToolResult


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
    """Deterministic tool for the legacy execute_tool projection."""

    name = "demo"
    description = "Demo tool"
    parameters = {
        "type": "object",
        "properties": {
            "value": {
                "type": "integer",
            },
        },
        "required": [
            "value",
        ],
        "additionalProperties": False,
    }

    def execute(
        self,
        arguments,
    ) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"processed:{arguments['value']}",
        )


class RecordingRecoveryExecutor:
    """Record unified requests entering recovery."""

    max_attempts = 3

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
                output="processed:42",
            ),
        )


def _build_agent(
    executor: RecordingRecoveryExecutor,
) -> Agent:
    registry = ToolRegistry()
    registry.register(
        DemoTool()
    )

    return Agent(
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


def test_execute_tool_and_direct_action_share_core_request_semantics() -> None:
    direct_executor = RecordingRecoveryExecutor()
    direct_agent = _build_agent(
        direct_executor
    )

    direct_result = direct_agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": 42,
            },
            request_id="core-direct-1",
        ),
        run_id="run-direct",
        task_id="task-direct",
    )

    tool_executor = RecordingRecoveryExecutor()
    tool_agent = _build_agent(
        tool_executor
    )

    tool_result = tool_agent.execute_tool(
        "demo",
        {
            "value": 42,
        },
        request_id="core-tool-1",
    )

    assert direct_result.success is True
    assert isinstance(
        direct_result.result,
        ActionResult,
    )
    assert direct_result.result.success is True
    assert direct_result.result.output == "processed:42"

    assert tool_result.success is True
    assert tool_result.output == "processed:42"

    assert len(direct_executor.calls) == 1
    assert len(tool_executor.calls) == 1

    direct_request = direct_executor.calls[0]
    tool_request = tool_executor.calls[0]

    assert direct_request.kind is ActionKind.TOOL
    assert tool_request.kind is ActionKind.TOOL

    assert direct_request.name == tool_request.name == "demo"

    assert direct_request.arguments == tool_request.arguments == {
        "value": 42,
    }

    assert direct_request.request_id == "core-direct-1"
    assert tool_request.request_id == "core-tool-1"


def test_direct_action_preserves_run_and_task_context_at_public_boundary() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _build_agent(executor)

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": 42,
            },
            request_id="core-context-1",
        ),
        run_id="run-088",
        task_id="task-088",
    )

    assert result.success is True
    assert len(executor.calls) == 1

    request = executor.calls[0]

    assert request.request_id == "core-context-1"
    assert request.kind is ActionKind.TOOL
    assert request.name == "demo"
    assert request.arguments == {
        "value": 42,
    }


def test_execute_tool_keeps_legacy_tool_result_projection() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _build_agent(executor)

    result = agent.execute_tool(
        "demo",
        {
            "value": 42,
        },
        request_id="core-projection-1",
    )

    assert isinstance(
        result,
        ToolResult,
    )
    assert result.success is True
    assert result.output == "processed:42"

    assert len(executor.calls) == 1
    assert executor.calls[0].request_id == "core-projection-1"
    assert executor.calls[0].kind is ActionKind.TOOL
    assert executor.calls[0].name == "demo"
