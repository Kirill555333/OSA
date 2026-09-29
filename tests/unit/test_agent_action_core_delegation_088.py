"""0.8.8.2 Unified action core delegation tests."""

from __future__ import annotations

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.models import ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry, ToolResult


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
        "properties": {
            "value": {
                "type": "integer",
            },
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, arguments) -> ToolResult:
        return ToolResult(
            success=True,
            output=f"tool:{arguments['value']}",
        )


class RecoveryExecutor:
    max_attempts = 2

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        return RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                request.request_id,
                output="recovery:42",
            ),
        )


def _agent(
    *,
    recovery: bool,
) -> Agent:
    registry = ToolRegistry()
    registry.register(DemoTool())

    from osa.core.agent_recovery import AgentRecoveryIntegration

    return Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=(
            AgentRecoveryIntegration(
                RecoveryExecutor()
            )
            if recovery
            else None
        ),
    )


def test_direct_public_api_returns_core_recovery_projection() -> None:
    agent = _agent(
        recovery=True
    )

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": 42,
            },
            request_id="delegation-recovery-1",
        ),
        run_id="run-1",
        task_id="task-1",
    )

    assert isinstance(
        result,
        RecoveryResult,
    )
    assert result.success is True
    assert isinstance(
        result.result,
        ActionResult,
    )
    assert result.result.output == "recovery:42"


def test_legacy_execute_tool_projects_core_into_tool_result() -> None:
    agent = _agent(
        recovery=False
    )

    result = agent.execute_tool(
        "demo",
        {
            "value": 42,
        },
        request_id="delegation-tool-1",
    )

    assert isinstance(
        result,
        ToolResult,
    )
    assert result.success is True
    assert result.output == "tool:42"


def test_core_without_recovery_uses_safety_pipeline_only() -> None:
    agent = _agent(
        recovery=False
    )

    (
        action_result,
        recovery_result,
        context,
    ) = agent._execute_action_core(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": 7,
            },
            request_id="delegation-core-1",
        )
    )

    assert recovery_result is None

    assert context.request_id == "delegation-core-1"
    assert context.run_id is None
    assert context.task_id is None
    assert context.round_number is None

    assert isinstance(
        action_result,
        ActionResult,
    )
    assert action_result.success is True
    assert action_result.output == "tool:7"


def test_core_with_recovery_returns_both_action_and_recovery_results() -> None:
    agent = _agent(
        recovery=True
    )

    (
        action_result,
        recovery_result,
        context,
    ) = agent._execute_action_core(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": 42,
            },
            request_id="delegation-core-2",
        ),
        run_id="run-core",
        task_id="task-core",
    )

    assert isinstance(
        recovery_result,
        RecoveryResult,
    )
    assert isinstance(
        action_result,
        ActionResult,
    )

    assert action_result.request_id == "delegation-core-2"
    assert action_result.success is True
    assert action_result.output == "recovery:42"

    assert recovery_result.result is action_result

    assert context.request_id == "delegation-core-2"
    assert context.run_id == "run-core"
    assert context.task_id == "task-core"
