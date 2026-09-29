"""0.8.8.7 Legacy Agent API compatibility regression tests."""

from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent, AgentError, PermissionDeniedError
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryFailureKind, RecoveryResult
from osa.tools import ToolRegistry, ToolResult


class FakeModel:
    """Minimal model for legacy API tests."""

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

    def execute(self, arguments):
        return ToolResult(
            success=True,
            output=f"processed:{arguments['value']}",
        )


class RecoveryExecutor:
    max_attempts = 2

    def __init__(
        self,
        result: RecoveryResult,
    ) -> None:
        self.result = result
        self.calls: list[ActionRequest] = []

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.calls.append(request)
        return self.result


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(DemoTool())
    return registry


def _agent(
    *,
    permission: PermissionLevel = PermissionLevel.ALLOW,
    recovery: RecoveryExecutor | None = None,
) -> Agent:
    return Agent(
        FakeModel(),
        tool_registry=_registry(),
        permission_policy=PermissionPolicy(
            {
                "demo": permission,
            }
        ),
        recovery_integration=(
            AgentRecoveryIntegration(recovery)
            if recovery is not None
            else None
        ),
    )


def test_execute_tool_keeps_tool_result_type_without_recovery() -> None:
    agent = _agent()

    result = agent.execute_tool(
        "demo",
        {"value": 42},
        request_id="legacy-1",
    )

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.output == "processed:42"


def test_execute_tool_keeps_permission_denied_error() -> None:
    agent = _agent(
        permission=PermissionLevel.DENY,
    )

    with pytest.raises(
        PermissionDeniedError,
        match="Permission denied",
    ):
        agent.execute_tool(
            "demo",
            {"value": 42},
            request_id="legacy-2",
        )


def test_execute_action_with_recovery_keeps_recovery_result_type() -> None:
    recovery = RecoveryExecutor(
        RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                "legacy-3",
                output="recovered",
            )
        )
    )

    agent = _agent(
        recovery=recovery,
    )

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={"value": 42},
            request_id="legacy-3",
        ),
        run_id="run-legacy",
        task_id="task-legacy",
    )

    assert isinstance(result, RecoveryResult)
    assert result.success is True
    assert isinstance(result.result, ActionResult)
    assert result.result.output == "recovered"


def test_execute_action_with_recovery_preserves_context() -> None:
    recovery = RecoveryExecutor(
        RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                "legacy-4",
                output="ok",
            )
        )
    )

    agent = _agent(
        recovery=recovery,
    )

    agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={},
            request_id="legacy-4",
        ),
        run_id="run-legacy-4",
        task_id="task-legacy-4",
    )

    assert len(recovery.calls) == 1

    request = recovery.calls[0]

    assert request.request_id == "legacy-4"
    assert request.kind is ActionKind.TOOL
    assert request.name == "demo"


def test_execute_action_with_recovery_stays_opt_in() -> None:
    agent = _agent()

    with pytest.raises(
        AgentError,
        match="not configured",
    ):
        agent.execute_action_with_recovery(
            ActionRequest(
                kind=ActionKind.TOOL,
                name="demo",
                arguments={},
                request_id="legacy-5",
            )
        )


def test_failed_recovery_result_preserves_failure_contract() -> None:
    recovery = RecoveryExecutor(
        RecoveryResult.failed(
            failure_kind=RecoveryFailureKind.BACKEND_ERROR,
            error="backend failure",
            exhausted=True,
        )
    )

    agent = _agent(
        recovery=recovery,
    )

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={},
            request_id="legacy-6",
        )
    )

    assert isinstance(result, RecoveryResult)
    assert result.success is False
    assert result.failure_kind is RecoveryFailureKind.BACKEND_ERROR
    assert result.error == "backend failure"
    assert result.exhausted is True
