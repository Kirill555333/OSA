from __future__ import annotations

from osa.actions.contracts import ActionKind, ActionResult
from osa.actions.pipeline import ActionSafetyPipeline
from osa.core.agent import Agent, PermissionDeniedError
from osa.core.agent_action_policy import LegacyPermissionActionPolicy
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_action import RecoverableActionExecutor
from osa.recovery_contracts import RecoveryFailureKind
from osa.recovery_policy import MappingRecoveryPolicy
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
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments):
        return ToolResult(
            success=True,
            output="executed",
        )


class RecordingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return ActionResult.succeeded(
            request.request_id,
            output="executed",
        )


class FailingBackend:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return ActionResult.failed(
            request.request_id,
            "temporary backend failure",
        )


def test_permission_denial_never_reaches_recovery_executor() -> None:
    registry = ToolRegistry()
    registry.register(DemoTool())

    recovery_executor = RecordingBackend()

    pipeline = ActionSafetyPipeline(
        router=_tool_router(registry),
        permission_policy=LegacyPermissionActionPolicy(
            PermissionPolicy(
                {
                    "demo": PermissionLevel.DENY,
                }
            )
        ),
    )

    recoverable = RecoverableActionExecutor(
        pipeline.dispatch,
        max_attempts=3,
        sleeper=lambda _: None,
    )

    integration = AgentRecoveryIntegration(
        recoverable
    )

    agent = Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.DENY,
            }
        ),
        recovery_integration=integration,
    )

    try:
        agent.execute_tool(
            "demo",
            {},
            request_id="denied-call",
        )
    except PermissionDeniedError:
        pass

    assert recovery_executor.calls == 0


def test_retryable_backend_failure_reenters_unified_pipeline() -> None:
    backend = FailingBackend()
    registry = ToolRegistry()

    class BackendTool:
        name = "demo"
        description = "Backend test tool"
        parameters = {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

        def execute(self, arguments):
            return backend.execute(
                _request("retry-call")
            )

    registry.register(BackendTool())

    from osa.actions.router import ActionRouter
    from osa.actions.tool import ToolRegistryActionAdapter

    router = ActionRouter(
        {
            ActionKind.TOOL: ToolRegistryActionAdapter(
                registry
            )
        }
    )

    pipeline = ActionSafetyPipeline(
        router=router,
        permission_policy=LegacyPermissionActionPolicy(
            PermissionPolicy(
                {
                    "demo": PermissionLevel.ALLOW,
                }
            )
        ),
    )

    recoverable = RecoverableActionExecutor(
        pipeline.dispatch,
        max_attempts=3,
        policy=MappingRecoveryPolicy(
            {
                RecoveryFailureKind.BACKEND_ERROR: True,
            }
        ),
        sleeper=lambda _: None,
    )

    result = recoverable.execute(
        _request("retry-call")
    )

    assert result.success is False
    assert result.failure_kind is RecoveryFailureKind.BACKEND_ERROR
    assert result.attempt_count == 3
    assert backend.calls == 3


def _request(request_id: str):
    from osa.actions.contracts import ActionRequest

    return ActionRequest(
        request_id=request_id,
        kind=ActionKind.TOOL,
        name="demo",
        arguments={},
    )


def _tool_router(registry):
    from osa.actions.router import ActionRouter
    from osa.actions.tool import ToolRegistryActionAdapter

    return ActionRouter(
        {
            ActionKind.TOOL: ToolRegistryActionAdapter(
                registry
            )
        }
    )
