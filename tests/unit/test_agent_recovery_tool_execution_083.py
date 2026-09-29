from __future__ import annotations

from osa.actions.contracts import ActionKind, ActionResult
from osa.core.agent import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
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
            "value": {"type": "integer"},
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def execute(self, arguments):
        return ToolResult(
            success=True,
            output=f"direct:{arguments['value']}",
        )


class RecordingRecoveryExecutor:
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

        action_result = ActionResult.succeeded(
            request.request_id,
            output=f"recovered:{request.arguments['value']}",
        )

        return RecoveryResult.succeeded(
            result=action_result,
        )


def test_execute_tool_uses_configured_recovery_integration() -> None:
    registry = ToolRegistry()
    registry.register(DemoTool())

    executor = RecordingRecoveryExecutor()
    integration = AgentRecoveryIntegration(executor)

    agent = Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=integration,
    )

    result = agent.execute_tool(
        "demo",
        {"value": 42},
        request_id="recovery-call-1",
    )

    assert result.success is True
    assert result.output == "recovered:42"

    request, run_id, task_id = executor.calls[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "demo"
    assert request.arguments == {"value": 42}
    assert request.request_id == "recovery-call-1"
    assert request.metadata["source"] == "agent"
    assert run_id is None
    assert task_id is None
