"""0.8.9.4 ExecutionContext to recovery correlation tests."""

from __future__ import annotations

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry


class FakeModel:
    def generate(self, request):
        return ModelResponse(
            content="ok",
            model_name="fake",
        )

    def generate_stream_events(self, request):
        raise NotImplementedError


class RecordingRecoveryExecutor:
    max_attempts = 4

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                ActionRequest,
                str | None,
                str | None,
            ]
        ] = []

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.calls.append(
            (
                request,
                run_id,
                task_id,
            )
        )

        return RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                request.request_id,
                output="recovered",
            ),
        )


def _agent(
    executor: RecordingRecoveryExecutor,
) -> Agent:
    return Agent(
        FakeModel(),
        tool_registry=ToolRegistry(),
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            executor
        ),
    )


def test_recovery_receives_canonical_explicit_scope() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _agent(executor)

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "value": 42,
            },
            request_id="context-recovery-1",
            metadata={
                "source": "voice",
                "run_id": "metadata-run",
                "task_id": "metadata-task",
                "round": 3,
            },
        ),
        run_id="explicit-run",
        task_id="explicit-task",
    )

    assert result.success is True
    assert len(executor.calls) == 1

    request, run_id, task_id = executor.calls[0]

    assert request.request_id == "context-recovery-1"
    assert request.kind is ActionKind.TOOL
    assert request.name == "demo"
    assert request.arguments == {
        "value": 42,
    }

    assert run_id == "explicit-run"
    assert task_id == "explicit-task"


def test_recovery_falls_back_to_request_context_metadata() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _agent(executor)

    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={},
        request_id="context-recovery-2",
        metadata={
            "source": "autonomous",
            "run_id": "metadata-run-2",
            "task_id": "metadata-task-2",
        },
    )

    result = agent.execute_action_with_recovery(
        request
    )

    assert result.success is True
    assert len(executor.calls) == 1

    recorded_request, run_id, task_id = executor.calls[0]

    assert recorded_request is request
    assert recorded_request.request_id == request.request_id
    assert run_id == "metadata-run-2"
    assert task_id == "metadata-task-2"


def test_recovery_request_identity_is_never_replaced_by_context() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _agent(executor)

    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={},
        request_id="context-recovery-3",
        metadata={
            "run_id": "run-3",
            "task_id": "task-3",
        },
    )

    result = agent.execute_action_with_recovery(
        request,
        run_id="different-run",
        task_id="different-task",
    )

    assert result.success is True

    recorded_request, run_id, task_id = executor.calls[0]

    assert recorded_request.request_id == (
        "context-recovery-3"
    )
    assert run_id == "different-run"
    assert task_id == "different-task"
