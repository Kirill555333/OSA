"""0.8.7.3 Voice/recovery parity regression tests."""

from __future__ import annotations

from dataclasses import dataclass

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter,
    VoiceActionResolver,
)
from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryFailureKind,
    RecoveryResult,
)
from osa.tasks.action import TaskAction


@dataclass
class FixedVoiceResolver:
    """Return one deterministic unified TOOL action."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        assert command == "run recovery parity"

        return ActionRequest(
            kind=ActionKind.TOOL,
            name="recovery_tool",
            arguments={
                "value": 42,
            },
            metadata={
                "source": "voice",
                "voice_task_id": "voice-task-087",
            },
            request_id="voice-recovery-087",
        )


class RecordingRecoveryAgent:
    """Simulate Agent recovery while recording the unified request."""

    def __init__(self) -> None:
        self.requests: list[ActionRequest] = []
        self.run_ids: list[str | None] = []
        self.task_ids: list[str | None] = []
        self.calls = 0

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.calls += 1
        self.requests.append(request)
        self.run_ids.append(run_id)
        self.task_ids.append(task_id)

        attempts = (
            RecoveryAttempt(
                attempt=1,
                max_attempts=2,
                success=False,
                failure_kind=RecoveryFailureKind.BACKEND_ERROR,
                error="temporary backend error",
            ),
            RecoveryAttempt(
                attempt=2,
                max_attempts=2,
                success=True,
                output_present=True,
            ),
        )

        return RecoveryResult.succeeded(
            attempts=attempts,
            result=ActionResult.succeeded(
                request.request_id,
                output="recovered-value",
            ),
        )


class FailureRecoveryAgent:
    """Return a failed recovery result without exposing internal retry logic."""

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        return RecoveryResult.failed(
            failure_kind=RecoveryFailureKind.BACKEND_ERROR,
            error="backend unavailable",
            exhausted=True,
        )


def test_voice_passes_one_unified_request_to_recovery() -> None:
    agent = RecordingRecoveryAgent()
    resolver: VoiceActionResolver = FixedVoiceResolver()

    adapter = AgentVoiceActionAdapter(
        agent,
        resolver,
    )

    response = adapter.chat(
        "run recovery parity"
    )

    assert response.content == "recovered-value"

    assert agent.calls == 1
    assert len(agent.requests) == 1

    request = agent.requests[0]

    assert isinstance(request, ActionRequest)
    assert request.kind is ActionKind.TOOL
    assert request.name == "recovery_tool"
    assert request.arguments == {
        "value": 42,
    }
    assert request.request_id == "voice-recovery-087"
    assert request.metadata == {
        "source": "voice",
        "voice_task_id": "voice-task-087",
    }

    assert agent.run_ids[0] is not None
    assert agent.run_ids[0].startswith("voice-")
    assert agent.task_ids[0] == "voice-task-087"


def test_voice_does_not_retry_or_interpret_recovery_policy_itself() -> None:
    agent = FailureRecoveryAgent()

    adapter = AgentVoiceActionAdapter(
        agent,
        FixedVoiceResolver(),
    )

    response = adapter.chat(
        "run recovery parity"
    )

    assert response.content == "backend unavailable"


def test_voice_resolver_and_task_action_share_tool_identity() -> None:
    task_action = TaskAction(
        tool_name="recovery_tool",
        arguments={
            "value": 42,
        },
    )

    request = FixedVoiceResolver().resolve(
        "run recovery parity"
    )

    assert request.kind is ActionKind.TOOL
    assert request.name == task_action.tool_name
    assert request.arguments == task_action.arguments
