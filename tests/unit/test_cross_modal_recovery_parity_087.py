"""0.8.7.5 Cross-modal recovery contract parity tests."""

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


@dataclass
class FixedRecoveryResolver:
    """Return one deterministic action request."""

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
            request_id="recovery-parity-087",
            metadata={
                "source": "voice",
                "voice_task_id": "voice-recovery-parity-087",
            },
        )


class SuccessfulRecoveryAgent:
    """Return a canonical successful recovery result."""

    def __init__(self) -> None:
        self.requests: list[ActionRequest] = []
        self.run_ids: list[str | None] = []
        self.task_ids: list[str | None] = []

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.requests.append(request)
        self.run_ids.append(run_id)
        self.task_ids.append(task_id)

        attempts = (
            RecoveryAttempt(
                attempt=1,
                max_attempts=2,
                success=False,
                failure_kind=RecoveryFailureKind.BACKEND_ERROR,
                error="temporary backend failure",
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


class FailedRecoveryAgent:
    """Return a canonical exhausted recovery failure."""

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        attempt = RecoveryAttempt(
            attempt=1,
            max_attempts=1,
            success=False,
            failure_kind=RecoveryFailureKind.BACKEND_ERROR,
            error="backend unavailable",
        )

        return RecoveryResult.failed(
            failure_kind=RecoveryFailureKind.BACKEND_ERROR,
            error="backend unavailable",
            attempts=(attempt,),
            exhausted=True,
        )


def test_voice_preserves_successful_recovery_contract() -> None:
    agent = SuccessfulRecoveryAgent()
    resolver: VoiceActionResolver = FixedRecoveryResolver()

    adapter = AgentVoiceActionAdapter(
        agent,
        resolver,
    )

    response = adapter.chat(
        "run recovery parity"
    )

    assert response.content == "recovered-value"

    assert len(agent.requests) == 1
    request = agent.requests[0]

    assert request.request_id == "recovery-parity-087"
    assert request.kind is ActionKind.TOOL
    assert request.name == "recovery_tool"
    assert request.arguments == {
        "value": 42,
    }

    assert agent.run_ids[0] is not None
    assert agent.run_ids[0].startswith("voice-")
    assert agent.task_ids[0] == "voice-recovery-parity-087"


def test_successful_recovery_contract_has_expected_attempt_semantics() -> None:
    agent = SuccessfulRecoveryAgent()

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="recovery_tool",
            arguments={"value": 42},
            request_id="recovery-parity-direct-087",
        ),
        run_id="run-087",
        task_id="task-087",
    )

    assert result.success is True
    assert result.failure_kind is None
    assert result.error is None
    assert result.exhausted is False

    assert result.attempt_count == 2
    assert result.last_attempt is not None
    assert result.last_attempt.attempt == 2
    assert result.last_attempt.max_attempts == 2
    assert result.last_attempt.success is True
    assert result.last_attempt.output_present is True

    assert isinstance(result.result, ActionResult)
    assert result.result.request_id == "recovery-parity-direct-087"
    assert result.result.success is True
    assert result.result.output == "recovered-value"


def test_voice_preserves_failed_recovery_contract() -> None:
    agent = FailedRecoveryAgent()

    adapter = AgentVoiceActionAdapter(
        agent,
        FixedRecoveryResolver(),
    )

    response = adapter.chat(
        "run recovery parity"
    )

    assert response.content == "backend unavailable"


def test_failed_recovery_contract_preserves_failure_kind_and_exhaustion() -> None:
    agent = FailedRecoveryAgent()

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="recovery_tool",
            arguments={"value": 42},
            request_id="recovery-failure-087",
        ),
    )

    assert result.success is False
    assert result.failure_kind is RecoveryFailureKind.BACKEND_ERROR
    assert result.error == "backend unavailable"
    assert result.exhausted is True

    assert result.attempt_count == 1
    assert result.last_attempt is not None
    assert result.last_attempt.attempt == 1
    assert result.last_attempt.max_attempts == 1
    assert result.last_attempt.success is False
    assert result.last_attempt.failure_kind is (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert result.last_attempt.error == "backend unavailable"

    assert result.result is None
