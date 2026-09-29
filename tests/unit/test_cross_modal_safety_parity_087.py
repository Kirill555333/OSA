"""0.8.7.4 Voice safety-boundary regression tests."""

from __future__ import annotations

from dataclasses import dataclass

from osa.actions.contracts import ActionKind, ActionRequest
from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter,
    VoiceActionResolver,
)
from osa.recovery_contracts import (
    RecoveryFailureKind,
    RecoveryResult,
)


@dataclass
class FixedSafetyResolver:
    """Return one deterministic action for the voice boundary."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        assert command == "execute protected action"

        return ActionRequest(
            kind=ActionKind.TOOL,
            name="protected_tool",
            arguments={
                "value": 42,
            },
            metadata={
                "source": "voice",
                "voice_task_id": "voice-safety-087",
            },
            request_id="voice-safety-087",
        )


class SafetyBoundaryAgent:
    """
    Model the Agent recovery boundary.

    Voice never receives or decides the permission/confirmation policy.
    The boundary returns the canonical recovery outcome.
    """

    def __init__(
        self,
        failure_kind: RecoveryFailureKind,
        error: str,
    ) -> None:
        self.failure_kind = failure_kind
        self.error = error
        self.requests: list[ActionRequest] = []
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

        return RecoveryResult.failed(
            failure_kind=self.failure_kind,
            error=self.error,
        )


def _build_adapter(
    failure_kind: RecoveryFailureKind,
    error: str,
) -> tuple[
    AgentVoiceActionAdapter,
    SafetyBoundaryAgent,
]:
    agent = SafetyBoundaryAgent(
        failure_kind,
        error,
    )

    resolver: VoiceActionResolver = FixedSafetyResolver()

    return (
        AgentVoiceActionAdapter(
            agent,
            resolver,
        ),
        agent,
    )


def test_voice_propagates_permission_denial_from_unified_recovery() -> None:
    adapter, agent = _build_adapter(
        RecoveryFailureKind.PERMISSION_DENIED,
        "Permission denied for protected_tool.",
    )

    response = adapter.chat(
        "execute protected action"
    )

    assert response.content == (
        "Permission denied for protected_tool."
    )

    assert agent.calls == 1
    assert len(agent.requests) == 1

    request = agent.requests[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "protected_tool"
    assert request.request_id == "voice-safety-087"
    assert request.metadata["source"] == "voice"


def test_voice_propagates_confirmation_denial_from_unified_recovery() -> None:
    adapter, agent = _build_adapter(
        RecoveryFailureKind.CONFIRMATION_DENIED,
        "Action confirmation was not granted.",
    )

    response = adapter.chat(
        "execute protected action"
    )

    assert response.content == (
        "Action confirmation was not granted."
    )

    assert agent.calls == 1
    assert len(agent.requests) == 1

    request = agent.requests[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "protected_tool"
    assert request.request_id == "voice-safety-087"


def test_voice_does_not_convert_safety_denial_into_success() -> None:
    for failure_kind, error in (
        (
            RecoveryFailureKind.PERMISSION_DENIED,
            "permission denied",
        ),
        (
            RecoveryFailureKind.CONFIRMATION_DENIED,
            "confirmation denied",
        ),
    ):
        adapter, agent = _build_adapter(
            failure_kind,
            error,
        )

        response = adapter.chat(
            "execute protected action"
        )

        assert response.content == error
        assert agent.calls == 1
        assert len(agent.requests) == 1
