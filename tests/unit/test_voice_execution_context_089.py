"""0.8.9.5 Voice execution-context parity tests."""

from __future__ import annotations

from dataclasses import dataclass

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.core.agent_voice_action import AgentVoiceActionAdapter
from osa.core.execution_context import ExecutionContext
from osa.models import ModelInterface, ModelRequest, ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry


@dataclass
class VoiceResolver:
    """Resolve one command into a canonical voice action."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        assert command == "run context voice"

        return ActionRequest(
            kind=ActionKind.TOOL,
            name="voice_tool",
            arguments={
                "value": 42,
            },
            request_id="voice-context-089",
            metadata={
                "source": "voice",
                "voice_task_id": "voice-task-089",
            },
        )


class RecordingRecoveryExecutor:
    """Record the request and recovery scope received from Voice."""

    max_attempts = 2

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
                output="42",
            ),
        )


class UnusedModel(ModelInterface):
    """Model surface required by Agent but unused by Voice."""

    @property
    def model_name(self) -> str:
        return "voice-context-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Voice context test must not call Agent model."
        )

    def health_check(self) -> bool:
        return True


def _agent(
    executor: RecordingRecoveryExecutor,
) -> Agent:
    return Agent(
        UnusedModel(),
        tool_registry=ToolRegistry(),
        permission_policy=PermissionPolicy(
            {
                "voice_tool": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            executor
        ),
    )


def test_voice_builds_canonical_execution_context() -> None:
    request = VoiceResolver().resolve(
        "run context voice"
    )

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.request_id == "voice-context-089"
    assert context.source == "voice"
    assert context.task_id == "voice-task-089"
    assert context.run_id is None
    assert context.round_number is None
    assert context.metadata == {}


def test_voice_propagates_context_scope_to_recovery() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _agent(executor)

    adapter = AgentVoiceActionAdapter(
        agent,
        VoiceResolver(),
    )

    response = adapter.chat(
        "run context voice"
    )

    assert response.content == "42"
    assert len(executor.calls) == 1

    request, run_id, task_id = executor.calls[0]

    context = ExecutionContext.from_action_request(
        request,
        run_id=run_id,
        task_id=task_id,
    )

    assert context.request_id == "voice-context-089"
    assert context.source == "voice"
    assert context.task_id == "voice-task-089"
    assert context.run_id == run_id
    assert context.run_id is not None
    assert context.run_id.startswith("voice-")
    assert context.round_number is None
    assert context.metadata == {}


def test_voice_context_does_not_copy_action_arguments() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="voice_tool",
        arguments={
            "value": 42,
            "secret": "must-not-become-context",
        },
        request_id="voice-context-privacy-089",
        metadata={
            "source": "voice",
            "voice_task_id": "voice-task-privacy-089",
        },
    )

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.request_id == (
        "voice-context-privacy-089"
    )
    assert context.task_id == (
        "voice-task-privacy-089"
    )
    assert context.source == "voice"
    assert context.metadata == {}
    assert "value" not in context.metadata
    assert "secret" not in context.metadata
    assert "must-not-become-context" not in str(context)


def test_voice_task_id_is_not_duplicated_in_metadata() -> None:
    request = ActionRequest(
        kind=ActionKind.TOOL,
        name="voice_tool",
        arguments={},
        request_id="voice-context-dedup-089",
        metadata={
            "source": "voice",
            "voice_task_id": "voice-task-dedup-089",
            "custom": "kept",
        },
    )

    context = ExecutionContext.from_action_request(
        request
    )

    assert context.task_id == "voice-task-dedup-089"
    assert context.metadata == {
        "custom": "kept",
    }
