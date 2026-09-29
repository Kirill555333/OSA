"""0.8.8.4 Voice integration with the unified action core."""

from __future__ import annotations

from dataclasses import dataclass

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter,
)
from osa.models import ModelInterface, ModelRequest, ModelResponse
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry


class FakeModel(ModelInterface):
    """Model that must never execute the Voice action itself."""

    @property
    def model_name(self) -> str:
        return "voice-core-088"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Voice action must use execute_action_with_recovery()."
        )

    def health_check(self) -> bool:
        return True


@dataclass
class FixedVoiceResolver:
    """Return one canonical TOOL ActionRequest."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        assert command == "run voice core"

        return ActionRequest(
            kind=ActionKind.TOOL,
            name="voice_tool",
            arguments={
                "value": 42,
            },
            request_id="voice-core-088",
            metadata={
                "source": "voice",
                "voice_task_id": "voice-task-core-088",
            },
        )


class RecordingRecoveryExecutor:
    """Record the exact unified request and execution context."""

    max_attempts = 3

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


def _build_agent(
    executor: RecordingRecoveryExecutor,
) -> Agent:
    registry = ToolRegistry()

    class VoiceTool:
        name = "voice_tool"
        description = "Voice integration tool"
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

        def execute(self, arguments):
            raise AssertionError(
                "Voice tool must not be executed directly."
            )

    registry.register(
        VoiceTool()
    )

    return Agent(
        FakeModel(),
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "voice_tool": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            executor
        ),
    )


def test_voice_uses_the_unified_action_core() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _build_agent(executor)

    adapter = AgentVoiceActionAdapter(
        agent,
        FixedVoiceResolver(),
    )

    response = adapter.chat(
        "run voice core"
    )

    assert response.content == "42"

    assert len(executor.calls) == 1

    request, run_id, task_id = executor.calls[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "voice_tool"
    assert request.arguments == {
        "value": 42,
    }
    assert request.request_id == "voice-core-088"

    assert request.metadata["source"] == "voice"
    assert request.metadata["voice_task_id"] == (
        "voice-task-core-088"
    )

    assert run_id is not None
    assert run_id.startswith("voice-")
    assert task_id == "voice-task-core-088"


def test_voice_does_not_call_model_or_tool_backend() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _build_agent(executor)

    adapter = AgentVoiceActionAdapter(
        agent,
        FixedVoiceResolver(),
    )

    response = adapter.chat(
        "run voice core"
    )

    assert response.content == "42"
    assert len(executor.calls) == 1


def test_voice_request_identity_reaches_recovery_unchanged() -> None:
    executor = RecordingRecoveryExecutor()
    agent = _build_agent(executor)

    adapter = AgentVoiceActionAdapter(
        agent,
        FixedVoiceResolver(),
    )

    adapter.chat(
        "run voice core"
    )

    request = executor.calls[0][0]

    assert request.request_id == "voice-core-088"
    assert request.kind is ActionKind.TOOL
    assert request.name == "voice_tool"
    assert request.arguments == {
        "value": 42,
    }
