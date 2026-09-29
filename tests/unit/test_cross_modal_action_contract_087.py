"""0.8.7 cross-modal action contract regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter,
    TaskActionResolverVoiceAdapter,
    VoiceActionResponse,
)
from osa.recovery_contracts import RecoveryResult
from osa.tasks.action import TaskAction


@dataclass
class StubTaskActionResolver:
    """Return one deterministic task action."""

    tool_name: str = "calculator"
    arguments: dict[str, Any] | None = None

    def resolve(
        self,
        task: Any,
        outputs: dict[str, str],
    ) -> TaskAction:
        assert task.description == "calculate 12 + 30"
        assert outputs == {}

        return TaskAction(
            tool_name=self.tool_name,
            arguments=self.arguments or {
                "expression": "12 + 30",
            },
        )


class RecordingVoiceAgent:
    """Capture the unified action request passed by the voice adapter."""

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

        return RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                request.request_id,
                output="42",
            )
        )


def test_voice_resolver_produces_standard_tool_action_request() -> None:
    resolver = TaskActionResolverVoiceAdapter(
        StubTaskActionResolver()
    )

    request = resolver.resolve(
        "calculate 12 + 30"
    )

    assert isinstance(request, ActionRequest)
    assert request.kind is ActionKind.TOOL
    assert request.name == "calculator"
    assert request.arguments == {
        "expression": "12 + 30",
    }
    assert request.metadata["source"] == "voice"

    voice_task_id = request.metadata["voice_task_id"]

    assert isinstance(voice_task_id, str)
    assert voice_task_id.strip()
    assert request.request_id


def test_voice_adapter_passes_standard_action_to_agent_recovery() -> None:
    agent = RecordingVoiceAgent()
    resolver = TaskActionResolverVoiceAdapter(
        StubTaskActionResolver(
            arguments={
                "expression": "12 + 30",
                "secret": "must-not-change",
            }
        )
    )

    adapter = AgentVoiceActionAdapter(
        agent,
        resolver,
    )

    response = adapter.chat(
        "calculate 12 + 30"
    )

    assert isinstance(response, VoiceActionResponse)
    assert response.content == "42"

    assert len(agent.requests) == 1

    request = agent.requests[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "calculator"
    assert request.arguments == {
        "expression": "12 + 30",
        "secret": "must-not-change",
    }
    assert request.metadata["source"] == "voice"

    voice_task_id = request.metadata["voice_task_id"]

    assert agent.run_ids[0] is not None
    assert agent.run_ids[0].startswith("voice-")
    assert agent.task_ids[0] == voice_task_id
