"""0.8.7.6 Cross-modal observability correlation regression tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Iterator

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_voice_action import (
    AgentVoiceActionAdapter,
)
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry, ToolResult
from osa.utils import EventLogger


class RecordingTool:
    """Deterministic tool used by text and streaming Agent paths."""

    name = "parity_tool"
    description = "Return a deterministic result."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(
        self,
        arguments,
    ) -> ToolResult:
        self.calls.append(dict(arguments))

        return ToolResult(
            success=True,
            output="42",
        )


class TextObservabilityModel(ModelInterface):
    """Text model that requests one deterministic tool."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "text-observability-parity"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="text-087",
                        name="parity_tool",
                        arguments={
                            "secret": "do-not-log",
                        },
                    ),
                ),
            )

        return ModelResponse(
            content="42",
            model_name=self.model_name,
            finish_reason="stop",
        )

    def health_check(self) -> bool:
        return True


class StreamingObservabilityModel(ModelInterface):
    """Streaming model that requests the same deterministic tool."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "streaming-observability-parity"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Streaming path must not call generate()."
        )

    def generate_stream_events(
        self,
        request: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        self.calls += 1

        if self.calls == 1:
            yield ModelStreamEvent(
                tool_calls=(
                    ToolCall(
                        id="stream-087",
                        name="parity_tool",
                        arguments={
                            "secret": "do-not-log",
                        },
                    ),
                )
            )
            return

        yield ModelStreamEvent(
            content="42"
        )

    def health_check(self) -> bool:
        return True


@dataclass
class FixedVoiceResolver:
    """Resolve one voice command into the standard TOOL request."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        assert command == "run voice parity"

        return ActionRequest(
            kind=ActionKind.TOOL,
            name="parity_tool",
            arguments={
                "secret": "do-not-log",
            },
            request_id="voice-087",
            metadata={
                "source": "voice",
                "voice_task_id": "voice-task-087",
            },
        )


class RecordingRecoveryExecutor:
    """Return successful ActionResults through the recovery boundary."""

    max_attempts = 2

    def __init__(self) -> None:
        self.calls: list[tuple[ActionRequest, str | None, str | None]] = []

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
            )
        )


def _build_agent(
    model: ModelInterface,
    *,
    logger: EventLogger,
    recovery_executor: RecordingRecoveryExecutor,
) -> Agent:
    registry = ToolRegistry()

    registry.register(
        RecordingTool()
    )

    from osa.core.agent_recovery import AgentRecoveryIntegration

    return Agent(
        model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "parity_tool": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            recovery_executor
        ),
        event_logger=logger,
    )


def _action_records(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if json.loads(line)["event"].startswith("action.")
    ]


def test_text_observability_correlation_is_stable(
    tmp_path,
) -> None:
    log_path = tmp_path / "text-events.jsonl"
    logger = EventLogger(log_path)
    recovery = RecordingRecoveryExecutor()

    agent = _build_agent(
        TextObservabilityModel(),
        logger=logger,
        recovery_executor=recovery,
    )

    response = agent.chat(
        "run parity"
    )

    assert response.content == "42"

    events = _action_records(log_path)

    assert [
        event["event"]
        for event in events
    ] == [
        "action.requested",
        "action.recovery.started",
        "action.recovery.completed",
        "action.completed",
    ]

    request_ids = {
        event["data"]["request_id"]
        for event in events
    }

    assert request_ids == {"text-087"}

    for event in events:
        data = event["data"]
        assert data["action_kind"] == "tool"
        assert data["action_name"] == "parity_tool"

    assert events[0]["data"]["round"] == 1
    assert events[-1]["data"]["round"] == 1

    serialized = log_path.read_text(
        encoding="utf-8"
    )

    assert "do-not-log" not in serialized
    assert "secret" not in serialized
    assert "arguments" not in serialized


def test_streaming_observability_correlation_is_stable(
    tmp_path,
) -> None:
    log_path = tmp_path / "stream-events.jsonl"
    logger = EventLogger(log_path)
    recovery = RecordingRecoveryExecutor()

    agent = _build_agent(
        StreamingObservabilityModel(),
        logger=logger,
        recovery_executor=recovery,
    )

    chunks = list(
        agent.chat_stream(
            "run parity"
        )
    )

    assert chunks == ["42"]

    events = _action_records(log_path)

    assert [
        event["event"]
        for event in events
    ] == [
        "action.requested",
        "action.recovery.started",
        "action.recovery.completed",
        "action.completed",
    ]

    assert {
        event["data"]["request_id"]
        for event in events
    } == {"stream-087"}

    assert events[0]["data"]["round"] == 1
    assert events[-1]["data"]["round"] == 1


def test_voice_observability_correlation_is_stable(
    tmp_path,
) -> None:
    log_path = tmp_path / "voice-events.jsonl"
    logger = EventLogger(log_path)
    recovery = RecordingRecoveryExecutor()

    agent = _build_agent(
        ModelResponseOnlyModel(),
        logger=logger,
        recovery_executor=recovery,
    )

    adapter = AgentVoiceActionAdapter(
        agent,
        FixedVoiceResolver(),
    )

    response = adapter.chat(
        "run voice parity"
    )

    assert response.content == "42"

    events = _action_records(log_path)

    assert [
        event["event"]
        for event in events
    ] == [
        "action.requested",
        "action.recovery.started",
        "action.recovery.completed",
        "action.completed",
    ]

    assert {
        event["data"]["request_id"]
        for event in events
    } == {"voice-087"}

    assert all(
        event["data"]["action_kind"] == "tool"
        for event in events
    )

    assert all(
        event["data"]["action_name"] == "parity_tool"
        for event in events
    )

    assert all(
        "round" not in event["data"]
        for event in events
    )

    serialized = log_path.read_text(
        encoding="utf-8"
    )

    assert "do-not-log" not in serialized
    assert "secret" not in serialized
    assert "arguments" not in serialized


class ModelResponseOnlyModel(ModelInterface):
    """Unused model surface required by Agent."""

    @property
    def model_name(self) -> str:
        return "voice-observability-agent"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Voice adapter should use execute_action_with_recovery()."
        )

    def health_check(self) -> bool:
        return True
