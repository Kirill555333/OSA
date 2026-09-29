"""0.8.9.7 Canonical execution-context observability tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.core.agent_voice_action import AgentVoiceActionAdapter
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult
from osa.tools import ToolRegistry
from osa.utils import EventLogger


class UnusedTool:
    """Registered tool that must never be called directly."""

    name = "demo"
    description = "Observability context test tool"
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments):
        raise AssertionError(
            "Tool backend must not be called directly."
        )


class RecordingRecoveryExecutor:
    """Return one successful ActionResult through recovery."""

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


class TextModel(ModelInterface):
    """Text model producing one tool call."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "context-text-089"

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
                        id="context-text-089",
                        name="demo",
                        arguments={},
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


class StreamingModel(ModelInterface):
    """Streaming model producing one tool call."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "context-stream-089"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Streaming test must not call generate()."
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
                        id="context-stream-089",
                        name="demo",
                        arguments={},
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
class VoiceResolver:
    """Resolve one voice command into a canonical action."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        assert command == "run context observability"

        return ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={},
            request_id="context-voice-089",
            metadata={
                "source": "voice",
                "voice_task_id": "voice-task-observability-089",
            },
        )


class UnusedModel(ModelInterface):
    """Model surface required by Agent but unused by Voice."""

    @property
    def model_name(self) -> str:
        return "unused-context-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Voice path must not call Agent model."
        )

    def health_check(self) -> bool:
        return True


def _build_agent(
    model: ModelInterface,
    recovery: RecordingRecoveryExecutor,
    logger: EventLogger,
) -> Agent:
    registry = ToolRegistry()
    registry.register(
        UnusedTool()
    )

    return Agent(
        model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            recovery
        ),
        event_logger=logger,
    )


def _read_action_events(path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8",
        ).splitlines()
        if json.loads(line)["event"].startswith("action.")
    ]


def _assert_correlation(
    events: list[dict],
    *,
    request_id: str,
    run_id: str | None,
    task_id: str | None,
    round_number: int | None,
    source: str,
) -> None:
    assert events

    for record in events:
        data = record["data"]

        assert data["request_id"] == request_id
        assert data["action_kind"] == "tool"
        assert data["action_name"] == "demo"

        if run_id is None:
            assert "run_id" not in data
        else:
            assert data["run_id"] == run_id

        if task_id is None:
            assert "task_id" not in data
        else:
            assert data["task_id"] == task_id

        if round_number is None:
            assert "round" not in data
        else:
            assert data["round"] == round_number

        assert data["source"] == source

        assert "metadata" not in data
        assert "arguments" not in data
        assert "argument_keys" not in data


def test_direct_action_events_use_canonical_context(
    tmp_path,
) -> None:
    log_path = tmp_path / "direct.jsonl"
    logger = EventLogger(log_path)

    recovery = RecordingRecoveryExecutor()
    agent = _build_agent(
        UnusedModel(),
        recovery,
        logger,
    )

    result = agent.execute_action_with_recovery(
        ActionRequest(
            kind=ActionKind.TOOL,
            name="demo",
            arguments={
                "secret": "must-not-log",
            },
            request_id="context-direct-089",
            metadata={
                "source": "agent",
                "run_id": "metadata-run",
                "task_id": "metadata-task",
                "round": 7,
            },
        ),
        run_id="explicit-run-089",
        task_id="explicit-task-089",
    )

    assert result.success is True

    events = _read_action_events(log_path)

    _assert_correlation(
        events,
        request_id="context-direct-089",
        run_id="explicit-run-089",
        task_id="explicit-task-089",
        round_number=7,
        source="agent",
    )

    serialized = log_path.read_text(
        encoding="utf-8",
    )

    assert "must-not-log" not in serialized
    assert "metadata-run" not in serialized
    assert "metadata-task" not in serialized


def test_text_events_use_round_context(
    tmp_path,
) -> None:
    log_path = tmp_path / "text.jsonl"
    logger = EventLogger(log_path)

    agent = _build_agent(
        TextModel(),
        RecordingRecoveryExecutor(),
        logger,
    )

    response = agent.chat(
        "run context observability"
    )

    assert response.content == "42"

    events = _read_action_events(log_path)

    _assert_correlation(
        events,
        request_id="context-text-089",
        run_id=None,
        task_id=None,
        round_number=1,
        source="agent",
    )


def test_streaming_events_use_round_context(
    tmp_path,
) -> None:
    log_path = tmp_path / "stream.jsonl"
    logger = EventLogger(log_path)

    agent = _build_agent(
        StreamingModel(),
        RecordingRecoveryExecutor(),
        logger,
    )

    chunks = list(
        agent.chat_stream(
            "run context observability"
        )
    )

    assert chunks == ["42"]

    events = _read_action_events(log_path)

    _assert_correlation(
        events,
        request_id="context-stream-089",
        run_id=None,
        task_id=None,
        round_number=1,
        source="agent",
    )


def test_voice_events_use_voice_context(
    tmp_path,
) -> None:
    log_path = tmp_path / "voice.jsonl"
    logger = EventLogger(log_path)

    recovery = RecordingRecoveryExecutor()
    agent = _build_agent(
        UnusedModel(),
        recovery,
        logger,
    )

    adapter = AgentVoiceActionAdapter(
        agent,
        VoiceResolver(),
    )

    response = adapter.chat(
        "run context observability"
    )

    assert response.content == "42"

    events = _read_action_events(log_path)

    assert len(events) == 4

    run_id = events[0]["data"].get("run_id")

    assert isinstance(run_id, str)
    assert run_id.startswith("voice-")

    _assert_correlation(
        events,
        request_id="context-voice-089",
        run_id=run_id,
        task_id="voice-task-observability-089",
        round_number=None,
        source="voice",
    )
