"""0.8.8.3 Text/Streaming integration with the unified action core."""

from __future__ import annotations

from collections.abc import Iterator

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_recovery import AgentRecoveryIntegration
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.recovery_contracts import RecoveryResult


class RecordingRecoveryExecutor:
    """Record every ActionRequest entering unified recovery."""

    max_attempts = 2

    def __init__(self) -> None:
        self.calls: list[ActionRequest] = []

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        self.calls.append(request)

        return RecoveryResult.succeeded(
            result=ActionResult.succeeded(
                request.request_id,
                output="42",
            ),
        )


class TextRoundModel(ModelInterface):
    """Model used to verify the regular chat round path."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "core-text-round-088"

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
                        id="text-core-088",
                        name="demo",
                        arguments={
                            "value": 42,
                        },
                    ),
                ),
            )

        tool_messages = [
            message
            for message in request.messages
            if message.role == "tool"
        ]

        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "text-core-088"
        assert tool_messages[0].content == "42"

        return ModelResponse(
            content="42",
            model_name=self.model_name,
            finish_reason="stop",
        )

    def health_check(self) -> bool:
        return True


class StreamingRoundModel(ModelInterface):
    """Model used to verify the streaming round path."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "core-stream-round-088"

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
                        id="stream-core-088",
                        name="demo",
                        arguments={
                            "value": 42,
                        },
                    ),
                )
            )
            return

        tool_messages = [
            message
            for message in request.messages
            if message.role == "tool"
        ]

        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "stream-core-088"
        assert tool_messages[0].content == "42"

        yield ModelStreamEvent(
            content="42"
        )

    def health_check(self) -> bool:
        return True


def _build_agent(
    model: ModelInterface,
    executor: RecordingRecoveryExecutor,
) -> Agent:
    registry = __import__(
        "osa.tools",
        fromlist=["ToolRegistry"],
    ).ToolRegistry()

    class DemoTool:
        name = "demo"
        description = "Demo tool"
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
                "Tool backend must not be called directly when recovery is configured."
            )

    registry.register(DemoTool())

    return Agent(
        model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "demo": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=AgentRecoveryIntegration(
            executor
        ),
    )


def test_text_round_enters_unified_action_core_once() -> None:
    executor = RecordingRecoveryExecutor()

    agent = _build_agent(
        TextRoundModel(),
        executor,
    )

    response = agent.chat(
        "run demo"
    )

    assert response.content == "42"
    assert len(executor.calls) == 1

    request = executor.calls[0]

    assert request.request_id == "text-core-088"
    assert request.kind is ActionKind.TOOL
    assert request.name == "demo"
    assert request.arguments == {
        "value": 42,
    }


def test_streaming_round_enters_unified_action_core_once() -> None:
    executor = RecordingRecoveryExecutor()

    agent = _build_agent(
        StreamingRoundModel(),
        executor,
    )

    chunks = list(
        agent.chat_stream(
            "run demo"
        )
    )

    assert chunks == ["42"]
    assert len(executor.calls) == 1

    request = executor.calls[0]

    assert request.request_id == "stream-core-088"
    assert request.kind is ActionKind.TOOL
    assert request.name == "demo"
    assert request.arguments == {
        "value": 42,
    }


def test_text_and_streaming_use_identical_action_payloads() -> None:
    text_executor = RecordingRecoveryExecutor()
    text_agent = _build_agent(
        TextRoundModel(),
        text_executor,
    )

    text_agent.chat(
        "run demo"
    )

    stream_executor = RecordingRecoveryExecutor()
    stream_agent = _build_agent(
        StreamingRoundModel(),
        stream_executor,
    )

    list(
        stream_agent.chat_stream(
            "run demo"
        )
    )

    text_request = text_executor.calls[0]
    stream_request = stream_executor.calls[0]

    assert text_request.kind is stream_request.kind is ActionKind.TOOL
    assert text_request.name == stream_request.name == "demo"
    assert text_request.arguments == stream_request.arguments == {
        "value": 42,
    }

    assert text_request.metadata["round"] == 1
    assert stream_request.metadata["round"] == 1

    assert text_request.request_id == "text-core-088"
    assert stream_request.request_id == "stream-core-088"
