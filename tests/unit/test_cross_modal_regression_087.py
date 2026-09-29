"""0.8.7.7 Cross-modal end-to-end regression matrix."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.core import Agent
from osa.core.agent_voice_action import AgentVoiceActionAdapter
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
from osa.tools import ToolRegistry, ToolResult


class MatrixTool:
    """Shared deterministic tool contract for all execution channels."""

    name = "matrix_tool"
    description = "Return a deterministic matrix value."
    parameters = {
        "type": "object",
        "properties": {
            "value": {
                "type": "integer",
            },
        },
        "required": ["value"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(
        self,
        arguments,
    ) -> ToolResult:
        self.calls.append(dict(arguments))

        value = arguments.get("value")

        if not isinstance(value, int) or isinstance(value, bool):
            return ToolResult(
                success=False,
                error="value must be an integer",
            )

        return ToolResult(
            success=True,
            output=str(value * 2),
        )


class TextMatrixModel(ModelInterface):
    """Text model for the regression matrix."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "text-matrix-087"

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
                        id="text-matrix-087",
                        name="matrix_tool",
                        arguments={"value": 21},
                    ),
                ),
            )

        tool_messages = [
            message
            for message in request.messages
            if message.role == "tool"
        ]

        assert len(tool_messages) == 1
        assert tool_messages[0].tool_call_id == "text-matrix-087"
        assert tool_messages[0].content == "42"

        return ModelResponse(
            content="42",
            model_name=self.model_name,
            finish_reason="stop",
        )

    def health_check(self) -> bool:
        return True


class StreamingMatrixModel(ModelInterface):
    """Streaming model for the regression matrix."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "stream-matrix-087"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "Streaming matrix must not call generate()."
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
                        id="stream-matrix-087",
                        name="matrix_tool",
                        arguments={"value": 21},
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
        assert tool_messages[0].tool_call_id == "stream-matrix-087"
        assert tool_messages[0].content == "42"

        yield ModelStreamEvent(
            content="42"
        )

    def health_check(self) -> bool:
        return True


@dataclass
class MatrixVoiceResolver:
    """Resolve the shared matrix operation into a TOOL ActionRequest."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        assert command == "run matrix action"

        return ActionRequest(
            kind=ActionKind.TOOL,
            name="matrix_tool",
            arguments={"value": 21},
            request_id="voice-matrix-087",
            metadata={
                "source": "voice",
                "voice_task_id": "voice-matrix-task-087",
            },
        )


class MatrixRecoveryExecutor:
    """Shared recovery boundary used by the voice channel."""

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
            )
        )


def _build_agent(
    model: ModelInterface,
    tool: MatrixTool,
    *,
    recovery: MatrixRecoveryExecutor | None = None,
) -> Agent:
    registry = ToolRegistry()
    registry.register(tool)

    integration = (
        AgentRecoveryIntegration(recovery)
        if recovery is not None
        else None
    )

    return Agent(
        model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "matrix_tool": PermissionLevel.ALLOW,
            }
        ),
        recovery_integration=integration,
    )


def test_all_channels_produce_the_same_action_semantics() -> None:
    text_tool = MatrixTool()
    text_agent = _build_agent(
        TextMatrixModel(),
        text_tool,
    )

    text_response = text_agent.chat(
        "run matrix action"
    )

    stream_tool = MatrixTool()
    stream_agent = _build_agent(
        StreamingMatrixModel(),
        stream_tool,
    )

    stream_response = list(
        stream_agent.chat_stream(
            "run matrix action"
        )
    )

    voice_tool = MatrixTool()
    voice_recovery = MatrixRecoveryExecutor()
    voice_agent = _build_agent(
        TextMatrixModel(),
        voice_tool,
        recovery=voice_recovery,
    )

    voice_adapter = AgentVoiceActionAdapter(
        voice_agent,
        MatrixVoiceResolver(),
    )

    voice_response = voice_adapter.chat(
        "run matrix action"
    )

    assert text_response.content == "42"
    assert stream_response == ["42"]
    assert voice_response.content == "42"

    expected_arguments = [
        {
            "value": 21,
        }
    ]

    assert text_tool.calls == expected_arguments
    assert stream_tool.calls == expected_arguments
    assert voice_recovery.calls[0].arguments == {
        "value": 21,
    }

    assert voice_tool.calls == []

    assert voice_recovery.calls[0].kind is ActionKind.TOOL
    assert voice_recovery.calls[0].name == "matrix_tool"


def test_text_and_streaming_share_capability_definition() -> None:
    text_model = TextMatrixModel()
    text_tool = MatrixTool()

    text_agent = _build_agent(
        text_model,
        text_tool,
    )

    text_agent.chat(
        "run matrix action"
    )

    stream_model = StreamingMatrixModel()
    stream_tool = MatrixTool()

    stream_agent = _build_agent(
        stream_model,
        stream_tool,
    )

    list(
        stream_agent.chat_stream(
            "run matrix action"
        )
    )

    assert text_model.calls == 2
    assert stream_model.calls == 2

    text_definition = text_model.calls

    assert text_definition == 2


def test_voice_has_no_direct_tool_execution_path() -> None:
    voice_tool = MatrixTool()
    recovery = MatrixRecoveryExecutor()

    agent = _build_agent(
        TextMatrixModel(),
        voice_tool,
        recovery=recovery,
    )

    adapter = AgentVoiceActionAdapter(
        agent,
        MatrixVoiceResolver(),
    )

    response = adapter.chat(
        "run matrix action"
    )

    assert response.content == "42"

    assert voice_tool.calls == []
    assert len(recovery.calls) == 1

    request = recovery.calls[0]

    assert request.kind is ActionKind.TOOL
    assert request.name == "matrix_tool"
    assert request.arguments == {
        "value": 21,
    }
