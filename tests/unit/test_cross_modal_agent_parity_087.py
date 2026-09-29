"""0.8.7.2 Text/Streaming unified action parity regression tests."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from osa.core import Agent
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import ToolRegistry, ToolResult


@dataclass
class RecordingTool:
    """Deterministic tool used to compare text and streaming execution."""

    name: str = "parity_tool"
    description: str = "Return the supplied value."
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                },
            },
            "required": ["value"],
        },
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    def execute(
        self,
        arguments: dict[str, Any],
    ) -> ToolResult:
        self.calls.append(dict(arguments))

        value = arguments.get("value")

        if not isinstance(value, str):
            return ToolResult(
                success=False,
                error="value must be a string",
            )

        return ToolResult(
            success=True,
            output=value,
        )


class TextParityModel(ModelInterface):
    """Text model using one deterministic tool call."""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "text-parity-test"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls.append(request)

        if len(self.calls) == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                finish_reason="tool_calls",
                tool_calls=(
                    ToolCall(
                        id="parity-call",
                        name="parity_tool",
                        arguments={
                            "value": "same-action-result",
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
        assert tool_messages[0].tool_call_id == "parity-call"
        assert tool_messages[0].content == "same-action-result"

        return ModelResponse(
            content="same final response",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


class StreamingParityModel(ModelInterface):
    """Streaming model using the same semantic tool call."""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "streaming-parity-test"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "generate() should not be used by the streaming parity test."
        )

    def generate_stream_events(
        self,
        request: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        self.calls.append(request)

        if len(self.calls) == 1:
            yield ModelStreamEvent(
                tool_calls=(
                    ToolCall(
                        id="parity-call",
                        name="parity_tool",
                        arguments={
                            "value": "same-action-result",
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
        assert tool_messages[0].tool_call_id == "parity-call"
        assert tool_messages[0].content == "same-action-result"

        yield ModelStreamEvent(
            content="same final response"
        )

    def health_check(self) -> bool:
        return True


def _build_agent(
    model: ModelInterface,
    tool: RecordingTool,
) -> Agent:
    registry = ToolRegistry()
    registry.register(tool)

    return Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "parity_tool": PermissionLevel.ALLOW,
            }
        ),
    )


def test_text_and_streaming_have_matching_tool_execution_semantics() -> None:
    text_tool = RecordingTool()
    text_model = TextParityModel()
    text_agent = _build_agent(
        text_model,
        text_tool,
    )

    text_response = text_agent.chat(
        "run the parity tool"
    )

    stream_tool = RecordingTool()
    stream_model = StreamingParityModel()
    stream_agent = _build_agent(
        stream_model,
        stream_tool,
    )

    stream_chunks = list(
        stream_agent.chat_stream(
            "run the parity tool"
        )
    )

    assert text_response.content == "same final response"
    assert stream_chunks == ["same final response"]

    assert text_tool.calls == [
        {
            "value": "same-action-result",
        }
    ]

    assert stream_tool.calls == [
        {
            "value": "same-action-result",
        }
    ]

    assert text_tool.calls == stream_tool.calls


def test_text_and_streaming_share_the_same_tool_call_contract() -> None:
    text_model = TextParityModel()
    text_tool = RecordingTool()
    text_agent = _build_agent(
        text_model,
        text_tool,
    )

    text_agent.chat(
        "run the parity tool"
    )

    stream_model = StreamingParityModel()
    stream_tool = RecordingTool()
    stream_agent = _build_agent(
        stream_model,
        stream_tool,
    )

    list(
        stream_agent.chat_stream(
            "run the parity tool"
        )
    )

    text_first_request = text_model.calls[0]
    stream_first_request = stream_model.calls[0]

    assert text_first_request.tools == stream_first_request.tools

    assert text_first_request.tools[0].name == "parity_tool"
    assert stream_first_request.tools[0].name == "parity_tool"
