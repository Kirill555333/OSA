from __future__ import annotations

from collections.abc import Iterator

from osa.core import Agent
from osa.models import (
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCall,
)
from osa.permissions import PermissionLevel, PermissionPolicy
from osa.tools import CalculatorTool, ToolRegistry


class StreamingToolModel(ModelInterface):
    """Fake streaming model that emits two tool calls, then final text."""

    def __init__(self) -> None:
        self.stream_calls: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "streaming-tool-test"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        raise AssertionError(
            "generate() should not be used by this streaming test."
        )

    def generate_stream_events(
        self,
        request: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        self.stream_calls.append(request)

        if len(self.stream_calls) == 1:
            yield ModelStreamEvent(
                tool_calls=(
                    ToolCall(
                        id="stream_call_a",
                        name="calculator",
                        arguments={
                            "expression": "7 + 8",
                        },
                    ),
                    ToolCall(
                        id="stream_call_b",
                        name="calculator",
                        arguments={
                            "expression": "6 * 9",
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

        assert len(tool_messages) == 2
        assert tool_messages[0].tool_call_id == "stream_call_a"
        assert tool_messages[0].content == "15"
        assert tool_messages[1].tool_call_id == "stream_call_b"
        assert tool_messages[1].content == "54"

        yield ModelStreamEvent(
            content="Ответ: 15 и 54."
        )

    def health_check(self) -> bool:
        return True


def test_agent_streaming_executes_all_tool_calls_in_one_round() -> None:
    model = StreamingToolModel()

    registry = ToolRegistry()
    registry.register(CalculatorTool())

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.ALLOW,
            }
        ),
    )

    chunks = list(
        agent.chat_stream(
            "Посчитай 7 + 8 и 6 * 9."
        )
    )

    assert chunks == [
        "Ответ: 15 и 54."
    ]
    assert len(model.stream_calls) == 2
