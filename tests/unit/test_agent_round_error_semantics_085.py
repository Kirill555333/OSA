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
from osa.tools import ToolRegistry


class DeniedChatModel(ModelInterface):
    """Model that requests one denied tool, then finishes."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "denied-chat-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                tool_calls=(
                    ToolCall(
                        id="denied-1",
                        name="calculator",
                        arguments={
                            "expression": "1 + 1",
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
        assert tool_messages[0].tool_call_id == "denied-1"
        assert "Tool execution denied" in tool_messages[0].content

        return ModelResponse(
            content="Запрос отклонён.",
            model_name=self.model_name,
        )

    def generate_stream_events(
        self,
        request: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        raise AssertionError(
            "Streaming is not used by this model."
        )

    def health_check(self) -> bool:
        return True


class DeniedStreamModel(ModelInterface):
    """Streaming model that requests one denied tool, then finishes."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "denied-stream-model"

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
        self.calls += 1

        if self.calls == 1:
            yield ModelStreamEvent(
                tool_calls=(
                    ToolCall(
                        id="denied-stream-1",
                        name="calculator",
                        arguments={
                            "expression": "1 + 1",
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
        assert tool_messages[0].tool_call_id == "denied-stream-1"
        assert "Tool execution denied" in tool_messages[0].content

        yield ModelStreamEvent(
            content="Запрос отклонён."
        )

    def health_check(self) -> bool:
        return True


class ExplodingTool:
    name = "exploding"
    description = "Tool that raises an exception."
    parameters = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    def execute(self, arguments):
        raise RuntimeError("backend exploded")


class ExplodingModel(ModelInterface):
    """Model that requests a tool which raises."""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "exploding-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.calls += 1

        if self.calls == 1:
            return ModelResponse(
                content="",
                model_name=self.model_name,
                tool_calls=(
                    ToolCall(
                        id="explode-1",
                        name="exploding",
                        arguments={},
                    ),
                ),
            )

        assert request.messages[-1].role == "tool"
        assert "backend exploded" in request.messages[-1].content

        return ModelResponse(
            content="Ошибка обработана.",
            model_name=self.model_name,
        )

    def generate_stream_events(
        self,
        request: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        if self.calls == 0:
            self.calls += 1

            yield ModelStreamEvent(
                tool_calls=(
                    ToolCall(
                        id="explode-stream-1",
                        name="exploding",
                        arguments={},
                    ),
                )
            )
            return

        assert request.messages[-1].role == "tool"
        assert "backend exploded" in request.messages[-1].content

        yield ModelStreamEvent(
            content="Ошибка обработана."
        )

    def health_check(self) -> bool:
        return True


class DummyCalculator:
    name = "calculator"
    description = "Dummy calculator."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"},
        },
        "required": ["expression"],
        "additionalProperties": False,
    }

    def execute(self, arguments):
        raise AssertionError(
            "Denied calculator must never execute."
        )


def test_chat_continues_after_tool_denial() -> None:
    model = DeniedChatModel()

    registry = ToolRegistry()
    registry.register(DummyCalculator())

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.DENY,
            }
        ),
    )

    response = agent.chat("run calculator")

    assert response.content == "Запрос отклонён."
    assert model.calls == 2

    tool_messages = [
        message
        for message in agent.context.messages()
        if message.role == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "denied-1"
    assert "Tool execution denied" in tool_messages[0].content


def test_chat_stream_continues_after_tool_denial() -> None:
    model = DeniedStreamModel()

    registry = ToolRegistry()
    registry.register(DummyCalculator())

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "calculator": PermissionLevel.DENY,
            }
        ),
    )

    chunks = list(
        agent.chat_stream("run calculator")
    )

    assert chunks == [
        "Запрос отклонён."
    ]
    assert model.calls == 2

    tool_messages = [
        message
        for message in agent.context.messages()
        if message.role == "tool"
    ]

    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "denied-stream-1"
    assert "Tool execution denied" in tool_messages[0].content


def test_chat_preserves_tool_exception_for_next_model_round() -> None:
    model = ExplodingModel()

    registry = ToolRegistry()
    registry.register(ExplodingTool())

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "exploding": PermissionLevel.ALLOW,
            }
        ),
    )

    response = agent.chat("run exploding")

    assert response.content == "Ошибка обработана."
    assert model.calls == 2

    tool_messages = [
        message
        for message in agent.context.messages()
        if message.role == "tool"
    ]

    assert len(tool_messages) == 1
    assert "backend exploded" in tool_messages[0].content


def test_chat_stream_preserves_tool_exception_for_next_model_round() -> None:
    model = ExplodingModel()

    registry = ToolRegistry()
    registry.register(ExplodingTool())

    agent = Agent(
        model=model,
        tool_registry=registry,
        permission_policy=PermissionPolicy(
            {
                "exploding": PermissionLevel.ALLOW,
            }
        ),
    )

    chunks = list(
        agent.chat_stream("run exploding")
    )

    assert chunks == [
        "Ошибка обработана."
    ]

    tool_messages = [
        message
        for message in agent.context.messages()
        if message.role == "tool"
    ]

    assert len(tool_messages) == 1
    assert "backend exploded" in tool_messages[0].content
