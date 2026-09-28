from __future__ import annotations

import pytest

from osa.core.agent import Agent, AgentError
from osa.models.interface import (
    ChatMessage,
    ModelConnectionError,
    ModelRequest,
    ModelResponse,
    ModelInterface,
)


class FakeStreamingModel(ModelInterface):
    """Simple model double for streaming tests."""

    def __init__(
        self,
        chunks: tuple[str, ...] = ("Hel", "lo", "!"),
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._error = error
        self.last_request: ModelRequest | None = None

    @property
    def model_name(self) -> str:
        return "fake-streaming"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        return ModelResponse(
            content="fallback",
            finish_reason="stop",
            tool_calls=(),
        )

    def generate_stream(
        self,
        request: ModelRequest,
    ):
        self.last_request = request

        if self._error is not None:
            raise self._error

        yield from self._chunks

    def health_check(self) -> bool:
        return True


def test_chat_stream_yields_chunks_and_commits_response() -> None:
    model = FakeStreamingModel(
        chunks=("Hello", " ", "OSA", "!"),
    )
    agent = Agent(model)

    chunks = list(
        agent.chat_stream("Say hello")
    )

    assert chunks == [
        "Hello",
        " ",
        "OSA",
        "!",
    ]

    messages = agent.context.messages()

    assert messages[0] == ChatMessage(
        role="user",
        content="Say hello",
    )
    assert messages[1] == ChatMessage(
        role="assistant",
        content="Hello OSA!",
    )


def test_chat_stream_does_not_send_tools() -> None:
    model = FakeStreamingModel()
    agent = Agent(model)

    list(
        agent.chat_stream("Hello")
    )

    assert model.last_request is not None
    assert model.last_request.tools == ()


def test_chat_stream_restores_context_when_model_fails() -> None:
    model = FakeStreamingModel(
        error=ModelConnectionError(
            "connection lost"
        )
    )
    agent = Agent(model)

    with pytest.raises(AgentError, match="Model streaming failed"):
        list(
            agent.chat_stream("Hello")
        )

    assert agent.context.messages() == ()
