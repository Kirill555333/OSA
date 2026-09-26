import pytest

from osa.core import Agent, AgentError
from osa.models import (
    ChatMessage,
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
)


class FakeModel(ModelInterface):
    """Predictable model used for Agent Core tests."""

    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []
        self.fail = False

    @property
    def model_name(self) -> str:
        return "fake"

    def generate(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)

        if self.fail:
            raise ModelError("fake failure")

        return ModelResponse(
            content=f"Echo: {request.messages[-1].content}",
            model_name=self.model_name,
        )

    def health_check(self) -> bool:
        return True


def test_agent_keeps_conversation_context() -> None:
    model = FakeModel()
    agent = Agent(
        model,
        system_prompt="You are OSA.",
    )

    first = agent.chat("Hello")
    second = agent.chat("How are you?")

    assert first.content == "Echo: Hello"
    assert second.content == "Echo: How are you?"

    messages = agent.context.messages()

    assert messages == (
        ChatMessage(role="system", content="You are OSA."),
        ChatMessage(role="user", content="Hello"),
        ChatMessage(role="assistant", content="Echo: Hello"),
        ChatMessage(role="user", content="How are you?"),
        ChatMessage(role="assistant", content="Echo: How are you?"),
    )


def test_agent_rejects_empty_input() -> None:
    agent = Agent(FakeModel())

    with pytest.raises(ValueError):
        agent.chat("   ")


def test_agent_removes_failed_user_message() -> None:
    model = FakeModel()
    agent = Agent(model)
    model.fail = True

    with pytest.raises(AgentError):
        agent.chat("This will fail")

    assert len(agent.context) == 0


def test_agent_reset_preserves_system_prompt() -> None:
    agent = Agent(
        FakeModel(),
        system_prompt="You are OSA.",
    )

    agent.chat("Hello")
    agent.reset()

    assert agent.context.messages() == (
        ChatMessage(
            role="system",
            content="You are OSA.",
        ),
    )
