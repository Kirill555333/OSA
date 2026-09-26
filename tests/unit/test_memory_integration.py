from pathlib import Path

from osa.core import Agent
from osa.memory.integration import MemoryIntegration
from osa.memory.long_term import LongTermMemory
from osa.memory.retrieval import MemoryRetriever
from osa.models import (
    ChatMessage,
    ModelInterface,
    ModelRequest,
    ModelResponse,
)


class RecordingModel(ModelInterface):
    """Fake model that records requests for integration tests."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        self.requests.append(request)

        return ModelResponse(
            content="Python is the primary language.",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


def create_integration(
    tmp_path: Path,
) -> MemoryIntegration:
    memory = LongTermMemory(
        tmp_path / "memory.db"
    )

    retriever = MemoryRetriever(
        memory
    )

    return MemoryIntegration(
        retriever
    )


def test_natural_language_query_finds_relevant_memory(
    tmp_path: Path,
) -> None:
    memory = LongTermMemory(
        tmp_path / "memory.db"
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
        category="general",
    )

    integration = MemoryIntegration(
        MemoryRetriever(memory)
    )

    result = integration.retrieve(
        "Что ты знаешь об основном языке разработки OSA?"
    )

    assert len(result.memories) == 1
    assert (
        result.memories[0].memory.content
        == "Основной язык разработки OSA — Python."
    )
    assert result.prompt is not None


def test_irrelevant_query_returns_no_memory(
    tmp_path: Path,
) -> None:
    integration = create_integration(
        tmp_path
    )

    result = integration.retrieve(
        "Какая сегодня погода?"
    )

    assert result.memories == ()
    assert result.prompt is None


def test_memory_context_is_transient(
    tmp_path: Path,
) -> None:
    memory = LongTermMemory(
        tmp_path / "memory.db"
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
        category="general",
    )

    integration = MemoryIntegration(
        MemoryRetriever(memory)
    )

    model = RecordingModel()

    agent = Agent(
        model=model,
        system_prompt="You are OSA.",
        memory_integration=integration,
    )

    response = agent.chat(
        "Что ты знаешь об основном языке разработки OSA?"
    )

    assert response.content

    assert len(model.requests) == 1

    request_messages = model.requests[0].messages

    assert any(
        message.role == "system"
        and "Основной язык разработки OSA" in message.content
        for message in request_messages
    )

    assert all(
        "Основной язык разработки OSA" not in message.content
        for message in agent.context.messages()
        if message.role != "system"
    )


def test_without_memory_integration_behavior_is_unchanged() -> None:
    model = RecordingModel()

    agent = Agent(
        model=model,
        system_prompt="You are OSA.",
    )

    agent.chat("Hello")

    assert len(model.requests) == 1

    assert model.requests[0].messages == (
        ChatMessage(
            role="system",
            content="You are OSA.",
        ),
        ChatMessage(
            role="user",
            content="Hello",
        ),
    )
