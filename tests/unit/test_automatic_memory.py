from pathlib import Path

import pytest

from osa.core import Agent, AgentError
from osa.memory import (
    AutomaticMemory,
    LongTermMemory,
    MemoryRetriever,
)
from osa.models import (
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
)


class FakeModel(ModelInterface):
    """Minimal model for automatic-memory integration tests."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    @property
    def model_name(self) -> str:
        return "test-model"

    def generate(
        self,
        request: ModelRequest,
    ) -> ModelResponse:
        if self.fail:
            raise ModelError(
                "simulated model failure"
            )

        return ModelResponse(
            content="Acknowledged.",
            model_name=self.model_name,
            finish_reason="stop",
            usage={},
            tool_calls=(),
        )

    def health_check(self) -> bool:
        return True


def create_memory(
    tmp_path: Path,
) -> LongTermMemory:
    return LongTermMemory(
        tmp_path / "memory.db"
    )


def test_automatic_memory_saves_stable_preference(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    result = automatic.capture(
        "Я предпочитаю Python для разработки OSA."
    )

    assert result.saved is True
    assert result.memory_id is not None

    record = memory.get(
        result.memory_id
    )

    assert record.category == "preference"
    assert record.importance == 8
    assert (
        record.content
        == "Я предпочитаю Python для разработки OSA."
    )


def test_automatic_memory_saves_project_fact(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    result = automatic.capture(
        "Основной язык разработки OSA — Python."
    )

    assert result.saved is True
    assert result.memory_id is not None

    record = memory.get(
        result.memory_id
    )

    assert record.category == "project"


def test_automatic_memory_saves_platform_fact(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    result = automatic.capture(
        "Целевая операционная система OSA — Windows."
    )

    assert result.saved is True
    assert result.memory_id is not None

    record = memory.get(
        result.memory_id
    )

    assert record.category == "platform"


def test_automatic_memory_skips_explicit_remember_request(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    result = automatic.capture(
        "Запомни, что основной язык разработки OSA — Python."
    )

    assert result.saved is False
    assert result.reason == "explicit_memory_request"
    assert memory.count() == 0


@pytest.mark.parametrize(
    "message",
    (
        "Привет, OSA!",
        "Что ты знаешь о Python?",
        "Посчитай 25 * 17.",
        "Сегодня я работаю дома.",
        "Сейчас я изучаю Python.",
    ),
)
def test_automatic_memory_skips_non_persistent_messages(
    tmp_path: Path,
    message: str,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    result = automatic.capture(message)

    assert result.saved is False
    assert memory.count() == 0


def test_automatic_memory_skips_sensitive_information(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    result = automatic.capture(
        "Мой API key это abc123."
    )

    assert result.saved is False
    assert result.reason == "sensitive_content"
    assert memory.count() == 0


def test_automatic_memory_deduplicates_content(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    first = automatic.capture(
        "Я предпочитаю Python для разработки OSA."
    )

    second = automatic.capture(
        "  Я предпочитаю Python для разработки OSA.  "
    )

    assert first.saved is True
    assert second.saved is False
    assert second.reason == "duplicate"
    assert memory.count() == 1


def test_agent_captures_automatic_memory_after_success(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    model = FakeModel()

    agent = Agent(
        model,
        system_prompt="You are OSA.",
        automatic_memory=automatic,
    )

    response = agent.chat(
        "Основной язык разработки OSA — Python."
    )

    assert response.content == "Acknowledged."
    assert memory.count() == 1

    record = memory.recent(
        limit=1
    )[0]

    assert (
        record.content
        == "Основной язык разработки OSA — Python."
    )


def test_agent_does_not_save_when_model_fails(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    agent = Agent(
        FakeModel(fail=True),
        system_prompt="You are OSA.",
        automatic_memory=automatic,
    )

    with pytest.raises(AgentError):
        agent.chat(
            "Основной язык разработки OSA — Python."
        )

    assert memory.count() == 0


def test_automatic_memory_integrates_with_retrieval(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    automatic = AutomaticMemory(memory)

    result = automatic.capture(
        "Основной язык разработки OSA — Python."
    )

    assert result.saved is True

    retriever = MemoryRetriever(
        memory
    )

    results = retriever.search(
        "основной язык разработки OSA"
    )

    assert results
    assert (
        results[0].memory.content
        == "Основной язык разработки OSA — Python."
    )
