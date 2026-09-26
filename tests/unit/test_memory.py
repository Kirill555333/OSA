from pathlib import Path

import pytest

from osa.memory import (
    LongTermMemory,
    MemoryRetriever,
)


def create_memory(tmp_path: Path) -> LongTermMemory:
    """Create a temporary memory database."""
    return LongTermMemory(
        tmp_path / "memory.db"
    )


def test_memory_database_is_created(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"

    memory = LongTermMemory(database)

    assert database.exists()
    assert memory.count() == 0


def test_memory_can_be_saved_and_loaded(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    record = memory.save(
        "Пользователь разрабатывает OSA на MacBook.",
        category="user",
        importance=8,
    )

    loaded = memory.get(record.id)

    assert loaded.content == (
        "Пользователь разрабатывает OSA на MacBook."
    )
    assert loaded.category == "user"
    assert loaded.importance == 8


def test_memory_survives_reopening_database(tmp_path: Path) -> None:
    database = tmp_path / "memory.db"

    first = LongTermMemory(database)

    record = first.save(
        "OSA uses Python.",
        category="project",
        importance=7,
    )

    second = LongTermMemory(database)

    loaded = second.get(record.id)

    assert loaded.content == "OSA uses Python."


def test_memory_search(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    memory.save(
        "OSA is being developed on a MacBook.",
        category="project",
        importance=8,
    )

    memory.save(
        "The target runtime is Windows.",
        category="platform",
        importance=6,
    )

    results = memory.search("MacBook")

    assert len(results) == 1
    assert results[0].content == (
        "OSA is being developed on a MacBook."
    )


def test_recent_memories_are_returned(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    memory.save(
        "First memory",
        importance=3,
    )

    memory.save(
        "Second memory",
        importance=5,
    )

    recent = memory.recent(limit=1)

    assert len(recent) == 1
    assert recent[0].content == "Second memory"


def test_memory_delete(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    record = memory.save(
        "Temporary memory"
    )

    memory.delete(record.id)

    assert memory.count() == 0

    with pytest.raises(KeyError):
        memory.get(record.id)


def test_memory_validates_content(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    with pytest.raises(ValueError):
        memory.save("   ")


def test_memory_validates_importance(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    with pytest.raises(ValueError):
        memory.save(
            "Test",
            importance=11,
        )


def test_memory_retriever(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    memory.save(
        "User prefers Python for OSA development.",
        category="preference",
        importance=9,
    )

    memory.save(
        "OSA runtime target is Windows.",
        category="platform",
        importance=6,
    )

    retriever = MemoryRetriever(memory)

    results = retriever.search(
        "Python OSA"
    )

    assert len(results) == 1
    assert results[0].memory.content == (
        "User prefers Python for OSA development."
    )
    assert results[0].score > 0
