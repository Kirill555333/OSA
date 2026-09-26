from pathlib import Path

from osa.memory.long_term import LongTermMemory
from osa.memory.retrieval import MemoryRetriever


def create_retriever(
    tmp_path: Path,
) -> tuple[LongTermMemory, MemoryRetriever]:
    memory = LongTermMemory(
        tmp_path / "memory.db"
    )

    return memory, MemoryRetriever(memory)


def test_retrieval_handles_natural_language_and_cyrillic(
    tmp_path: Path,
) -> None:
    memory, retriever = create_retriever(
        tmp_path
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
        category="general",
        importance=8,
    )

    results = retriever.search(
        "Что ты знаешь об основном языке разработки OSA?"
    )

    assert len(results) == 1
    assert (
        results[0].memory.content
        == "Основной язык разработки OSA — Python."
    )


def test_retrieval_prefers_memory_with_more_matching_terms(
    tmp_path: Path,
) -> None:
    memory, retriever = create_retriever(
        tmp_path
    )

    partial = memory.save(
        "OSA использует Python.",
        category="general",
        importance=5,
    )

    exact = memory.save(
        "Основной язык разработки OSA — Python.",
        category="general",
        importance=5,
    )

    results = retriever.search(
        "язык разработки OSA Python",
        limit=2,
    )

    assert len(results) == 1
    assert results[0].memory.id == exact.id


def test_retrieval_can_match_memory_category(
    tmp_path: Path,
) -> None:
    memory, retriever = create_retriever(
        tmp_path
    )

    record = memory.save(
        "Qwen3-4B используется как локальная модель.",
        category="models",
        importance=7,
    )

    results = retriever.search(
        "локальная models",
    )

    assert results
    assert results[0].memory.id == record.id


def test_retrieval_ignores_stopword_only_query(
    tmp_path: Path,
) -> None:
    memory, retriever = create_retriever(
        tmp_path
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
    )

    results = retriever.search(
        "что это и как",
    )

    assert results == ()


def test_retrieval_is_unicode_case_insensitive(
    tmp_path: Path,
) -> None:
    memory, retriever = create_retriever(
        tmp_path
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
    )

    results = retriever.search(
        "ОСНОВНОЙ ЯЗЫК РАЗРАБОТКИ OSA",
    )

    assert results
    assert (
        results[0].memory.content
        == "Основной язык разработки OSA — Python."
    )
