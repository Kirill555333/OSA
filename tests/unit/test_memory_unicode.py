from pathlib import Path

from osa.memory.long_term import LongTermMemory


def test_search_is_case_insensitive_for_cyrillic(
    tmp_path: Path,
) -> None:
    memory = LongTermMemory(
        tmp_path / "memory.db"
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
        category="general",
    )

    results = memory.search(
        "основной язык разработки OSA"
    )

    assert len(results) == 1
    assert (
        results[0].content
        == "Основной язык разработки OSA — Python."
    )


def test_search_handles_uppercase_cyrillic(
    tmp_path: Path,
) -> None:
    memory = LongTermMemory(
        tmp_path / "memory.db"
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
        category="general",
    )

    results = memory.search(
        "ОСНОВНОЙ ЯЗЫК РАЗРАБОТКИ OSA"
    )

    assert len(results) == 1


def test_search_still_requires_all_terms(
    tmp_path: Path,
) -> None:
    memory = LongTermMemory(
        tmp_path / "memory.db"
    )

    memory.save(
        "Основной язык разработки OSA — Python.",
        category="general",
    )

    assert memory.search("OSA Python")
    assert not memory.search("OSA Java")
