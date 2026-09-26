from pathlib import Path

from osa.memory import LongTermMemory
from osa.tools import (
    ForgetTool,
    RecallTool,
    RememberTool,
)


def create_memory(tmp_path: Path) -> LongTermMemory:
    """Create an isolated memory database for testing."""
    return LongTermMemory(
        tmp_path / "memory.db"
    )


def test_remember_tool_saves_memory(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)
    tool = RememberTool(memory)

    result = tool.execute(
        {
            "content": "OSA uses Python.",
            "category": "project",
            "importance": 8,
        }
    )

    assert result.success is True
    assert "Memory ID:" in result.output
    assert memory.count() == 1


def test_remember_tool_requires_content(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)
    tool = RememberTool(memory)

    result = tool.execute({})

    assert result.success is False
    assert "content" in result.error


def test_recall_tool_finds_memory(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    memory.save(
        "OSA uses Python.",
        category="project",
        importance=8,
    )

    tool = RecallTool(memory)

    result = tool.execute(
        {
            "query": "Python OSA",
        }
    )

    assert result.success is True
    assert "OSA uses Python." in result.output


def test_recall_tool_returns_empty_result(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)
    tool = RecallTool(memory)

    result = tool.execute(
        {
            "query": "does not exist",
        }
    )

    assert result.success is True
    assert result.output == "No matching memories were found."


def test_forget_tool_deletes_memory(tmp_path: Path) -> None:
    memory = create_memory(tmp_path)

    record = memory.save(
        "Temporary memory.",
        category="test",
    )

    tool = ForgetTool(memory)

    result = tool.execute(
        {
            "memory_id": record.id,
        }
    )

    assert result.success is True
    assert memory.count() == 0


def test_forget_tool_rejects_unknown_memory(
    tmp_path: Path,
) -> None:
    memory = create_memory(tmp_path)
    tool = ForgetTool(memory)

    result = tool.execute(
        {
            "memory_id": 999,
        }
    )

    assert result.success is False
    assert "does not exist" in result.error
