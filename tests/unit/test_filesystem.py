from pathlib import Path

from osa.tools import (
    FileExistsTool,
    ListDirectoryTool,
    ReadFileTool,
    SafeFilesystem,
)


def create_workspace(tmp_path: Path) -> SafeFilesystem:
    """Create a temporary filesystem workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    return SafeFilesystem(workspace)


def test_safe_filesystem_lists_directory(tmp_path) -> None:
    filesystem = create_workspace(tmp_path)

    (filesystem.root / "alpha.txt").write_text(
        "alpha",
        encoding="utf-8",
    )

    (filesystem.root / "folder").mkdir()

    entries = filesystem.list_directory()

    assert entries == [
        "folder",
        "alpha.txt",
    ]


def test_safe_filesystem_reads_text_file(tmp_path) -> None:
    filesystem = create_workspace(tmp_path)

    file_path = filesystem.root / "hello.txt"
    file_path.write_text(
        "Hello OSA",
        encoding="utf-8",
    )

    assert filesystem.read_file("hello.txt") == "Hello OSA"


def test_safe_filesystem_rejects_parent_escape(tmp_path) -> None:
    filesystem = create_workspace(tmp_path)

    try:
        filesystem.resolve_path("../secret.txt")
    except RuntimeError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError(
            "Path traversal should have been rejected."
        )


def test_file_exists_tool(tmp_path) -> None:
    filesystem = create_workspace(tmp_path)

    (filesystem.root / "test.txt").write_text(
        "test",
        encoding="utf-8",
    )

    tool = FileExistsTool(filesystem)

    result = tool.execute(
        {"path": "test.txt"}
    )

    assert result.success is True
    assert result.output == "true"


def test_file_exists_tool_for_missing_file(tmp_path) -> None:
    filesystem = create_workspace(tmp_path)

    tool = FileExistsTool(filesystem)

    result = tool.execute(
        {"path": "missing.txt"}
    )

    assert result.success is True
    assert result.output == "false"


def test_read_file_tool(tmp_path) -> None:
    filesystem = create_workspace(tmp_path)

    (filesystem.root / "test.txt").write_text(
        "OSA",
        encoding="utf-8",
    )

    tool = ReadFileTool(filesystem)

    result = tool.execute(
        {"path": "test.txt"}
    )

    assert result.success is True
    assert result.output == "OSA"


def test_list_directory_tool(tmp_path) -> None:
    filesystem = create_workspace(tmp_path)

    (filesystem.root / "test.txt").write_text(
        "OSA",
        encoding="utf-8",
    )

    tool = ListDirectoryTool(filesystem)

    result = tool.execute(
        {"path": "."}
    )

    assert result.success is True
    assert result.output == "test.txt"
