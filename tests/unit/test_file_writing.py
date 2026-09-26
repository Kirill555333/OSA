from pathlib import Path

from osa.tools import (
    SafeFilesystem,
    WriteFileTool,
)


def create_workspace(tmp_path: Path) -> SafeFilesystem:
    """Create a temporary filesystem workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    return SafeFilesystem(workspace)


def test_safe_filesystem_writes_new_file(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    characters = filesystem.write_file(
        "hello.txt",
        "Hello OSA",
    )

    assert characters == 9

    assert (
        filesystem.read_file("hello.txt")
        == "Hello OSA"
    )


def test_safe_filesystem_creates_parent_directories(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    filesystem.write_file(
        "notes/test.txt",
        "OSA",
    )

    assert (
        filesystem.read_file("notes/test.txt")
        == "OSA"
    )


def test_safe_filesystem_rejects_overwrite_by_default(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    filesystem.write_file(
        "test.txt",
        "first",
    )

    try:
        filesystem.write_file(
            "test.txt",
            "second",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError(
            "Existing files must not be overwritten by default."
        )

    assert (
        filesystem.read_file("test.txt")
        == "first"
    )


def test_safe_filesystem_allows_explicit_overwrite(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    filesystem.write_file(
        "test.txt",
        "first",
    )

    filesystem.write_file(
        "test.txt",
        "second",
        overwrite=True,
    )

    assert (
        filesystem.read_file("test.txt")
        == "second"
    )


def test_safe_filesystem_rejects_parent_escape(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    try:
        filesystem.write_file(
            "../secret.txt",
            "secret",
        )
    except RuntimeError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError(
            "Path traversal should have been rejected."
        )


def test_write_file_tool_writes_file(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    tool = WriteFileTool(
        filesystem
    )

    result = tool.execute(
        {
            "path": "test.txt",
            "content": "OSA",
        }
    )

    assert result.success is True
    assert "File written successfully" in result.output

    assert (
        filesystem.read_file("test.txt")
        == "OSA"
    )


def test_write_file_tool_rejects_invalid_content(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    tool = WriteFileTool(
        filesystem
    )

    result = tool.execute(
        {
            "path": "test.txt",
            "content": 123,
        }
    )

    assert result.success is False
    assert "content" in result.error


def test_write_file_tool_rejects_overwrite_by_default(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    filesystem.write_file(
        "test.txt",
        "original",
    )

    tool = WriteFileTool(
        filesystem
    )

    result = tool.execute(
        {
            "path": "test.txt",
            "content": "replacement",
        }
    )

    assert result.success is False

    assert (
        filesystem.read_file("test.txt")
        == "original"
    )


def test_write_file_tool_supports_explicit_overwrite(
    tmp_path: Path,
) -> None:
    filesystem = create_workspace(
        tmp_path
    )

    filesystem.write_file(
        "test.txt",
        "original",
    )

    tool = WriteFileTool(
        filesystem
    )

    result = tool.execute(
        {
            "path": "test.txt",
            "content": "replacement",
            "overwrite": True,
        }
    )

    assert result.success is True

    assert (
        filesystem.read_file("test.txt")
        == "replacement"
    )
