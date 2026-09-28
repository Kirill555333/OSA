from pathlib import Path

from osa.memory import LongTermMemory
from osa.tools import SafeFilesystem


def test_filesystem_supports_cross_platform_style_workspace(
    tmp_path: Path,
) -> None:
    workspace = (
        tmp_path
        / "OSA Data"
        / "workspace files"
    )

    workspace.mkdir(
        parents=True
    )

    filesystem = SafeFilesystem(
        workspace
    )

    written = filesystem.write_file(
        "notes/test file.txt",
        "OSA works across platforms.",
    )

    assert written == len(
        "OSA works across platforms."
    )

    assert filesystem.file_exists(
        "notes/test file.txt"
    )

    assert filesystem.read_file(
        "notes/test file.txt"
    ) == "OSA works across platforms."

    assert filesystem.list_directory(
        "notes"
    ) == ["notes/test file.txt"]


def test_memory_works_in_cross_platform_path(
    tmp_path: Path,
) -> None:
    database = (
        tmp_path
        / "OSA Data"
        / "memory store"
        / "osa-memory.db"
    )

    memory = LongTermMemory(
        database
    )

    record = memory.save(
        "OSA runtime is cross-platform.",
        category="project",
        importance=8,
    )

    loaded = memory.get(
        record.id
    )

    assert loaded.content == (
        "OSA runtime is cross-platform."
    )
    assert memory.count() == 1
