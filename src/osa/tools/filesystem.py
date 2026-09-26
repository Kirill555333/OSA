"""Safe read-only filesystem tools for OSA."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from osa.tools.registry import ToolInterface, ToolResult


class FilesystemToolError(RuntimeError):
    """Raised when a filesystem operation cannot be completed."""


class SafeFilesystem:
    """Restricted filesystem access within a single workspace."""

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

        if not self._root.exists():
            raise FileNotFoundError(
                f"Filesystem workspace does not exist: {self._root}"
            )

        if not self._root.is_dir():
            raise NotADirectoryError(
                f"Filesystem workspace is not a directory: {self._root}"
            )

    @property
    def root(self) -> Path:
        """Return the resolved workspace root."""
        return self._root

    def resolve_path(self, requested_path: str) -> Path:
        """Resolve a path and prevent access outside the workspace."""
        normalized = requested_path.strip()

        if not normalized:
            raise FilesystemToolError(
                "Path cannot be empty."
            )

        candidate = (self._root / normalized).resolve()

        if not candidate.is_relative_to(self._root):
            raise FilesystemToolError(
                "Access outside the OSA workspace is not allowed."
            )

        return candidate

    def list_directory(self, requested_path: str = ".") -> list[str]:
        """Return directory entries inside the workspace."""
        directory = self.resolve_path(requested_path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {requested_path}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {requested_path}"
            )

        entries = sorted(
            directory.iterdir(),
            key=lambda path: (not path.is_dir(), path.name.lower()),
        )

        return [
            self._display_path(entry)
            for entry in entries
        ]

    def read_file(
        self,
        requested_path: str,
        *,
        max_characters: int = 50_000,
    ) -> str:
        """Read a UTF-8 text file inside the workspace."""
        if max_characters <= 0:
            raise ValueError(
                "max_characters must be greater than zero."
            )

        file_path = self.resolve_path(requested_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"File does not exist: {requested_path}"
            )

        if not file_path.is_file():
            raise IsADirectoryError(
                f"Path is not a file: {requested_path}"
            )

        try:
            content = file_path.read_text(
                encoding="utf-8"
            )
        except UnicodeDecodeError as exc:
            raise FilesystemToolError(
                f"File is not valid UTF-8 text: {requested_path}"
            ) from exc

        return content[:max_characters]

    def file_exists(self, requested_path: str) -> bool:
        """Check whether a path exists inside the workspace."""
        return self.resolve_path(requested_path).exists()

    def write_file(
        self,
        requested_path: str,
        content: str,
        *,
        overwrite: bool = False,
    ) -> int:
        """Write UTF-8 text inside the workspace."""
        if not isinstance(content, str):
            raise TypeError(
                "content must be a string."
            )

        file_path = self.resolve_path(
            requested_path
        )

        if file_path.exists():
            if not file_path.is_file():
                raise IsADirectoryError(
                    f"Path is not a file: {requested_path}"
                )

            if not overwrite:
                raise FileExistsError(
                    f"File already exists: {requested_path}"
                )

        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            written = file_path.write_text(
                content,
                encoding="utf-8",
            )
        except OSError as exc:
            raise FilesystemToolError(
                f"Failed to write file: {requested_path}"
            ) from exc

        return written

    def _display_path(self, path: Path) -> str:
        """Return a workspace-relative display path."""
        relative = path.relative_to(self._root)

        if not relative.parts:
            return "."

        return relative.as_posix()


class ListDirectoryTool(ToolInterface):
    """List entries in the OSA workspace."""

    def __init__(self, filesystem: SafeFilesystem) -> None:
        self._filesystem = filesystem

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "list_directory"

    @property
    def description(self) -> str:
        """Return a human-readable tool description."""
        return "List files and directories inside the OSA workspace."

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative directory path. "
                        "Use '.' for the workspace root."
                    ),
                    "default": ".",
                }
            },
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """List directory contents."""
        path = arguments.get("path", ".")

        if not isinstance(path, str):
            return ToolResult(
                success=False,
                error="Argument 'path' must be a string.",
            )

        try:
            entries = self._filesystem.list_directory(path)
        except (
            FilesystemToolError,
            FileNotFoundError,
            NotADirectoryError,
            ValueError,
        ) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        if not entries:
            return ToolResult(
                success=True,
                output="Directory is empty.",
            )

        return ToolResult(
            success=True,
            output="\n".join(entries),
        )


class ReadFileTool(ToolInterface):
    """Read text files inside the OSA workspace."""

    def __init__(self, filesystem: SafeFilesystem) -> None:
        self._filesystem = filesystem

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "read_file"

    @property
    def description(self) -> str:
        """Return the contents of a UTF-8 text file inside the OSA workspace."""

        return "Read a UTF-8 text file inside the OSA workspace."

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative file path.",
                },
                "max_characters": {
                    "type": "integer",
                    "description": "Maximum number of characters to read.",
                    "minimum": 1,
                    "maximum": 200_000,
                    "default": 50_000,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Read a UTF-8 text file."""
        path = arguments.get("path")
        max_characters = arguments.get(
            "max_characters",
            50_000,
        )

        if not isinstance(path, str):
            return ToolResult(
                success=False,
                error="Argument 'path' must be a string.",
            )

        if not isinstance(max_characters, int) or isinstance(
            max_characters,
            bool,
        ):
            return ToolResult(
                success=False,
                error="Argument 'max_characters' must be an integer.",
            )

        try:
            content = self._filesystem.read_file(
                path,
                max_characters=max_characters,
            )
        except (
            FilesystemToolError,
            FileNotFoundError,
            IsADirectoryError,
            UnicodeDecodeError,
            ValueError,
        ) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=content,
        )


class FileExistsTool(ToolInterface):
    """Check whether a path exists inside the OSA workspace."""

    def __init__(self, filesystem: SafeFilesystem) -> None:
        self._filesystem = filesystem

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "file_exists"

    @property
    def description(self) -> str:
        """Return a human-readable existence check."""
        return "Check whether a file or directory exists in the OSA workspace."

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative path.",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Check path existence."""
        path = arguments.get("path")

        if not isinstance(path, str):
            return ToolResult(
                success=False,
                error="Argument 'path' must be a string.",
            )

        try:
            exists = self._filesystem.file_exists(path)
        except (
            FilesystemToolError,
            ValueError,
        ) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output="true" if exists else "false",
        )
class WriteFileTool(ToolInterface):
    """Write UTF-8 text files inside the OSA workspace."""

    def __init__(
        self,
        filesystem: SafeFilesystem,
    ) -> None:
        self._filesystem = filesystem

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "write_file"

    @property
    def description(self) -> str:
        """Return a human-readable tool description."""
        return (
            "Write UTF-8 text to a file inside the OSA workspace. "
            "Existing files require overwrite=true."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Workspace-relative file path."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "UTF-8 text content to write."
                    ),
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "Allow replacing an existing file."
                    ),
                    "default": False,
                },
            },
            "required": [
                "path",
                "content",
            ],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Write a UTF-8 text file."""
        path = arguments.get("path")
        content = arguments.get("content")
        overwrite = arguments.get(
            "overwrite",
            False,
        )

        if not isinstance(path, str):
            return ToolResult(
                success=False,
                error="Argument 'path' must be a string.",
            )

        if not isinstance(content, str):
            return ToolResult(
                success=False,
                error="Argument 'content' must be a string.",
            )

        if not isinstance(overwrite, bool):
            return ToolResult(
                success=False,
                error="Argument 'overwrite' must be a boolean.",
            )

        try:
            characters_written = self._filesystem.write_file(
                path,
                content,
                overwrite=overwrite,
            )
        except (
            FilesystemToolError,
            FileExistsError,
            IsADirectoryError,
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=(
                f"File written successfully: {path} "
                f"({characters_written} characters)."
            ),
            metadata={
                "path": path,
                "characters_written": characters_written,
                "overwritten": overwrite,
            },
        )
