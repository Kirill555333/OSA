"""Safe workspace shell execution tool for OSA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Mapping

from osa.tools.registry import ToolInterface, ToolResult


class SafeShellError(RuntimeError):
    """Raised when shell execution fails or is rejected."""


@dataclass(frozen=True, slots=True)
class ShellExecutionResult:
    """Result of shell command execution."""

    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    truncated: bool


class SafeShell:
    """Restricted shell command executor inside an isolated workspace root."""

    BLOCKED_PATTERNS = (
        re.compile(
            r"\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*[rf][a-zA-Z]*\s+(?:/|~|\$HOME)(?:\s|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+-(?:[a-zA-Z]*f[a-zA-Z]*\s+)+(?:/|~|\$HOME)(?:\s|$)",
            re.IGNORECASE,
        ),
        re.compile(r"\bmkfs", re.IGNORECASE),
        re.compile(r"\bdd\s+if=/dev/zero", re.IGNORECASE),
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        re.compile(r"\bshutdown\b", re.IGNORECASE),
        re.compile(r"\breboot\b", re.IGNORECASE),
        re.compile(r"\binit\s+0\b", re.IGNORECASE),
    )

    def __init__(
        self,
        workspace_root: Path,
        *,
        default_timeout_s: float = 30.0,
        max_output_chars: int = 15_000,
    ) -> None:
        self._workspace_root = workspace_root.expanduser().resolve()
        if not self._workspace_root.exists() or not self._workspace_root.is_dir():
            raise NotADirectoryError(f"Workspace directory does not exist: {self._workspace_root}")

        if default_timeout_s <= 0:
            raise ValueError("default_timeout_s must be greater than zero.")

        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be greater than zero.")

        self._default_timeout_s = default_timeout_s
        self._max_output_chars = max_output_chars

    @property
    def workspace_root(self) -> Path:
        """Return the workspace root directory."""
        return self._workspace_root

    def execute(
        self,
        command: str,
        *,
        working_directory: str | None = None,
        timeout_s: float | None = None,
    ) -> ShellExecutionResult:
        """Execute a shell command with safety checks and timeout protection."""
        norm_cmd = command.strip()
        if not norm_cmd:
            raise SafeShellError("Command cannot be empty.")

        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(norm_cmd):
                raise SafeShellError(f"Command blocked for system safety: '{norm_cmd}'")

        cwd = self._resolve_cwd(working_directory)
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s

        started_at = time.perf_counter()

        try:
            res = subprocess.run(
                norm_cmd,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            duration_ms = (time.perf_counter() - started_at) * 1000.0

            raw_stdout = res.stdout
            raw_stderr = res.stderr

            truncated = False
            if len(raw_stdout) > self._max_output_chars:
                raw_stdout = (
                    raw_stdout[: self._max_output_chars]
                    + f"\n...[stdout truncated at {self._max_output_chars} characters]"
                )
                truncated = True

            if len(raw_stderr) > self._max_output_chars:
                raw_stderr = (
                    raw_stderr[: self._max_output_chars]
                    + f"\n...[stderr truncated at {self._max_output_chars} characters]"
                )
                truncated = True

            return ShellExecutionResult(
                returncode=res.returncode,
                stdout=raw_stdout,
                stderr=raw_stderr,
                duration_ms=duration_ms,
                truncated=truncated,
            )

        except subprocess.TimeoutExpired as exc:
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            raise SafeShellError(
                f"Command timed out after {timeout} seconds: '{norm_cmd}'"
            ) from exc
        except Exception as exc:
            if isinstance(exc, SafeShellError):
                raise
            raise SafeShellError(f"Failed to execute command '{norm_cmd}': {exc}") from exc

    def _resolve_cwd(self, requested: str | None) -> Path:
        if not requested or requested.strip() in (".", ""):
            return self._workspace_root

        target = (self._workspace_root / requested.strip()).resolve()
        if not target.is_relative_to(self._workspace_root):
            raise SafeShellError("Working directory outside workspace root is forbidden.")

        if not target.exists() or not target.is_dir():
            raise SafeShellError(f"Working directory does not exist: {requested}")

        return target


class TerminalExecuteTool(ToolInterface):
    """Execute a shell command inside the OSA workspace."""

    def __init__(self, shell: SafeShell) -> None:
        self._shell = shell

    @property
    def name(self) -> str:
        return "terminal_execute"

    @property
    def description(self) -> str:
        return (
            "Execute a shell command inside the OSA workspace with safety guards and output capture."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command line to execute.",
                },
                "working_directory": {
                    "type": "string",
                    "description": "Relative directory inside workspace (default: '.').",
                    "default": ".",
                },
                "timeout_s": {
                    "type": "number",
                    "description": "Timeout in seconds (default: 30.0).",
                    "minimum": 1.0,
                    "maximum": 120.0,
                    "default": 30.0,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        command = arguments.get("command")
        working_dir = arguments.get("working_directory", ".")
        timeout_s = arguments.get("timeout_s", 30.0)

        if not isinstance(command, str) or not command.strip():
            return ToolResult(
                success=False,
                error="Argument 'command' must be a non-empty string.",
            )

        if not isinstance(working_dir, str):
            return ToolResult(
                success=False,
                error="Argument 'working_directory' must be a string.",
            )

        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
            return ToolResult(
                success=False,
                error="Argument 'timeout_s' must be a number.",
            )

        try:
            result = self._shell.execute(
                command,
                working_directory=working_dir,
                timeout_s=float(timeout_s),
            )
        except SafeShellError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        output_parts: list[str] = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append(f"[stderr]\n{result.stderr.strip()}")

        combined_output = "\n".join(output_parts) or "(command completed with no output)"

        success = result.returncode == 0
        error_msg = None if success else f"Command exited with non-zero status code: {result.returncode}"

        return ToolResult(
            success=success,
            output=combined_output,
            error=error_msg,
            metadata={
                "returncode": result.returncode,
                "duration_ms": result.duration_ms,
                "truncated": result.truncated,
            },
        )
