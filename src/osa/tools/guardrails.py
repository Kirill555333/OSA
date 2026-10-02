"""Tool execution guardrails: output truncation and loop detection."""

from __future__ import annotations

from typing import Any
from osa.tools.registry import ToolResult


class OutputTruncator:
    """Safely truncates huge outputs from shell/browser/file tools."""

    def __init__(self, max_characters: int = 4000, head_lines: int = 30, tail_lines: int = 30) -> None:
        self.max_characters = max_characters
        self.head_lines = head_lines
        self.tail_lines = tail_lines

    def truncate(self, text: str) -> str:
        if not text or len(text) <= self.max_characters:
            return text

        lines = text.splitlines()
        if len(lines) <= (self.head_lines + self.tail_lines):
            return text[: self.max_characters] + "\n... [TRUNCATED DUE TO CONTEXT LIMIT] ..."

        head = "\n".join(lines[: self.head_lines])
        tail = "\n".join(lines[-self.tail_lines :])
        omitted = len(lines) - self.head_lines - self.tail_lines
        return f"{head}\n\n... [{omitted} lines omitted to fit context budget] ...\n\n{tail}"


class ActionLoopDetector:
    """Detects repetitive calls to the same tool with identical arguments."""

    def __init__(self, max_repeated_calls: int = 3) -> None:
        self.max_repeated_calls = max_repeated_calls
        self._history: list[tuple[str, str]] = []

    def record_and_check(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Returns True if a loop is detected, False otherwise."""
        arg_key = str(sorted(arguments.items()))
        entry = (tool_name, arg_key)
        self._history.append(entry)

        if len(self._history) < self.max_repeated_calls:
            return False

        recent = self._history[-self.max_repeated_calls :]
        return all(item == entry for item in recent)

    def reset(self) -> None:
        self._history.clear()


def apply_tool_guardrail(result: ToolResult, truncator: OutputTruncator | None = None) -> ToolResult:
    """Ensure tool result output does not blow up LLM context window."""
    if not result.success or not result.output:
        return result

    active_truncator = truncator or OutputTruncator()
    safe_output = active_truncator.truncate(result.output)
    if safe_output != result.output:
        return ToolResult(
            success=result.success,
            output=safe_output,
            error=result.error,
        )
    return result
