"""Tools that let the user rename the assistant's spoken name (OSA 1.5.2.4).

The model decides when the user asks to be allowed to address the assistant by another
name. The built-in name keeps working; only one extra name exists at a time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osa.tools.registry import ToolInterface, ToolResult
from osa.voice.names import BASE_DISPLAY_NAME


class SetWakeNameTool(ToolInterface):
    """Set the extra spoken name of the assistant."""

    def __init__(self, service: Any) -> None:
        self._service = service

    @property
    def name(self) -> str:
        return "set_wake_name"

    @property
    def description(self) -> str:
        return (
            "Set an additional spoken name for the assistant. Use ONLY when the user "
            "explicitly says they want to call the assistant by another name (for example "
            "'from now on I will call you Jarvis'). The original name keeps working too. "
            "Write the name in Russian letters as it sounds."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The new name, a single word, for example Джарвис.",
                },
            },
            "required": ["name"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            change = self._service.set_custom_name(str(arguments.get("name", "")))
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        shown = change.name.capitalize()
        output = f"Теперь я откликаюсь на «{shown}» и всегда на «{BASE_DISPLAY_NAME}»."

        if not change.persisted:
            output += " Имя не удалось сохранить в файл, оно действует до перезапуска."

        return ToolResult(
            success=True,
            output=output,
            metadata={"name": change.name, "persisted": change.persisted},
        )


class ClearWakeNameTool(ToolInterface):
    """Remove the extra spoken name of the assistant."""

    def __init__(self, service: Any) -> None:
        self._service = service

    @property
    def name(self) -> str:
        return "clear_wake_name"

    @property
    def description(self) -> str:
        return (
            "Remove the additional spoken name of the assistant. Use ONLY when the user "
            "explicitly asks to stop using the extra name; the original name keeps working."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        had_name = self._service.clear_custom_name()

        if not had_name:
            return ToolResult(
                success=True,
                output=f"Дополнительного имени не было, я откликаюсь на «{BASE_DISPLAY_NAME}».",
                metadata={"removed": False},
            )

        return ToolResult(
            success=True,
            output=f"Дополнительное имя удалено, я откликаюсь на «{BASE_DISPLAY_NAME}».",
            metadata={"removed": True},
        )
