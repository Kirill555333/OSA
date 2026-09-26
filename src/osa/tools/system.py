"""System information tool for OSA."""

from __future__ import annotations

import json
from typing import Any, Mapping

from osa.system import (
    SystemProvider,
    create_system_provider,
)
from osa.tools.registry import (
    ToolInterface,
    ToolResult,
)


class SystemInfoTool(ToolInterface):
    """Expose read-only host system information."""

    def __init__(
        self,
        provider: SystemProvider | None = None,
    ) -> None:
        self._provider = (
            provider
            or create_system_provider()
        )

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "system_info"

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return (
            "Get read-only information about the operating system, "
            "hardware architecture, CPU count, and Python version."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Return system information as JSON."""
        if arguments:
            return ToolResult(
                success=False,
                error=(
                    "system_info does not accept any arguments."
                ),
            )

        try:
            info = self._provider.get_system_info()
        except Exception as exc:
            return ToolResult(
                success=False,
                error=f"Failed to read system information: {exc}",
            )

        payload = {
            "operating_system": info.operating_system,
            "os_release": info.os_release,
            "os_version": info.os_version,
            "machine": info.machine,
            "processor": info.processor,
            "architecture": info.architecture,
            "cpu_count": info.cpu_count,
            "python_version": info.python_version,
            "platform_name": info.platform_name,
        }

        return ToolResult(
            success=True,
            output=json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            metadata=payload,
        )
