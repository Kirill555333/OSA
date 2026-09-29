"""Unified TOOL action adapter for OSA."""

from __future__ import annotations

from typing import Any

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.tools import ToolRegistry, ToolResult


class ToolActionAdapterError(RuntimeError):
    """Raised when a TOOL action cannot be executed by the adapter."""


class ToolRegistryActionAdapter:
    """Execute unified TOOL requests through a ToolRegistry."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be a ToolRegistry.")

        self._tool_registry = tool_registry

    @property
    def tool_registry(self) -> ToolRegistry:
        """Return the backing tool registry."""
        return self._tool_registry

    def execute(self, request: ActionRequest) -> ActionResult:
        """Execute one TOOL request and normalize the ToolResult."""
        if not isinstance(request, ActionRequest):
            raise ToolActionAdapterError(
                "request must be an ActionRequest."
            )

        if request.kind is not ActionKind.TOOL:
            raise ToolActionAdapterError(
                "ToolRegistryActionAdapter only supports TOOL actions."
            )

        try:
            tool = self._tool_registry.get(request.name)
        except KeyError as exc:
            return ActionResult.failed(
                request.request_id,
                str(exc),
                metadata={
                    "tool_error": "tool_not_registered",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        result = tool.execute(request.arguments)

        if not isinstance(result, ToolResult):
            return ActionResult.failed(
                request.request_id,
                "Tool returned an invalid result type.",
                metadata={
                    "tool_error": "invalid_tool_result",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                    "result_type": type(result).__name__,
                },
            )

        tool_data: dict[str, Any] = {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "metadata": dict(result.metadata),
        }

        if result.success:
            return ActionResult.succeeded(
                request.request_id,
                output=result.output,
                data={"tool_result": tool_data},
                metadata={
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        return ActionResult.failed(
            request.request_id,
            result.error or "Tool execution failed.",
            data={"tool_result": tool_data},
            metadata={
                "action_kind": request.kind.value,
                "action_name": request.name,
                "tool_failed": True,
            },
        )
