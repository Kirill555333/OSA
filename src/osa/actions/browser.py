"""Browser adapter for the unified OSA action system."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.actions.router import ActionHandler


class BrowserActionAdapterError(RuntimeError):
    """Raised when the browser action adapter is misconfigured."""


class BrowserActionTool(Protocol):
    """Minimal protocol for an existing browser action tool."""

    @property
    def name(self) -> str:
        """Return the stable tool name."""
        ...

    def execute(self, arguments: Mapping[str, Any]) -> Any:
        """Execute the browser tool."""
        ...


BROWSER_ACTION_NAMES = (
    "browser_open",
    "browser_find",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_read",
    "browser_wait",
    "browser_observe",
    "browser_screenshot",
)


class BrowserActionAdapter(ActionHandler):
    """Adapt the existing browser tool layer to unified actions."""

    def __init__(
        self,
        tools: Iterable[BrowserActionTool],
    ) -> None:
        self._tools: dict[str, BrowserActionTool] = {}

        for tool in tools:
            self.register(tool)

    @property
    def supported_actions(self) -> tuple[str, ...]:
        """Return supported browser action names."""
        return tuple(
            name
            for name in BROWSER_ACTION_NAMES
            if name in self._tools
        )

    def register(
        self,
        tool: BrowserActionTool,
    ) -> None:
        """Register one browser action tool."""
        name = getattr(tool, "name", None)

        if not isinstance(name, str) or not name.strip():
            raise BrowserActionAdapterError(
                "Browser action tool must expose a non-empty name."
            )

        normalized_name = name.strip()

        if normalized_name not in BROWSER_ACTION_NAMES:
            raise BrowserActionAdapterError(
                f"Unsupported browser action '{normalized_name}'."
            )

        execute = getattr(tool, "execute", None)

        if not callable(execute):
            raise BrowserActionAdapterError(
                f"Browser action '{normalized_name}' must expose execute()."
            )

        self._tools[normalized_name] = tool

    def unregister(
        self,
        action_name: str,
    ) -> None:
        """Remove a browser action tool when present."""
        self._tools.pop(action_name, None)

    def dispatch(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        """Alias for execute, matching the adapter terminology."""
        return self.execute(request)

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        """Execute one browser action through the existing tool layer."""
        if not isinstance(request, ActionRequest):
            raise BrowserActionAdapterError(
                "request must be an ActionRequest."
            )

        if request.kind is not ActionKind.BROWSER:
            return ActionResult.failed(
                request.request_id,
                (
                    "BrowserActionAdapter received "
                    f"'{request.kind.value}' action."
                ),
                metadata={
                    "adapter_error": "wrong_action_kind",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        tool = self._tools.get(request.name)

        if tool is None:
            return ActionResult.failed(
                request.request_id,
                f"Unsupported browser action '{request.name}'.",
                metadata={
                    "adapter_error": "action_not_registered",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        try:
            tool_result = tool.execute(request.arguments)
        except Exception as exc:
            return ActionResult.failed(
                request.request_id,
                f"Browser action failed: {exc}",
                metadata={
                    "adapter_error": "tool_exception",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                    "exception_type": type(exc).__name__,
                },
            )

        success = getattr(tool_result, "success", None)
        output = getattr(tool_result, "output", None)
        error = getattr(tool_result, "error", None)

        if not isinstance(success, bool):
            return ActionResult.failed(
                request.request_id,
                "Browser action returned an invalid tool result.",
                metadata={
                    "adapter_error": "invalid_tool_result",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                    "result_type": type(tool_result).__name__,
                },
            )

        if not isinstance(output, str):
            return ActionResult.failed(
                request.request_id,
                "Browser action returned non-string output.",
                metadata={
                    "adapter_error": "invalid_tool_output",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        if success:
            return ActionResult.succeeded(
                request.request_id,
                output=output,
                metadata={
                    "adapter": "browser",
                    "action_name": request.name,
                },
            )

        if not isinstance(error, str) or not error.strip():
            return ActionResult.failed(
                request.request_id,
                "Browser action failed without an error message.",
                metadata={
                    "adapter_error": "missing_tool_error",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        return ActionResult.failed(
            request.request_id,
            error,
            output=output,
            metadata={
                "adapter": "browser",
                "action_name": request.name,
            },
        )

    def tool_for(
        self,
        action_name: str,
    ) -> BrowserActionTool | None:
        """Return the registered browser action tool."""
        return self._tools.get(action_name)
