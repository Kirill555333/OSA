"""Unified action routing for OSA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult


class ActionRouterError(RuntimeError):
    """Raised when the action router configuration is invalid."""


class ActionHandler(Protocol):
    """Protocol implemented by concrete action backends."""

    def execute(self, request: ActionRequest) -> ActionResult:
        """Execute one action request."""
        ...


class ActionRouter:
    """Route backend-neutral action requests to registered handlers."""

    def __init__(
        self,
        handlers: Mapping[ActionKind | str, ActionHandler] | None = None,
    ) -> None:
        self._handlers: dict[ActionKind, ActionHandler] = {}

        if handlers is not None:
            for kind, handler in handlers.items():
                self.register(kind, handler)

    @property
    def supported_kinds(self) -> tuple[ActionKind, ...]:
        """Return registered action kinds in deterministic order."""
        return tuple(
            sorted(
                self._handlers,
                key=lambda item: item.value,
            )
        )

    def register(
        self,
        kind: ActionKind | str,
        handler: ActionHandler,
    ) -> None:
        """Register or replace the handler for one action kind."""
        normalized_kind = self._normalize_kind(kind)

        if not callable(getattr(handler, "execute", None)):
            raise ActionRouterError(
                f"Handler for '{normalized_kind.value}' must expose execute()."
            )

        self._handlers[normalized_kind] = handler

    def unregister(
        self,
        kind: ActionKind | str,
    ) -> None:
        """Remove the handler for one action kind when present."""
        normalized_kind = self._normalize_kind(kind)
        self._handlers.pop(normalized_kind, None)

    def handler_for(
        self,
        kind: ActionKind | str,
    ) -> ActionHandler | None:
        """Return the registered handler for one action kind."""
        normalized_kind = self._normalize_kind(kind)
        return self._handlers.get(normalized_kind)

    def dispatch(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        """Dispatch one request and normalize backend failures."""
        if not isinstance(request, ActionRequest):
            raise ActionRouterError(
                "request must be an ActionRequest."
            )

        handler = self._handlers.get(request.kind)

        if handler is None:
            return ActionResult.failed(
                request.request_id,
                (
                    "No action handler is registered for "
                    f"'{request.kind.value}'."
                ),
                metadata={
                    "router_error": "handler_not_registered",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        try:
            result = handler.execute(request)
        except Exception as exc:
            return ActionResult.failed(
                request.request_id,
                f"Action handler failed: {exc}",
                metadata={
                    "router_error": "handler_exception",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                    "exception_type": type(exc).__name__,
                },
            )

        if not isinstance(result, ActionResult):
            return ActionResult.failed(
                request.request_id,
                "Action handler returned an invalid result type.",
                metadata={
                    "router_error": "invalid_handler_result",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                    "result_type": type(result).__name__,
                },
            )

        if result.request_id != request.request_id:
            return ActionResult.failed(
                request.request_id,
                "Action handler returned a mismatched request_id.",
                metadata={
                    "router_error": "request_id_mismatch",
                    "action_kind": request.kind.value,
                    "action_name": request.name,
                },
            )

        return result

    @staticmethod
    def _normalize_kind(
        kind: ActionKind | str,
    ) -> ActionKind:
        if isinstance(kind, ActionKind):
            return kind

        try:
            return ActionKind(kind)
        except (TypeError, ValueError) as exc:
            raise ActionRouterError(
                "kind must be a valid ActionKind."
            ) from exc
