from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from osa.actions.contracts import ActionRequest, ActionResult
from osa.actions.router import ActionRouter
from osa.observability.context import ObservabilityContext
from osa.observability.events import ObservabilityEvent
from osa.observability.logger import (
    InMemoryObservabilityLogger,
    ObservabilityLogger,
)


class ActionRouterObservabilityError(RuntimeError):
    """Raised when router observability configuration is invalid."""


class ActionRouterLike(Protocol):
    """Minimal router interface required by ObservableActionRouter."""

    def dispatch(self, request: ActionRequest) -> ActionResult:
        """Dispatch an action request."""
        ...

    @property
    def supported_kinds(self) -> tuple[str, ...]:
        """Return supported action kinds."""
        ...

    def handler_for(self, kind: Any) -> Any | None:
        """Return the registered handler for an action kind."""
        ...

    def register(self, handler: Any) -> None:
        """Register a handler."""
        ...

    def unregister(self, kind: Any) -> Any | None:
        """Unregister a handler."""
        ...


ContextProvider = Any


@dataclass(frozen=True, slots=True)
class _RouterEventContext:
    run_id: str | None = None
    task_id: str | None = None


class ObservableActionRouter:
    """
    Observability wrapper around the existing ActionRouter.

    This class must never change routing semantics:
    - the wrapped router remains authoritative;
    - router exceptions are re-raised;
    - ActionResult values are returned unchanged;
    - logging failures are swallowed.
    """

    def __init__(
        self,
        router: ActionRouterLike | ActionRouter,
        logger: ObservabilityLogger | None = None,
        context_provider: ContextProvider | None = None,
    ) -> None:
        if router is None:
            raise ActionRouterObservabilityError("router is required")

        if logger is None:
            logger = InMemoryObservabilityLogger()

        self._router = router
        self._logger = logger
        self._context_provider = (
            context_provider
            if context_provider is not None
            else self._default_context_provider
        )

    @property
    def router(self) -> ActionRouterLike | ActionRouter:
        return self._router

    @property
    def logger(self) -> ObservabilityLogger:
        return self._logger

    @property
    def context_provider(self) -> ContextProvider:
        return self._context_provider

    @property
    def supported_kinds(self) -> tuple[str, ...]:
        """Preserve the wrapped router's property-based API."""
        return tuple(self._router.supported_kinds)

    def handler_for(self, kind: Any) -> Any | None:
        return self._router.handler_for(kind)

    def register(self, handler: Any) -> None:
        self._router.register(handler)

    def unregister(self, kind: Any) -> Any | None:
        return self._router.unregister(kind)

    def dispatch(self, request: ActionRequest) -> ActionResult:
        if not isinstance(request, ActionRequest):
            raise ActionRouterObservabilityError(
                "request must be an ActionRequest"
            )

        started = perf_counter()
        context = self._resolve_context(request)

        self._safe_log(
            ObservabilityEvent(
                event="router.started",
                source="action_router",
                request_id=request.request_id,
                run_id=context.run_id,
                task_id=context.task_id,
                action_kind=request.kind.value,
                action_name=request.name,
                status="started",
                duration_ms=0.0,
                metadata={
                    "arguments_present": bool(request.arguments),
                },
            )
        )

        try:
            result = self._router.dispatch(request)
        except Exception as exc:
            self._safe_log(
                self._build_failure_event(
                    request=request,
                    context=context,
                    duration_ms=self._duration_ms(started),
                    error=str(exc),
                )
            )
            raise

        duration_ms = self._duration_ms(started)

        if not isinstance(result, ActionResult):
            self._safe_log(
                self._build_failure_event(
                    request=request,
                    context=context,
                    duration_ms=duration_ms,
                    error="wrapped router returned invalid result",
                )
            )
            return ActionResult.failed(
                request.request_id,
                "wrapped router returned invalid result",
            )

        if result.request_id != request.request_id:
            self._safe_log(
                self._build_failure_event(
                    request=request,
                    context=context,
                    duration_ms=duration_ms,
                    error="wrapped router returned mismatched request_id",
                )
            )
            return ActionResult.failed(
                request.request_id,
                "wrapped router returned mismatched request_id",
            )

        if result.success:
            self._safe_log(
                ObservabilityEvent(
                    event="router.completed",
                    source="action_router",
                    request_id=request.request_id,
                    run_id=context.run_id,
                    task_id=context.task_id,
                    action_kind=request.kind.value,
                    action_name=request.name,
                    status="completed",
                    duration_ms=duration_ms,
                    metadata={
                        "output_present": result.output is not None,
                        "data_present": result.data is not None,
                    },
                )
            )
        else:
            self._safe_log(
                self._build_failure_event(
                    request=request,
                    context=context,
                    duration_ms=duration_ms,
                    error=result.error or "router dispatch failed",
                )
            )

        return result

    def _resolve_context(
        self,
        request: ActionRequest,
    ) -> _RouterEventContext:
        try:
            context = self._context_provider(request)
        except Exception:
            context = None

        if isinstance(context, ObservabilityContext):
            return _RouterEventContext(
                run_id=context.run_id,
                task_id=context.task_id,
            )

        if isinstance(context, Mapping):
            return _RouterEventContext(
                run_id=self._normalize_identifier(
                    context.get("run_id")
                ),
                task_id=self._normalize_identifier(
                    context.get("task_id")
                ),
            )

        return self._default_context_provider(request)

    @staticmethod
    def _default_context_provider(
        request: ActionRequest,
    ) -> _RouterEventContext:
        metadata = request.metadata

        run_id = ObservableActionRouter._normalize_identifier(
            metadata.get("run_id")
        )
        task_id = ObservableActionRouter._normalize_identifier(
            metadata.get("task_id")
        )

        return _RouterEventContext(
            run_id=run_id,
            task_id=task_id,
        )

    @staticmethod
    def _normalize_identifier(
        value: Any,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            return None

        normalized = value.strip()
        return normalized or None

    @staticmethod
    def _duration_ms(started: float) -> float:
        return round(
            max(0.0, (perf_counter() - started) * 1000.0),
            3,
        )

    @staticmethod
    def _build_failure_event(
        *,
        request: ActionRequest,
        context: _RouterEventContext,
        duration_ms: float,
        error: str,
    ) -> ObservabilityEvent:
        return ObservabilityEvent(
            event="router.failed",
            source="action_router",
            request_id=request.request_id,
            run_id=context.run_id,
            task_id=context.task_id,
            action_kind=request.kind.value,
            action_name=request.name,
            status="failed",
            duration_ms=duration_ms,
            error=error,
            metadata={
                "arguments_present": bool(request.arguments),
            },
        )

    def _safe_log(
        self,
        event: ObservabilityEvent,
    ) -> None:
        try:
            self._logger.log(event)
        except Exception:
            # Observability must never break routing.
            return
