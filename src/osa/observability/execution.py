from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.observability.context import ObservabilityContext
from osa.observability.events import ObservabilityEvent
from osa.observability.logger import (
    InMemoryObservabilityLogger,
    ObservabilityLogger,
)


class ActionExecutionObservabilityError(RuntimeError):
    """Raised when execution observability is configured incorrectly."""


class ActionExecutionHandler(Protocol):
    """Minimal action handler interface."""

    def execute(self, request: ActionRequest) -> ActionResult:
        """Execute an action request."""
        ...


OBSERVABLE_EXECUTION_SOURCES = (
    "browser",
    "desktop",
    "tool",
)


ContextProvider = Any


@dataclass(frozen=True, slots=True)
class _ExecutionEventContext:
    run_id: str | None = None
    task_id: str | None = None


class ObservableActionExecution:
    """
    Observability wrapper around a concrete action handler.

    The wrapped handler remains authoritative. This wrapper only records
    execution lifecycle events and otherwise preserves execution semantics.
    """

    def __init__(
        self,
        handler: ActionExecutionHandler,
        *,
        source: str | None = None,
        logger: ObservabilityLogger | None = None,
        context_provider: ContextProvider | None = None,
    ) -> None:
        if handler is None:
            raise ActionExecutionObservabilityError(
                "handler is required"
            )

        normalized_source = self._normalize_source(source)

        self._handler = handler
        self._source = normalized_source
        self._logger = (
            InMemoryObservabilityLogger()
            if logger is None
            else logger
        )
        self._context_provider = (
            context_provider
            if context_provider is not None
            else self._default_context_provider
        )

    @property
    def handler(self) -> ActionExecutionHandler:
        return self._handler

    @property
    def source(self) -> str:
        return self._source

    @property
    def logger(self) -> ObservabilityLogger:
        return self._logger

    @property
    def context_provider(self) -> ContextProvider:
        return self._context_provider

    @classmethod
    def browser(
        cls,
        handler: ActionExecutionHandler,
        *,
        logger: ObservabilityLogger | None = None,
        context_provider: ContextProvider | None = None,
    ) -> ObservableActionExecution:
        return cls(
            handler,
            source="browser",
            logger=logger,
            context_provider=context_provider,
        )

    @classmethod
    def desktop(
        cls,
        handler: ActionExecutionHandler,
        *,
        logger: ObservabilityLogger | None = None,
        context_provider: ContextProvider | None = None,
    ) -> ObservableActionExecution:
        return cls(
            handler,
            source="desktop",
            logger=logger,
            context_provider=context_provider,
        )

    @classmethod
    def tool(
        cls,
        handler: ActionExecutionHandler,
        *,
        logger: ObservabilityLogger | None = None,
        context_provider: ContextProvider | None = None,
    ) -> ObservableActionExecution:
        return cls(
            handler,
            source="tool",
            logger=logger,
            context_provider=context_provider,
        )

    def execute(self, request: ActionRequest) -> ActionResult:
        if not isinstance(request, ActionRequest):
            raise ActionExecutionObservabilityError(
                "request must be an ActionRequest"
            )

        started = perf_counter()
        context = self._resolve_context(request)

        source = self._source_for_request(request)

        self._safe_log(
            ObservabilityEvent(
                event="execution.started",
                source=source,
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
            result = self._handler.execute(request)
        except Exception as exc:
            self._safe_log(
                self._build_failed_event(
                    request=request,
                    source=source,
                    context=context,
                    duration_ms=self._duration_ms(started),
                    error=str(exc),
                )
            )
            raise

        duration_ms = self._duration_ms(started)

        if not isinstance(result, ActionResult):
            error = "wrapped handler returned invalid result"

            self._safe_log(
                self._build_failed_event(
                    request=request,
                    source=source,
                    context=context,
                    duration_ms=duration_ms,
                    error=error,
                )
            )

            return ActionResult.failed(
                request.request_id,
                error,
            )

        if result.request_id != request.request_id:
            error = "wrapped handler returned mismatched request_id"

            self._safe_log(
                self._build_failed_event(
                    request=request,
                    source=source,
                    context=context,
                    duration_ms=duration_ms,
                    error=error,
                )
            )

            return ActionResult.failed(
                request.request_id,
                error,
            )

        if result.success:
            self._safe_log(
                ObservabilityEvent(
                    event="execution.completed",
                    source=source,
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
                self._build_failed_event(
                    request=request,
                    source=source,
                    context=context,
                    duration_ms=duration_ms,
                    error=result.error or "action execution failed",
                )
            )

        return result

    def _resolve_context(
        self,
        request: ActionRequest,
    ) -> _ExecutionEventContext:
        try:
            context = self._context_provider(request)
        except Exception:
            context = None

        if isinstance(context, ObservabilityContext):
            return _ExecutionEventContext(
                run_id=context.run_id,
                task_id=context.task_id,
            )

        if isinstance(context, Mapping):
            return _ExecutionEventContext(
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
    ) -> _ExecutionEventContext:
        metadata = request.metadata

        return _ExecutionEventContext(
            run_id=ObservableActionExecution._normalize_identifier(
                metadata.get("run_id")
            ),
            task_id=ObservableActionExecution._normalize_identifier(
                metadata.get("task_id")
            ),
        )

    def _source_for_request(
        self,
        request: ActionRequest,
    ) -> str:
        request_source = request.kind.value

        if request_source in OBSERVABLE_EXECUTION_SOURCES:
            return request_source

        return self._source

    @staticmethod
    def _normalize_source(
        source: str | None,
    ) -> str:
        if source is None:
            raise ActionExecutionObservabilityError(
                "source is required"
            )

        if not isinstance(source, str):
            raise ActionExecutionObservabilityError(
                "source must be a string"
            )

        normalized = source.strip().lower()

        if normalized not in OBSERVABLE_EXECUTION_SOURCES:
            raise ActionExecutionObservabilityError(
                "source must be one of: "
                + ", ".join(OBSERVABLE_EXECUTION_SOURCES)
            )

        return normalized

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
    def _duration_ms(
        started: float,
    ) -> float:
        return round(
            max(0.0, (perf_counter() - started) * 1000.0),
            3,
        )

    @staticmethod
    def _build_failed_event(
        *,
        request: ActionRequest,
        source: str,
        context: _ExecutionEventContext,
        duration_ms: float,
        error: str,
    ) -> ObservabilityEvent:
        return ObservabilityEvent(
            event="execution.failed",
            source=source,
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
            # Observability must never break action execution.
            return
