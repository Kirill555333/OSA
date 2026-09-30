"""Observability wrapper for autonomous action execution."""

from __future__ import annotations

from time import perf_counter

from osa.actions.contracts import ActionRequest
from osa.observability.events import ObservabilityEvent
from osa.observability.logger import (
    InMemoryObservabilityLogger,
    ObservabilityLogger,
)
from osa.tasks.action_bridge import AutonomousActionBridge
from osa.tasks.autonomous import AutonomousTaskResult


class AutonomousActionObservabilityError(RuntimeError):
    """Raised when autonomous action observability is misconfigured."""


class ObservableAutonomousActionBridge:
    """
    Observability wrapper around AutonomousActionBridge.

    The wrapped bridge remains authoritative for resolution, safety,
    routing and execution. This wrapper only emits correlation-safe
    lifecycle events.
    """

    def __init__(
        self,
        bridge: AutonomousActionBridge,
        *,
        logger: ObservabilityLogger | None = None,
    ) -> None:
        if bridge is None:
            raise AutonomousActionObservabilityError(
                "bridge is required"
            )

        self._bridge = bridge
        self._logger = (
            InMemoryObservabilityLogger()
            if logger is None
            else logger
        )

    @property
    def bridge(self) -> AutonomousActionBridge:
        """Return the wrapped autonomous action bridge."""
        return self._bridge

    @property
    def logger(self) -> ObservabilityLogger:
        """Return the configured observability logger."""
        return self._logger

    def execute(
        self,
        run_id: str,
        task_id: str,
        request: ActionRequest,
    ) -> AutonomousTaskResult:
        """Execute one autonomous action and emit lifecycle events."""
        normalized_run_id = self._normalize_identifier(
            run_id,
            "run_id",
        )
        normalized_task_id = self._normalize_identifier(
            task_id,
            "task_id",
        )

        if not isinstance(request, ActionRequest):
            raise AutonomousActionObservabilityError(
                "request must be an ActionRequest"
            )

        started_at = perf_counter()

        self._safe_log(
            ObservabilityEvent(
                event="autonomous.action.started",
                source="autonomous",
                request_id=request.request_id,
                run_id=normalized_run_id,
                task_id=normalized_task_id,
                action_kind=request.kind,
                action_name=request.name,
                status="started",
                duration_ms=0.0,
                metadata={
                    "arguments_present": bool(
                        request.arguments
                    ),
                },
            )
        )

        try:
            result = self._bridge.execute(
                normalized_run_id,
                normalized_task_id,
                request,
            )
        except Exception as exc:
            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.action.failed",
                    source="autonomous",
                    request_id=request.request_id,
                    run_id=normalized_run_id,
                    task_id=normalized_task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=self._duration_ms(
                        started_at
                    ),
                    error=str(exc),
                    metadata={
                        "arguments_present": bool(
                            request.arguments
                        ),
                    },
                )
            )
            raise

        duration_ms = self._duration_ms(
            started_at
        )

        if not isinstance(
            result,
            AutonomousTaskResult,
        ):
            error = (
                "wrapped autonomous bridge returned "
                "invalid task result"
            )

            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.action.failed",
                    source="autonomous",
                    request_id=request.request_id,
                    run_id=normalized_run_id,
                    task_id=normalized_task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=duration_ms,
                    error=error,
                )
            )

            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                error=error,
            )

        if result.task_id != normalized_task_id:
            error = (
                "wrapped autonomous bridge returned "
                "mismatched task_id"
            )

            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.action.failed",
                    source="autonomous",
                    request_id=request.request_id,
                    run_id=normalized_run_id,
                    task_id=normalized_task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=duration_ms,
                    error=error,
                )
            )

            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                output=result.output,
                error=error,
            )

        if result.status == "completed":
            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.action.completed",
                    source="autonomous",
                    request_id=request.request_id,
                    run_id=normalized_run_id,
                    task_id=normalized_task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="completed",
                    duration_ms=duration_ms,
                    metadata={
                        "output_present": result.output is not None,
                    },
                )
            )
            return result

        if result.status == "failed":
            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.action.failed",
                    source="autonomous",
                    request_id=request.request_id,
                    run_id=normalized_run_id,
                    task_id=normalized_task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=duration_ms,
                    error=result.error,
                    metadata={
                        "output_present": result.output is not None,
                    },
                )
            )
            return result

        error = (
            "wrapped autonomous bridge returned "
            f"unsupported task status '{result.status}'"
        )

        self._safe_log(
            ObservabilityEvent(
                event="autonomous.action.failed",
                source="autonomous",
                request_id=request.request_id,
                run_id=normalized_run_id,
                task_id=normalized_task_id,
                action_kind=request.kind,
                action_name=request.name,
                status="failed",
                duration_ms=duration_ms,
                error=error,
            )
        )

        return AutonomousTaskResult(
            task_id=normalized_task_id,
            status="failed",
            output=result.output,
            error=error,
        )

    @staticmethod
    def _normalize_identifier(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized

    @staticmethod
    def _duration_ms(
        started_at: float,
    ) -> float:
        return round(
            max(
                0.0,
                (perf_counter() - started_at) * 1000.0,
            ),
            3,
        )

    def _safe_log(
        self,
        event: ObservabilityEvent,
    ) -> None:
        try:
            self._logger.log(event)
        except Exception:
            return


__all__ = [
    "AutonomousActionObservabilityError",
    "ObservableAutonomousActionBridge",
]
