"""Agent-level action observability helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from osa.actions.contracts import ActionRequest, ActionResult
from osa.core.execution_context import ExecutionContext
from osa.recovery_contracts import RecoveryResult


class AgentObservability:
    """Emit correlation-safe telemetry for one Agent action lifecycle."""

    def __init__(
        self,
        logger: Callable[..., None] | None,
    ) -> None:
        self._logger = logger

    def action_requested(
        self,
        request: ActionRequest,
        *,
        context: ExecutionContext | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit the start of one unified action."""
        self._emit(
            "action.requested",
            **self._correlation_data(
                request,
                context=context,
                round_number=round_number,
            ),
            action_kind=request.kind.value,
            action_name=request.name,
            status="requested",
        )

    def action_completed(
        self,
        request: ActionRequest,
        result: ActionResult,
        *,
        recovery: RecoveryResult | None = None,
        context: ExecutionContext | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit successful or failed action completion."""
        self._emit(
            "action.completed",
            **self._correlation_data(
                request,
                context=context,
                round_number=round_number,
            ),
            action_kind=request.kind.value,
            action_name=request.name,
            status=(
                "completed"
                if result.success
                else "failed"
            ),
            success=result.success,
            output_length=len(result.output),
            error=result.error,
            recovery_attempts=(
                recovery.attempt_count
                if recovery is not None
                else None
            ),
            recovery_failure_kind=(
                recovery.failure_kind.value
                if (
                    recovery is not None
                    and recovery.failure_kind is not None
                )
                else None
            ),
        )

    def action_denied(
        self,
        request: ActionRequest,
        *,
        decision: str,
        reason: str | None = None,
        context: ExecutionContext | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit one denied action."""
        self._emit(
            "action.denied",
            **self._correlation_data(
                request,
                context=context,
                round_number=round_number,
            ),
            action_kind=request.kind.value,
            action_name=request.name,
            status="denied",
            decision=decision,
            reason=reason,
        )

    def recovery_started(
        self,
        request: ActionRequest,
        *,
        max_attempts: int,
        context: ExecutionContext | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit the start of recovery for one action."""
        self._emit(
            "action.recovery.started",
            **self._correlation_data(
                request,
                context=context,
                round_number=round_number,
            ),
            action_kind=request.kind.value,
            action_name=request.name,
            status="started",
            max_attempts=max_attempts,
        )

    def recovery_completed(
        self,
        request: ActionRequest,
        recovery: RecoveryResult,
        *,
        context: ExecutionContext | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit recovery completion."""
        self._emit(
            "action.recovery.completed",
            **self._correlation_data(
                request,
                context=context,
                round_number=round_number,
            ),
            action_kind=request.kind.value,
            action_name=request.name,
            status=(
                "completed"
                if recovery.success
                else "failed"
            ),
            attempt=(
                recovery.attempt_count
                if recovery.attempt_count > 0
                else None
            ),
            max_attempts=(
                recovery.last_attempt.max_attempts
                if recovery.last_attempt is not None
                else None
            ),
            error=recovery.error,
            failure_kind=(
                recovery.failure_kind.value
                if recovery.failure_kind is not None
                else None
            ),
        )

    @staticmethod
    def _correlation_data(
        request: ActionRequest,
        *,
        context: ExecutionContext | None,
        round_number: int | None,
    ) -> dict[str, Any]:
        """Build telemetry correlation data without action arguments."""
        if context is not None:
            return {
                "request_id": context.request_id,
                "run_id": context.run_id,
                "task_id": context.task_id,
                "round": context.round_number,
                "source": context.source,
            }

        return {
            "request_id": request.request_id,
            "round": round_number,
        }

    def _emit(
        self,
        event: str,
        **data: Any,
    ) -> None:
        """Emit one event while omitting absent optional fields."""
        if self._logger is None:
            return

        safe_data = {
            key: value
            for key, value in data.items()
            if value is not None
        }

        self._logger(
            event,
            **safe_data,
        )


__all__ = [
    "AgentObservability",
]
