"""Agent-level action observability helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from osa.actions.contracts import ActionRequest, ActionResult
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
        round_number: int | None = None,
    ) -> None:
        """Emit the start of one unified action."""
        self._emit(
            "action.requested",
            request_id=request.request_id,
            action_kind=request.kind.value,
            action_name=request.name,
            status="requested",
            round=round_number,
        )

    def action_completed(
        self,
        request: ActionRequest,
        result: ActionResult,
        *,
        recovery: RecoveryResult | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit successful or failed action completion."""
        self._emit(
            "action.completed",
            request_id=request.request_id,
            action_kind=request.kind.value,
            action_name=request.name,
            status=(
                "completed"
                if result.success
                else "failed"
            ),
            round=round_number,
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
                if recovery is not None
                and recovery.failure_kind is not None
                else None
            ),
        )

    def action_denied(
        self,
        request: ActionRequest,
        *,
        decision: str,
        reason: str | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit a safety/permission/confirmation denial."""
        self._emit(
            "action.denied",
            request_id=request.request_id,
            action_kind=request.kind.value,
            action_name=request.name,
            status="denied",
            decision=decision,
            reason=reason,
            round=round_number,
        )

    def recovery_started(
        self,
        request: ActionRequest,
        *,
        max_attempts: int,
        round_number: int | None = None,
    ) -> None:
        """Emit the start of unified action recovery."""
        self._emit(
            "action.recovery.started",
            request_id=request.request_id,
            action_kind=request.kind.value,
            action_name=request.name,
            status="started",
            max_attempts=max_attempts,
            round=round_number,
        )

    def recovery_completed(
        self,
        request: ActionRequest,
        recovery: RecoveryResult,
        *,
        round_number: int | None = None,
    ) -> None:
        """Emit the result of unified action recovery."""
        self._emit(
            "action.recovery.completed",
            request_id=request.request_id,
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
            round=round_number,
        )

    def _emit(
        self,
        event: str,
        **data: Any,
    ) -> None:
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
