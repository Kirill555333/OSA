from __future__ import annotations

import time
from typing import Any

from osa.recovery_contracts import (
    RecoveryRequest,
    RecoveryResult,
)
from osa.recovery_controller import RecoveryController

from osa.observability.logger import (
    ObservabilityLogger,
    NullObservabilityLogger,
)


class ObservableRecoveryControllerError(RuntimeError):
    """Raised when observable recovery integration is invalid."""


class ObservableRecoveryController:
    """
    Observe unified recovery execution without changing its semantics.

    Observability failures are isolated from recovery execution: a logging
    failure must never turn a recovery success into a failure or vice versa.
    """

    def __init__(
        self,
        controller: RecoveryController,
        *,
        logger: ObservabilityLogger | None = None,
    ) -> None:
        if controller is None:
            raise ValueError("controller is required.")

        self._controller = controller
        self._logger = (
            logger
            if logger is not None
            else NullObservabilityLogger()
        )

    @property
    def controller(self) -> RecoveryController:
        return self._controller

    @property
    def logger(self) -> ObservabilityLogger:
        return self._logger

    def execute(
        self,
        request: RecoveryRequest,
    ) -> RecoveryResult:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise TypeError(
                "request must be a RecoveryRequest."
            )

        started_at = time.perf_counter()

        self._safe_log(
            "recovery.started",
            source="recovery",
            run_id=request.run_id,
            task_id=request.task_id,
            request_id=request.request_id,
            action_kind=self._enum_value(
                request.action_kind
            ),
            action_name=request.action_name,
            status="started",
            attempt=request.attempt,
            max_attempts=request.max_attempts,
            metadata={
                "metadata_present": bool(
                    request.metadata
                ),
            },
        )

        try:
            result = self._controller.execute(
                request
            )
        except Exception as exc:
            self._safe_log(
                "recovery.failed",
                source="recovery",
                run_id=request.run_id,
                task_id=request.task_id,
                request_id=request.request_id,
                action_kind=self._enum_value(
                    request.action_kind
                ),
                action_name=request.action_name,
                status="failed",
                attempt=request.attempt,
                max_attempts=request.max_attempts,
                duration_ms=self._duration_ms(
                    started_at
                ),
                error=str(exc),
                metadata={
                    "controller_exception": True,
                },
            )
            raise

        if not isinstance(
            result,
            RecoveryResult,
        ):
            self._safe_log(
                "recovery.failed",
                source="recovery",
                run_id=request.run_id,
                task_id=request.task_id,
                request_id=request.request_id,
                action_kind=self._enum_value(
                    request.action_kind
                ),
                action_name=request.action_name,
                status="failed",
                attempt=request.attempt,
                max_attempts=request.max_attempts,
                duration_ms=self._duration_ms(
                    started_at
                ),
                error=(
                    "RecoveryController returned "
                    "an invalid result."
                ),
                metadata={
                    "invalid_result": True,
                },
            )
            raise ObservableRecoveryControllerError(
                "RecoveryController returned "
                "an invalid result."
            )

        self._log_attempts(
            request,
            result,
        )

        duration_ms = self._duration_ms(
            started_at
        )

        if result.success:
            self._safe_log(
                "recovery.completed",
                source="recovery",
                run_id=request.run_id,
                task_id=request.task_id,
                request_id=request.request_id,
                action_kind=self._enum_value(
                    request.action_kind
                ),
                action_name=request.action_name,
                status="completed",
                attempt=self._result_attempt(
                    result,
                    request.attempt,
                ),
                max_attempts=self._result_max_attempts(
                    result,
                    request.max_attempts,
                ),
                duration_ms=duration_ms,
                metadata={
                    "attempt_count": result.attempt_count,
                    "exhausted": False,
                },
            )
            return result

        if result.exhausted:
            self._safe_log(
                "recovery.exhausted",
                source="recovery",
                run_id=request.run_id,
                task_id=request.task_id,
                request_id=request.request_id,
                action_kind=self._enum_value(
                    request.action_kind
                ),
                action_name=request.action_name,
                status="exhausted",
                attempt=self._result_attempt(
                    result,
                    request.attempt,
                ),
                max_attempts=self._result_max_attempts(
                    result,
                    request.max_attempts,
                ),
                duration_ms=duration_ms,
                error=result.error,
                metadata={
                    "attempt_count": result.attempt_count,
                    "exhausted": True,
                },
            )
            return result

        self._safe_log(
            "recovery.failed",
            source="recovery",
            run_id=request.run_id,
            task_id=request.task_id,
            request_id=request.request_id,
            action_kind=self._enum_value(
                request.action_kind
            ),
            action_name=request.action_name,
            status="failed",
            attempt=self._result_attempt(
                result,
                request.attempt,
            ),
            max_attempts=self._result_max_attempts(
                result,
                request.max_attempts,
            ),
            duration_ms=duration_ms,
            error=result.error,
            metadata={
                "attempt_count": result.attempt_count,
                "exhausted": False,
            },
        )

        return result

    def _log_attempts(
        self,
        request: RecoveryRequest,
        result: RecoveryResult,
    ) -> None:
        attempts = result.attempts

        for index, attempt in enumerate(
            attempts
        ):
            is_last = index == len(attempts) - 1

            if attempt.success:
                decision = "success"
                status = "completed"
            elif is_last:
                decision = "stop"
                status = "failed"
            else:
                decision = "retry"
                status = "failed"

            self._safe_log(
                "recovery.attempt",
                source="recovery",
                run_id=request.run_id,
                task_id=request.task_id,
                request_id=request.request_id,
                action_kind=self._enum_value(
                    request.action_kind
                ),
                action_name=request.action_name,
                status=status,
                attempt=attempt.attempt,
                max_attempts=attempt.max_attempts,
                decision=decision,
                error=attempt.error,
                metadata={
                    "output_present": (
                        attempt.output_present
                    ),
                    "attempt_metadata_present": bool(
                        attempt.metadata
                    ),
                },
            )

    def _safe_log(
        self,
        event: str,
        **data: Any,
    ) -> None:
        try:
            self._logger.log(
                event,
                **data,
            )
        except Exception:
            # Observability is strictly non-authoritative.
            return

    @staticmethod
    def _enum_value(
        value: Any,
    ) -> Any:
        return getattr(
            value,
            "value",
            value,
        )

    @staticmethod
    def _result_attempt(
        result: RecoveryResult,
        fallback: int,
    ) -> int:
        if result.last_attempt is None:
            return fallback

        return result.last_attempt.attempt

    @staticmethod
    def _result_max_attempts(
        result: RecoveryResult,
        fallback: int,
    ) -> int:
        if result.last_attempt is None:
            return fallback

        return result.last_attempt.max_attempts

    @staticmethod
    def _duration_ms(
        started_at: float,
    ) -> float:
        return round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )
