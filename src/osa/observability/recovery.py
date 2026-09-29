from __future__ import annotations

from time import perf_counter

from osa.observability.events import ObservabilityEvent
from osa.observability.logger import (
    InMemoryObservabilityLogger,
    ObservabilityLogger,
)
from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_recovery import AutonomousRecoveryExecutor


class RecoveryObservabilityError(RuntimeError):
    """Raised when recovery observability is misconfigured."""


class ObservableAutonomousRecovery:
    """
    Observability wrapper around AutonomousRecoveryExecutor.

    Recovery semantics remain owned by the wrapped executor:
    - retry policy is unchanged;
    - attempt limits are unchanged;
    - task execution is unchanged;
    - returned task results are unchanged;
    - logger failures never break recovery.
    """

    def __init__(
        self,
        executor: AutonomousRecoveryExecutor,
        *,
        logger: ObservabilityLogger | None = None,
    ) -> None:
        if executor is None:
            raise RecoveryObservabilityError(
                "executor is required"
            )

        if logger is None:
            logger = InMemoryObservabilityLogger()

        self._executor = executor
        self._logger = logger

    @property
    def executor(self) -> AutonomousRecoveryExecutor:
        return self._executor

    @property
    def logger(self) -> ObservabilityLogger:
        return self._logger

    @property
    def policy(self):
        return self._executor.policy

    @property
    def max_attempts(self) -> int:
        return self._executor.max_attempts

    def execute(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        started = perf_counter()

        self._safe_log(
            ObservabilityEvent(
                event="recovery.started",
                source="autonomous_recovery",
                run_id=run_id,
                task_id=task_id,
                status="started",
                duration_ms=0.0,
                max_attempts=self._executor.max_attempts,
                metadata={
                    "recovery_enabled": (
                        self._executor.max_attempts > 1
                    ),
                },
            )
        )

        try:
            result = self._executor.execute(
                run_id,
                task_id,
            )
        except Exception as exc:
            self._safe_log(
                ObservabilityEvent(
                    event="recovery.failed",
                    source="autonomous_recovery",
                    run_id=run_id,
                    task_id=task_id,
                    status="failed",
                    duration_ms=self._duration_ms(started),
                    max_attempts=self._executor.max_attempts,
                    error=str(exc),
                )
            )
            raise

        duration_ms = self._duration_ms(started)

        if not isinstance(result, AutonomousTaskResult):
            error = "recovery executor returned invalid result"

            self._safe_log(
                ObservabilityEvent(
                    event="recovery.failed",
                    source="autonomous_recovery",
                    run_id=run_id,
                    task_id=task_id,
                    status="failed",
                    duration_ms=duration_ms,
                    max_attempts=self._executor.max_attempts,
                    error=error,
                )
            )

            raise RecoveryObservabilityError(error)

        if result.status == "completed":
            self._safe_log(
                ObservabilityEvent(
                    event="recovery.completed",
                    source="autonomous_recovery",
                    run_id=run_id,
                    task_id=result.task_id,
                    status="completed",
                    duration_ms=duration_ms,
                    max_attempts=self._executor.max_attempts,
                    metadata={
                        "has_output": result.output is not None,
                    },
                )
            )
            return result

        if result.status == "failed":
            self._safe_log(
                self._failure_event(
                    run_id=run_id,
                    result=result,
                    duration_ms=duration_ms,
                )
            )
            return result

        normalized_result = AutonomousTaskResult(
            task_id=result.task_id,
            status="failed",
            output=result.output,
            error=(
                "recovery_observability_invalid_status: "
                f"unsupported task status '{result.status}'."
            ),
        )

        self._safe_log(
            self._failure_event(
                run_id=run_id,
                result=normalized_result,
                duration_ms=duration_ms,
            )
        )

        return normalized_result

    def _failure_event(
        self,
        *,
        run_id: str,
        result: AutonomousTaskResult,
        duration_ms: float,
    ) -> ObservabilityEvent:
        error = result.error or "autonomous recovery failed"

        if error.startswith(
            "autonomous_recovery_policy_failed:"
        ):
            event_name = "recovery.failed"
        else:
            event_name = "recovery.exhausted"

        return ObservabilityEvent(
            event=event_name,
            source="autonomous_recovery",
            run_id=run_id,
            task_id=result.task_id,
            status="failed",
            duration_ms=duration_ms,
            max_attempts=self._executor.max_attempts,
            error=error,
            metadata={
                "has_output": result.output is not None,
                "reason_present": bool(result.error),
            },
        )

    @staticmethod
    def _duration_ms(started: float) -> float:
        return round(
            max(
                0.0,
                (perf_counter() - started) * 1000.0,
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
            # Observability must never break recovery.
            return
