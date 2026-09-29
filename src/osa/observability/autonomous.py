from __future__ import annotations

from time import perf_counter
from typing import Any

from osa.observability.context import ObservabilityContext
from osa.observability.events import ObservabilityEvent
from osa.observability.logger import (
    InMemoryObservabilityLogger,
    ObservabilityLogger,
)
from osa.tasks.autonomous import (
    AutonomousLoop,
    AutonomousRunReport,
    AutonomousTaskResult,
)


class AutonomousObservabilityError(RuntimeError):
    """Raised when autonomous observability is configured incorrectly."""


class ObservableAutonomousLoop:
    """
    Observability wrapper around AutonomousLoop.

    The wrapped loop remains authoritative:
    - run semantics are unchanged;
    - task execution is unchanged;
    - safety is unchanged;
    - report values are returned unchanged;
    - logging failures never break autonomous execution.
    """

    def __init__(
        self,
        loop: AutonomousLoop,
        *,
        logger: ObservabilityLogger | None = None,
        context_provider: Any | None = None,
    ) -> None:
        if loop is None:
            raise AutonomousObservabilityError(
                "loop is required"
            )

        if logger is None:
            logger = InMemoryObservabilityLogger()

        self._loop = loop
        self._logger = logger
        self._context_provider = (
            context_provider
            if context_provider is not None
            else self._default_context_provider
        )

    @property
    def loop(self) -> AutonomousLoop:
        return self._loop

    @property
    def logger(self) -> ObservabilityLogger:
        return self._logger

    @property
    def context_provider(self) -> Any:
        return self._context_provider

    @property
    def config(self):
        return self._loop.config

    def stop(self) -> None:
        self._loop.stop()

    def reset(self) -> None:
        self._loop.reset()

    def run(self, goal: str) -> AutonomousRunReport:
        normalized_goal = goal.strip()

        if not normalized_goal:
            raise ValueError(
                "Autonomous goal cannot be empty."
            )

        started = perf_counter()

        context = self._resolve_context()

        self._safe_log(
            ObservabilityEvent(
                event="autonomous.run.started",
                source="autonomous",
                run_id=context.run_id,
                task_id=context.task_id,
                status="started",
                duration_ms=0.0,
                metadata={
                    "goal_length": len(normalized_goal),
                },
            )
        )

        try:
            report = self._loop.run(normalized_goal)
        except Exception as exc:
            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.run.failed",
                    source="autonomous",
                    run_id=context.run_id,
                    task_id=context.task_id,
                    status="failed",
                    duration_ms=self._duration_ms(started),
                    error=str(exc),
                )
            )
            raise

        if not isinstance(report, AutonomousRunReport):
            error = (
                "wrapped autonomous loop returned "
                "invalid report"
            )

            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.run.failed",
                    source="autonomous",
                    run_id=context.run_id,
                    task_id=context.task_id,
                    status="failed",
                    duration_ms=self._duration_ms(started),
                    error=error,
                )
            )

            raise AutonomousObservabilityError(error)

        self._log_report_events(
            report,
            started,
            context,
        )

        return report

    def _log_report_events(
        self,
        report: AutonomousRunReport,
        started: float,
        context: ObservabilityContext,
    ) -> None:
        for cycle in report.cycles:
            for result in cycle.results:
                self._log_task_event(
                    report=report,
                    result=result,
                    context=context,
                )

            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.cycle.completed",
                    source="autonomous",
                    run_id=report.run_id,
                    task_id=context.task_id,
                    status=(
                        "completed"
                        if cycle.progressed
                        else "failed"
                    ),
                    metadata={
                        "cycle": cycle.cycle,
                        "ready_task_count": len(
                            cycle.ready_task_ids
                        ),
                        "result_count": len(
                            cycle.results
                        ),
                        "completed_task_count": sum(
                            result.status == "completed"
                            for result in cycle.results
                        ),
                        "failed_task_count": sum(
                            result.status == "failed"
                            for result in cycle.results
                        ),
                        "progressed": cycle.progressed,
                    },
                )
            )

        duration_ms = self._duration_ms(started)

        if report.success:
            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.run.completed",
                    source="autonomous",
                    run_id=report.run_id,
                    task_id=context.task_id,
                    status="completed",
                    duration_ms=duration_ms,
                    metadata={
                        "cycle_count": report.cycle_count,
                        "completed_task_count": len(
                            report.completed_task_ids
                        ),
                        "failed_task_count": len(
                            report.failed_task_ids
                        ),
                        "stop_reason": report.stop_reason,
                    },
                )
            )
            return

        self._safe_log(
            ObservabilityEvent(
                event="autonomous.run.failed",
                source="autonomous",
                run_id=report.run_id,
                task_id=context.task_id,
                status="failed",
                duration_ms=duration_ms,
                error=report.stop_reason,
                metadata={
                    "cycle_count": report.cycle_count,
                    "completed_task_count": len(
                        report.completed_task_ids
                    ),
                    "failed_task_count": len(
                        report.failed_task_ids
                    ),
                    "stopped": report.stopped,
                    "stop_reason": report.stop_reason,
                },
            )
        )

    def _log_task_event(
        self,
        *,
        report: AutonomousRunReport,
        result: AutonomousTaskResult,
        context: ObservabilityContext,
    ) -> None:
        status = result.status.strip().lower()

        if status == "completed":
            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.task.completed",
                    source="autonomous",
                    run_id=report.run_id,
                    task_id=result.task_id,
                    status="completed",
                    metadata={
                        "has_output": result.output is not None,
                    },
                )
            )
            return

        if status == "failed":
            self._safe_log(
                ObservabilityEvent(
                    event="autonomous.task.failed",
                    source="autonomous",
                    run_id=report.run_id,
                    task_id=result.task_id,
                    status="failed",
                    error=result.error,
                    metadata={
                        "has_output": result.output is not None,
                    },
                )
            )
            return

        self._safe_log(
            ObservabilityEvent(
                event="autonomous.task.failed",
                source="autonomous",
                run_id=report.run_id,
                task_id=result.task_id,
                status="failed",
                error=(
                    result.error
                    or f"unexpected task status: {result.status}"
                ),
            )
        )

    def _resolve_context(self) -> ObservabilityContext:
        try:
            context = self._context_provider()
        except Exception:
            context = None

        if isinstance(context, ObservabilityContext):
            return context

        return ObservabilityContext()

    @staticmethod
    def _default_context_provider() -> ObservabilityContext:
        return ObservabilityContext()

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
            # Observability must never break autonomous execution.
            return
