from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from threading import Event
from typing import Protocol


class AutonomousLoopError(RuntimeError):
    """Raised when the autonomous loop cannot continue safely."""


@dataclass(frozen=True)
class AutonomousLoopConfig:
    max_cycles: int = 16
    max_tasks_per_cycle: int = 8

    def __post_init__(self) -> None:
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be at least 1.")
        if self.max_tasks_per_cycle < 1:
            raise ValueError("max_tasks_per_cycle must be at least 1.")


@dataclass(frozen=True)
class AutonomousTaskResult:
    task_id: str
    status: str
    output: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class AutonomousCycle:
    cycle: int
    ready_task_ids: tuple[str, ...]
    results: tuple[AutonomousTaskResult, ...]
    progressed: bool


@dataclass(frozen=True)
class AutonomousRunReport:
    goal: str
    run_id: str
    success: bool
    stopped: bool
    stop_reason: str | None
    cycles: tuple[AutonomousCycle, ...] = field(default_factory=tuple)

    @property
    def cycle_count(self) -> int:
        return len(self.cycles)

    @property
    def completed_task_ids(self) -> tuple[str, ...]:
        completed: list[str] = []

        for cycle in self.cycles:
            for result in cycle.results:
                if (
                    result.status == "completed"
                    and result.task_id not in completed
                ):
                    completed.append(result.task_id)

        return tuple(completed)

    @property
    def failed_task_ids(self) -> tuple[str, ...]:
        failed: list[str] = []

        for cycle in self.cycles:
            for result in cycle.results:
                if (
                    result.status == "failed"
                    and result.task_id not in failed
                ):
                    failed.append(result.task_id)

        return tuple(failed)


class AutonomousTaskAuthorizer(Protocol):
    """Authorize a task before it reaches the execution pipeline."""

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ):
        ...


class AutonomousBackend(Protocol):
    """Minimal adapter between the loop and the existing task pipeline."""

    def create_run(self, goal: str) -> str:
        ...

    def ready_task_ids(self, run_id: str) -> Sequence[str]:
        ...

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        ...

    def is_complete(self, run_id: str) -> bool:
        ...

    def has_failed(self, run_id: str) -> bool:
        ...


class CallbackAutonomousBackend:
    """
    Small adapter for wiring the autonomous loop to existing OSA services.

    The callbacks are intentionally injected so the autonomous layer does not
    duplicate Planner/TaskManager/TaskExecutor implementation details.
    """

    def __init__(
        self,
        *,
        create_run: Callable[[str], str],
        ready_task_ids: Callable[[str], Sequence[str]],
        execute_task: Callable[[str, str], AutonomousTaskResult],
        is_complete: Callable[[str], bool],
        has_failed: Callable[[str], bool],
    ) -> None:
        self._create_run = create_run
        self._ready_task_ids = ready_task_ids
        self._execute_task = execute_task
        self._is_complete = is_complete
        self._has_failed = has_failed

    def create_run(self, goal: str) -> str:
        return self._create_run(goal)

    def ready_task_ids(self, run_id: str) -> Sequence[str]:
        return self._ready_task_ids(run_id)

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        return self._execute_task(run_id, task_id)

    def is_complete(self, run_id: str) -> bool:
        return self._is_complete(run_id)

    def has_failed(self, run_id: str) -> bool:
        return self._has_failed(run_id)


class AutonomousLoop:
    """
    Deterministic orchestration layer for autonomous task execution.

    The loop does not decide what a task means and does not execute tools
    itself. It repeatedly asks the backend for ready work and delegates the
    actual task execution to the existing execution pipeline.
    """

    def __init__(
        self,
        backend: AutonomousBackend,
        *,
        config: AutonomousLoopConfig | None = None,
        authorizer: AutonomousTaskAuthorizer | None = None,
    ) -> None:
        self._backend = backend
        self._config = config or AutonomousLoopConfig()
        self._authorizer = authorizer
        self._stop_event = Event()

    @property
    def config(self) -> AutonomousLoopConfig:
        return self._config

    def stop(self) -> None:
        self._stop_event.set()

    def reset(self) -> None:
        self._stop_event.clear()

    def run(self, goal: str) -> AutonomousRunReport:
        normalized_goal = goal.strip()

        if not normalized_goal:
            raise ValueError(
                "Autonomous goal cannot be empty."
            )

        self.reset()

        run_id = self._backend.create_run(normalized_goal)
        cycles: list[AutonomousCycle] = []

        for cycle_number in range(
            1,
            self._config.max_cycles + 1,
        ):
            if self._stop_event.is_set():
                return AutonomousRunReport(
                    goal=normalized_goal,
                    run_id=run_id,
                    success=False,
                    stopped=True,
                    stop_reason="stop_requested",
                    cycles=tuple(cycles),
                )

            if self._backend.is_complete(run_id):
                return AutonomousRunReport(
                    goal=normalized_goal,
                    run_id=run_id,
                    success=True,
                    stopped=False,
                    stop_reason=None,
                    cycles=tuple(cycles),
                )

            if self._backend.has_failed(run_id):
                return AutonomousRunReport(
                    goal=normalized_goal,
                    run_id=run_id,
                    success=False,
                    stopped=False,
                    stop_reason="task_failed",
                    cycles=tuple(cycles),
                )

            ready_ids = tuple(
                self._backend.ready_task_ids(run_id)[
                    : self._config.max_tasks_per_cycle
                ]
            )

            if not ready_ids:
                return AutonomousRunReport(
                    goal=normalized_goal,
                    run_id=run_id,
                    success=False,
                    stopped=False,
                    stop_reason="no_ready_tasks",
                    cycles=tuple(cycles),
                )

            results: list[AutonomousTaskResult] = []

            for task_id in ready_ids:
                if self._stop_event.is_set():
                    return AutonomousRunReport(
                        goal=normalized_goal,
                        run_id=run_id,
                        success=False,
                        stopped=True,
                        stop_reason="stop_requested",
                        cycles=tuple(cycles),
                    )

                if self._authorizer is not None:
                    authorization = self._authorizer.authorize(
                        run_id,
                        task_id,
                    )

                    if not authorization.allowed:
                        results.append(
                            AutonomousTaskResult(
                                task_id=task_id,
                                status="failed",
                                error=(
                                    authorization.reason
                                    or "autonomous_task_denied"
                                ),
                            )
                        )
                        continue

                result = self._backend.execute_task(
                    run_id,
                    task_id,
                )
                results.append(result)

            progressed = any(
                result.status in {"completed", "failed"}
                for result in results
            )

            cycles.append(
                AutonomousCycle(
                    cycle=cycle_number,
                    ready_task_ids=ready_ids,
                    results=tuple(results),
                    progressed=progressed,
                )
            )

            if not progressed:
                return AutonomousRunReport(
                    goal=normalized_goal,
                    run_id=run_id,
                    success=False,
                    stopped=False,
                    stop_reason="no_progress",
                    cycles=tuple(cycles),
                )

        complete = self._backend.is_complete(run_id)

        return AutonomousRunReport(
            goal=normalized_goal,
            run_id=run_id,
            success=complete,
            stopped=False,
            stop_reason=(
                None
                if complete
                else "max_cycles_reached"
            ),
            cycles=tuple(cycles),
        )
