from __future__ import annotations

import pytest

from osa.observability import (
    InMemoryObservabilityLogger,
    ObservableAutonomousLoop,
    ObservabilityContext,
)
from osa.tasks.autonomous import (
    AutonomousCycle,
    AutonomousLoop,
    AutonomousLoopConfig,
    AutonomousRunReport,
    AutonomousTaskResult,
)


class _Backend:
    def __init__(
        self,
        *,
        results: list[AutonomousTaskResult],
        complete_after_execute: bool = True,
    ) -> None:
        self.results = list(results)
        self.complete_after_execute = complete_after_execute
        self.execute_calls: list[tuple[str, str]] = []
        self.created_goal: str | None = None
        self._complete = False
        self._failed = False

    def create_run(self, goal: str) -> str:
        self.created_goal = goal
        return "run-001"

    def ready_task_ids(self, run_id: str) -> tuple[str, ...]:
        if self._complete or self._failed:
            return ()

        return tuple(
            result.task_id
            for result in self.results
        )

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.execute_calls.append(
            (run_id, task_id)
        )

        result = next(
            result
            for result in self.results
            if result.task_id == task_id
        )

        if result.status == "completed":
            if self.complete_after_execute:
                self._complete = True
        elif result.status == "failed":
            self._failed = True

        return result

    def is_complete(self, run_id: str) -> bool:
        return self._complete

    def has_failed(self, run_id: str) -> bool:
        return self._failed


def test_successful_run_emits_run_cycle_and_task_events() -> None:
    logger = InMemoryObservabilityLogger()

    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-1",
                status="completed",
                output="sensitive output",
            )
        ]
    )

    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(
            max_cycles=2,
            max_tasks_per_cycle=2,
        ),
    )

    observed = ObservableAutonomousLoop(
        loop,
        logger=logger,
    )

    report = observed.run("finish the task")

    assert report.success is True
    assert report.run_id == "run-001"
    assert backend.created_goal == "finish the task"

    events = logger.events()

    assert [event.event for event in events] == [
        "autonomous.run.started",
        "autonomous.task.completed",
        "autonomous.cycle.completed",
        "autonomous.run.completed",
    ]

    assert events[0].run_id is None
    assert events[1].run_id == "run-001"
    assert events[1].task_id == "task-1"
    assert events[2].run_id == "run-001"
    assert events[3].run_id == "run-001"


def test_failed_task_emits_failed_task_and_failed_run() -> None:
    logger = InMemoryObservabilityLogger()

    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-failed",
                status="failed",
                error="execution failed",
            )
        ]
    )

    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(
            max_cycles=2,
            max_tasks_per_cycle=2,
        ),
    )

    observed = ObservableAutonomousLoop(
        loop,
        logger=logger,
    )

    report = observed.run("fail safely")

    assert report.success is False
    assert report.stop_reason == "task_failed"

    events = logger.events()

    assert [event.event for event in events] == [
        "autonomous.run.started",
        "autonomous.task.failed",
        "autonomous.cycle.completed",
        "autonomous.run.failed",
    ]

    assert events[1].task_id == "task-failed"
    assert events[1].error == "execution failed"
    assert events[-1].error == "task_failed"


def test_failed_run_preserves_report_without_changing_it() -> None:
    logger = InMemoryObservabilityLogger()

    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-no-progress",
                status="pending",
            )
        ],
        complete_after_execute=False,
    )

    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(
            max_cycles=1,
            max_tasks_per_cycle=1,
        ),
    )

    observed = ObservableAutonomousLoop(
        loop,
        logger=logger,
    )

    report = observed.run("make no progress")

    assert report.success is False
    assert report.stop_reason == "no_progress"
    assert report.cycle_count == 1

    events = logger.events()
    assert events[-1].event == "autonomous.run.failed"
    assert events[-1].error == "no_progress"


def test_exception_is_logged_and_reraised() -> None:
    class _BrokenLoop:
        @property
        def config(self):
            return None

        def stop(self) -> None:
            return None

        def reset(self) -> None:
            return None

        def run(self, goal: str):
            raise RuntimeError("autonomous exploded")

    logger = InMemoryObservabilityLogger()

    observed = ObservableAutonomousLoop(
        _BrokenLoop(),
        logger=logger,
    )

    with pytest.raises(
        RuntimeError,
        match="autonomous exploded",
    ):
        observed.run("boom")

    events = logger.events()

    assert [event.event for event in events] == [
        "autonomous.run.started",
        "autonomous.run.failed",
    ]
    assert events[-1].error == "autonomous exploded"


def test_explicit_context_provider_is_used() -> None:
    logger = InMemoryObservabilityLogger()

    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-ctx",
                status="completed",
            )
        ]
    )

    loop = AutonomousLoop(backend)

    context = ObservabilityContext(
        run_id="external-run",
        task_id="external-task",
    )

    observed = ObservableAutonomousLoop(
        loop,
        logger=logger,
        context_provider=lambda: context,
    )

    observed.run("context test")

    events = logger.events()

    assert events[0].run_id == "external-run"
    assert events[0].task_id == "external-task"


def test_raw_task_output_is_never_logged() -> None:
    logger = InMemoryObservabilityLogger()

    secret = "TOP_SECRET_AUTONOMOUS_OUTPUT"

    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-secret",
                status="completed",
                output=secret,
            )
        ]
    )

    loop = AutonomousLoop(backend)

    observed = ObservableAutonomousLoop(
        loop,
        logger=logger,
    )

    observed.run("test output redaction")

    for event in logger.events():
        assert secret not in str(event.to_dict())

    task_event = next(
        event
        for event in logger.events()
        if event.event == "autonomous.task.completed"
    )

    assert task_event.metadata["has_output"] is True
    assert "output" not in task_event.metadata


def test_passed_logger_identity_is_preserved() -> None:
    logger = InMemoryObservabilityLogger()

    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-identity",
                status="completed",
            )
        ]
    )

    loop = AutonomousLoop(backend)
    observed = ObservableAutonomousLoop(
        loop,
        logger=logger,
    )

    assert observed.logger is logger


def test_logging_failure_does_not_break_autonomous_execution() -> None:
    class _BrokenLogger:
        def log(self, event, **data) -> None:
            raise RuntimeError("logger failed")

    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-safe",
                status="completed",
            )
        ]
    )

    loop = AutonomousLoop(backend)

    observed = ObservableAutonomousLoop(
        loop,
        logger=_BrokenLogger(),
    )

    report = observed.run("continue")

    assert report.success is True


def test_control_methods_are_delegated() -> None:
    backend = _Backend(
        results=[
            AutonomousTaskResult(
                task_id="task-control",
                status="completed",
            )
        ]
    )

    loop = AutonomousLoop(backend)
    observed = ObservableAutonomousLoop(loop)

    assert observed.loop is loop
    assert observed.config is loop.config

    observed.stop()
    observed.reset()
