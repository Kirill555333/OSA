from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.tasks.autonomous import (
    AutonomousCycle,
    AutonomousLoop,
    AutonomousLoopConfig,
    AutonomousRunReport,
    AutonomousTaskResult,
    CallbackAutonomousBackend,
)


@dataclass
class FakeBackend:
    ready: tuple[str, ...] = ()
    complete: bool = False
    failed: bool = False

    def __post_init__(self) -> None:
        self.created_goals: list[str] = []
        self.executed_tasks: list[tuple[str, str]] = []
        self.run_id = "run-1"
        self.result_status = "completed"

    def create_run(self, goal: str) -> str:
        self.created_goals.append(goal)
        return self.run_id

    def ready_task_ids(self, run_id: str) -> tuple[str, ...]:
        return self.ready

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.executed_tasks.append((run_id, task_id))
        return AutonomousTaskResult(
            task_id=task_id,
            status=self.result_status,
            output=f"done:{task_id}",
        )

    def is_complete(self, run_id: str) -> bool:
        return self.complete

    def has_failed(self, run_id: str) -> bool:
        return self.failed


@dataclass
class FakeAuthorization:
    allowed: bool
    reason: str | None = None


@dataclass
class FakeAuthorizer:
    decisions: dict[str, FakeAuthorization]

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ) -> FakeAuthorization:
        return self.decisions.get(
            task_id,
            FakeAuthorization(True),
        )


def test_rejects_empty_goal_before_backend_creation() -> None:
    backend = FakeBackend()
    loop = AutonomousLoop(backend)

    with pytest.raises(ValueError, match="goal cannot be empty"):
        loop.run("   ")

    assert backend.created_goals == []


def test_normalizes_goal_before_create_run() -> None:
    backend = FakeBackend(complete=True)
    loop = AutonomousLoop(backend)

    report = loop.run("  finish the job  ")

    assert report.goal == "finish the job"
    assert backend.created_goals == ["finish the job"]
    assert report.success is True


def test_respects_max_tasks_per_cycle() -> None:
    backend = FakeBackend(
        ready=("task-1", "task-2", "task-3"),
    )
    config = AutonomousLoopConfig(max_cycles=1, max_tasks_per_cycle=2)
    loop = AutonomousLoop(backend, config=config)

    report = loop.run("execute tasks")

    assert backend.executed_tasks == [
        ("run-1", "task-1"),
        ("run-1", "task-2"),
    ]
    assert report.cycle_count == 1
    assert report.stop_reason == "max_cycles_reached"


def test_authorizer_denial_is_recorded_and_not_executed() -> None:
    backend = FakeBackend(
        ready=("task-1", "task-2"),
    )
    authorizer = FakeAuthorizer(
        decisions={
            "task-1": FakeAuthorization(
                allowed=False,
                reason="requires_confirmation",
            ),
        }
    )
    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(
            max_cycles=1,
            max_tasks_per_cycle=2,
        ),
        authorizer=authorizer,
    )

    report = loop.run("execute tasks")

    assert backend.executed_tasks == [
        ("run-1", "task-2"),
    ]
    assert report.failed_task_ids == ("task-1",)
    assert report.completed_task_ids == ("task-2",)


def test_returns_no_ready_tasks() -> None:
    backend = FakeBackend()
    loop = AutonomousLoop(backend)

    report = loop.run("wait for work")

    assert report.success is False
    assert report.stopped is False
    assert report.stop_reason == "no_ready_tasks"
    assert report.cycles == ()


def test_returns_task_failed() -> None:
    backend = FakeBackend(
        ready=("task-1",),
        failed=True,
    )
    loop = AutonomousLoop(backend)

    report = loop.run("run failed task")

    assert report.success is False
    assert report.stop_reason == "task_failed"
    assert report.cycles == ()


def test_returns_no_progress() -> None:
    backend = FakeBackend(
        ready=("task-1",),
    )
    backend.result_status = "pending"

    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(
            max_cycles=4,
            max_tasks_per_cycle=1,
        ),
    )

    report = loop.run("make progress")

    assert report.success is False
    assert report.stop_reason == "no_progress"
    assert report.cycle_count == 1
    assert report.cycles[0].progressed is False


def test_stops_when_stop_is_requested() -> None:
    backend = FakeBackend(
        ready=("task-1",),
    )
    loop = AutonomousLoop(backend)

    original_execute = backend.execute_task

    def execute_and_stop(
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        loop.stop()
        return original_execute(run_id, task_id)

    backend.execute_task = execute_and_stop  # type: ignore[method-assign]

    report = loop.run("stop after current task")

    assert report.success is False
    assert report.stopped is True
    assert report.stop_reason == "stop_requested"
    assert report.cycle_count == 1
    assert report.completed_task_ids == ("task-1",)


def test_report_properties_deduplicate_task_ids() -> None:
    first_cycle = AutonomousCycle(
        cycle=1,
        ready_task_ids=("task-1",),
        results=(
            AutonomousTaskResult(
                task_id="task-1",
                status="completed",
            ),
        ),
        progressed=True,
    )

    second_cycle = AutonomousCycle(
        cycle=2,
        ready_task_ids=("task-1", "task-2"),
        results=(
            AutonomousTaskResult(
                task_id="task-1",
                status="completed",
            ),
            AutonomousTaskResult(
                task_id="task-2",
                status="failed",
            ),
        ),
        progressed=True,
    )

    report = AutonomousRunReport(
        goal="demo",
        run_id="run-1",
        success=False,
        stopped=False,
        stop_reason="task_failed",
        cycles=(first_cycle, second_cycle),
    )

    assert report.cycle_count == 2
    assert report.completed_task_ids == ("task-1",)
    assert report.failed_task_ids == ("task-2",)


def test_callback_backend_forwards_operations() -> None:
    events: list[str] = []

    backend = CallbackAutonomousBackend(
        create_run=lambda goal: (
            events.append(f"create:{goal}") or "run-9"
        ),
        ready_task_ids=lambda run_id: (
            events.append(f"ready:{run_id}") or ("task-9",)
        ),
        execute_task=lambda run_id, task_id: (
            events.append(
                f"execute:{run_id}:{task_id}"
            )
            or AutonomousTaskResult(
                task_id=task_id,
                status="completed",
            )
        ),
        is_complete=lambda run_id: (
            events.append(f"complete:{run_id}") or True
        ),
        has_failed=lambda run_id: (
            events.append(f"failed:{run_id}") or False
        ),
    )

    assert backend.create_run("goal") == "run-9"
    assert backend.ready_task_ids("run-9") == ("task-9",)
    assert (
        backend.execute_task(
            "run-9",
            "task-9",
        ).status
        == "completed"
    )
    assert backend.is_complete("run-9") is True
    assert backend.has_failed("run-9") is False

    assert events == [
        "create:goal",
        "ready:run-9",
        "execute:run-9:task-9",
        "complete:run-9",
        "failed:run-9",
    ]
