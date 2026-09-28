from __future__ import annotations

from osa.tasks.autonomous import (
    AutonomousLoop,
    AutonomousLoopConfig,
    AutonomousTaskResult,
    CallbackAutonomousBackend,
)


class FakeBackend:
    def __init__(self) -> None:
        self.next_run = 1
        self.ready = {
            "task-1": ["task-1"],
            "task-2": ["task-2"],
            "task-3": [],
        }
        self.executed: list[str] = []
        self.completed = False
        self.failed = False

    def create_run(self, goal: str) -> str:
        assert goal == "finish the task"
        run_id = f"run-{self.next_run}"
        self.next_run += 1
        return run_id

    def ready_task_ids(self, run_id: str):
        if not self.executed:
            return ("task-1",)
        if self.executed == ["task-1"]:
            return ("task-2",)
        return ()

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.executed.append(task_id)

        if task_id == "task-2":
            self.completed = True

        return AutonomousTaskResult(
            task_id=task_id,
            status="completed",
            output=f"done: {task_id}",
        )

    def is_complete(self, run_id: str) -> bool:
        return self.completed

    def has_failed(self, run_id: str) -> bool:
        return self.failed


def test_autonomous_loop_runs_until_complete() -> None:
    backend = FakeBackend()
    loop = AutonomousLoop(backend)

    report = loop.run("finish the task")

    assert report.success is True
    assert report.stopped is False
    assert report.stop_reason is None
    assert report.cycle_count == 2
    assert report.completed_task_ids == ("task-1", "task-2")
    assert backend.executed == ["task-1", "task-2"]


def test_autonomous_loop_stops_when_no_work_is_ready() -> None:
    backend = FakeBackend()
    backend.completed = False

    def ready_task_ids(run_id: str):
        return ()

    adapted = CallbackAutonomousBackend(
        create_run=backend.create_run,
        ready_task_ids=ready_task_ids,
        execute_task=backend.execute_task,
        is_complete=backend.is_complete,
        has_failed=backend.has_failed,
    )

    loop = AutonomousLoop(adapted)

    report = loop.run("finish the task")

    assert report.success is False
    assert report.stop_reason == "no_ready_tasks"
    assert backend.executed == []


def test_autonomous_loop_stops_on_failure() -> None:
    backend = FakeBackend()
    backend.failed = True

    loop = AutonomousLoop(backend)

    report = loop.run("finish the task")

    assert report.success is False
    assert report.stop_reason == "task_failed"
    assert report.cycle_count == 0


def test_autonomous_loop_respects_max_cycles() -> None:
    backend = FakeBackend()
    backend.completed = False

    loop = AutonomousLoop(
        backend,
        config=AutonomousLoopConfig(max_cycles=1),
    )

    report = loop.run("finish the task")

    assert report.success is False
    assert report.stop_reason == "max_cycles_reached"
    assert report.cycle_count == 1


def test_autonomous_loop_can_be_stopped() -> None:
    backend = FakeBackend()
    loop = AutonomousLoop(backend)

    original_execute = backend.execute_task

    def execute_and_stop(
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        result = original_execute(run_id, task_id)
        loop.stop()
        return result

    backend.execute_task = execute_and_stop  # type: ignore[method-assign]

    report = loop.run("finish the task")

    assert report.success is False
    assert report.stopped is True
    assert report.stop_reason == "stop_requested"
    assert report.completed_task_ids == ("task-1",)


def test_autonomous_loop_rejects_empty_goal() -> None:
    backend = FakeBackend()
    loop = AutonomousLoop(backend)

    try:
        loop.run("   ")
    except ValueError as exc:
        assert str(exc) == "Autonomous goal cannot be empty."
    else:
        raise AssertionError("Expected ValueError")
