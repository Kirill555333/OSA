import pytest

from osa.planning import Plan, PlanStep
from osa.tasks import (
    TaskManager,
    TaskManagerError,
    TaskStatus,
)


def create_plan() -> Plan:
    return Plan(
        goal="Prepare release",
        steps=(
            PlanStep(
                step_id="inspect",
                description="Inspect the repository.",
            ),
            PlanStep(
                step_id="test",
                description="Run tests.",
                depends_on=("inspect",),
            ),
            PlanStep(
                step_id="release",
                description="Prepare release.",
                depends_on=("test",),
            ),
        ),
    )


def test_task_manager_creates_run_from_plan() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    assert run.run_id
    assert run.goal == "Prepare release"
    assert set(run.tasks) == {
        "inspect",
        "test",
        "release",
    }

    assert all(
        task.status == TaskStatus.PENDING
        for task in run.tasks.values()
    )


def test_ready_tasks_respect_dependencies() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    ready = manager.ready_tasks(
        run.run_id
    )

    assert [task.task_id for task in ready] == [
        "inspect"
    ]

    blocked = manager.blocked_tasks(
        run.run_id
    )

    assert {
        task.task_id
        for task in blocked
    } == {
        "test",
        "release",
    }


def test_completed_dependency_unlocks_next_task() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    manager.start_task(
        run.run_id,
        "inspect",
    )
    manager.complete_task(
        run.run_id,
        "inspect",
    )

    ready = manager.ready_tasks(
        run.run_id
    )

    assert [task.task_id for task in ready] == [
        "test"
    ]


def test_task_lifecycle() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    task = manager.start_task(
        run.run_id,
        "inspect",
    )

    assert task.status == TaskStatus.RUNNING

    task = manager.complete_task(
        run.run_id,
        "inspect",
    )

    assert task.status == TaskStatus.COMPLETED
    assert task.error is None


def test_task_cannot_start_before_dependencies_complete() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    with pytest.raises(
        TaskManagerError,
        match="incomplete dependencies",
    ):
        manager.start_task(
            run.run_id,
            "test",
        )


def test_failed_task_records_error() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    manager.start_task(
        run.run_id,
        "inspect",
    )

    task = manager.fail_task(
        run.run_id,
        "inspect",
        "Repository is unavailable.",
    )

    assert task.status == TaskStatus.FAILED
    assert (
        task.error
        == "Repository is unavailable."
    )

    assert manager.has_failed(
        run.run_id
    )


def test_failed_task_can_be_retried() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    manager.start_task(
        run.run_id,
        "inspect",
    )

    manager.fail_task(
        run.run_id,
        "inspect",
        "Temporary failure.",
    )

    task = manager.retry_task(
        run.run_id,
        "inspect",
    )

    assert task.status == TaskStatus.PENDING
    assert task.error is None


def test_cancel_task() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    task = manager.cancel_task(
        run.run_id,
        "inspect",
    )

    assert task.status == TaskStatus.CANCELLED


def test_progress_and_completion() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    assert manager.progress(
        run.run_id
    ) == (0, 3)

    manager.start_task(
        run.run_id,
        "inspect",
    )
    manager.complete_task(
        run.run_id,
        "inspect",
    )

    manager.start_task(
        run.run_id,
        "test",
    )
    manager.complete_task(
        run.run_id,
        "test",
    )

    manager.start_task(
        run.run_id,
        "release",
    )
    manager.complete_task(
        run.run_id,
        "release",
    )

    assert manager.progress(
        run.run_id
    ) == (3, 3)

    assert manager.is_complete(
        run.run_id
    )


def test_unknown_run_is_rejected() -> None:
    manager = TaskManager()

    with pytest.raises(
        TaskManagerError,
        match="does not exist",
    ):
        manager.get_run("missing")


def test_unknown_task_is_rejected() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    with pytest.raises(
        TaskManagerError,
        match="does not exist",
    ):
        manager.start_task(
            run.run_id,
            "missing",
        )


def test_invalid_state_transition_is_rejected() -> None:
    manager = TaskManager()

    run = manager.create_run(
        create_plan()
    )

    manager.start_task(
        run.run_id,
        "inspect",
    )
    manager.complete_task(
        run.run_id,
        "inspect",
    )

    with pytest.raises(
        TaskManagerError,
        match="cannot complete",
    ):
        manager.complete_task(
            run.run_id,
            "inspect",
        )
