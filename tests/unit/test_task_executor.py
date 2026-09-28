from __future__ import annotations

import pytest

from osa.planning import Plan, PlanStep
from osa.tasks import (
    TaskExecutor,
    TaskManager,
    TaskStatus,
)


def create_run():
    manager = TaskManager()

    plan = Plan(
        goal="Build OSA report",
        steps=(
            PlanStep(
                step_id="collect",
                description="Collect information",
            ),
            PlanStep(
                step_id="write",
                description="Write report",
                depends_on=("collect",),
            ),
            PlanStep(
                step_id="verify",
                description="Verify report",
                depends_on=("write",),
            ),
        ),
    )

    return manager, manager.create_run(plan)


def test_executor_respects_dependencies() -> None:
    manager, run = create_run()

    execution_order: list[str] = []

    def handler(
        task,
        outputs,
    ) -> str:
        execution_order.append(task.task_id)

        if task.task_id == "write":
            assert outputs["collect"] == "collected"

        if task.task_id == "verify":
            assert outputs["write"] == "written"

        return {
            "collect": "collected",
            "write": "written",
            "verify": "verified",
        }[task.task_id]

    executor = TaskExecutor(manager)

    report = executor.execute(
        run.run_id,
        handler,
    )

    assert report.success is True
    assert execution_order == [
        "collect",
        "write",
        "verify",
    ]
    assert report.completed == (
        "collect",
        "write",
        "verify",
    )
    assert report.outputs == {
        "collect": "collected",
        "write": "written",
        "verify": "verified",
    }

    assert manager.is_complete(
        run.run_id
    )


def test_executor_marks_failed_task_and_blocks_dependents() -> None:
    manager, run = create_run()

    def handler(
        task,
        outputs,
    ) -> str:
        if task.task_id == "collect":
            raise RuntimeError(
                "collection failed"
            )

        return "unexpected"

    executor = TaskExecutor(manager)

    report = executor.execute(
        run.run_id,
        handler,
    )

    assert report.success is False
    assert report.completed == ()
    assert report.failed == ("collect",)
    assert report.blocked == (
        "write",
        "verify",
    )
    assert report.errors == {
        "collect": "collection failed",
    }

    snapshot = manager.snapshot(
        run.run_id
    )

    assert snapshot["collect"].status == (
        TaskStatus.FAILED
    )
    assert snapshot["write"].status == (
        TaskStatus.PENDING
    )


def test_executor_rejects_non_string_handler_output() -> None:
    manager = TaskManager()

    plan = Plan(
        goal="Test",
        steps=(
            PlanStep(
                step_id="one",
                description="One",
            ),
        ),
    )

    run = manager.create_run(plan)

    def handler(
        task,
        outputs,
    ):
        return 123

    executor = TaskExecutor(manager)

    report = executor.execute(
        run.run_id,
        handler,
    )

    assert report.success is False
    assert report.failed == ("one",)
    assert (
        report.errors["one"]
        == "Task handler must return a string."
    )


def test_executor_rejects_impossible_deadlock() -> None:
    manager = TaskManager()

    plan = Plan(
        goal="Deadlock",
        steps=(
            PlanStep(
                step_id="one",
                description="One",
                depends_on=("missing",),
            ),
        ),
    )

    run = manager.create_run(plan)

    executor = TaskExecutor(manager)

    with pytest.raises(
        Exception,
        match="cannot make progress",
    ):
        executor.execute(
            run.run_id,
            lambda task, outputs: "done",
        )
