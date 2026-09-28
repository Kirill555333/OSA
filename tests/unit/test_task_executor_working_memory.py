from osa.planning.planner import Plan, PlanStep
from osa.tasks.executor import TaskExecutor
from osa.tasks.manager import TaskManager


def test_executor_populates_working_memory() -> None:
    manager = TaskManager()

    run = manager.create_run(
        Plan(
            goal="working memory integration",
            steps=(
                PlanStep(
                    step_id="step-1",
                    description="do something",
                ),
            ),
        )
    )

    executor = TaskExecutor(manager)

    report = executor.execute(
        run.run_id,
        lambda task, outputs: "completed",
    )

    snapshot = executor.working_memory.snapshot(run.run_id)

    assert report.success
    assert snapshot.goal == "working memory integration"
    assert snapshot.completed_task_ids == ("step-1",)
    assert snapshot.failed_task_ids == ()
    assert snapshot.attempts == {"step-1": 1}
    assert snapshot.outputs == {"step-1": "completed"}
    assert snapshot.last_error is None


def test_executor_working_memory_tracks_retry() -> None:
    manager = TaskManager()

    run = manager.create_run(
        Plan(
            goal="retry state",
            steps=(
                PlanStep(
                    step_id="step-1",
                    description="retry transient failure",
                ),
            ),
        )
    )

    calls = 0

    def handler(task, outputs):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise TimeoutError("temporary")

        return "recovered"

    from osa.tasks.recovery import TaskRecoveryPolicy

    executor = TaskExecutor(
        manager,
        recovery=TaskRecoveryPolicy(max_retries=1),
    )

    report = executor.execute(
        run.run_id,
        handler,
    )

    snapshot = executor.working_memory.snapshot(run.run_id)

    assert report.success
    assert calls == 2
    assert snapshot.completed_task_ids == ("step-1",)
    assert snapshot.failed_task_ids == ()
    assert snapshot.attempts == {"step-1": 2}
    assert snapshot.outputs == {"step-1": "recovered"}
