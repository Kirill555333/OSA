from osa.planning.planner import Plan, PlanStep
from osa.tasks.executor import TaskExecutor
from osa.tasks.manager import TaskManager
from osa.tasks.recovery import TaskRecoveryPolicy


def _run_with_one_task() -> tuple[TaskManager, str]:
    manager = TaskManager()
    plan = Plan(
        goal="test recovery",
        steps=(
            PlanStep(
                step_id="step-1",
                description="run test task",
            ),
        ),
    )
    run = manager.create_run(plan)
    return manager, run.run_id


def test_recovery_retries_transient_failure() -> None:
    manager, run_id = _run_with_one_task()
    calls = 0

    def handler(task, outputs):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise TimeoutError("temporary failure")

        return "ok"

    executor = TaskExecutor(
        manager,
        recovery=TaskRecoveryPolicy(max_retries=1),
    )

    report = executor.execute(run_id, handler)

    assert report.success
    assert calls == 2
    assert report.attempts["step-1"] == 2
    assert report.outputs["step-1"] == "ok"


def test_recovery_does_not_retry_non_transient_failure() -> None:
    manager, run_id = _run_with_one_task()
    calls = 0

    def handler(task, outputs):
        nonlocal calls
        calls += 1
        raise ValueError("bad input")

    executor = TaskExecutor(
        manager,
        recovery=TaskRecoveryPolicy(max_retries=2),
    )

    report = executor.execute(run_id, handler)

    assert not report.success
    assert calls == 1
    assert report.attempts["step-1"] == 1
    assert "step-1" in report.failed
    assert "step-1" in report.errors


def test_recovery_respects_retry_limit() -> None:
    manager, run_id = _run_with_one_task()
    calls = 0

    def handler(task, outputs):
        nonlocal calls
        calls += 1
        raise ConnectionError("still unavailable")

    executor = TaskExecutor(
        manager,
        recovery=TaskRecoveryPolicy(max_retries=2),
    )

    report = executor.execute(run_id, handler)

    assert not report.success
    assert calls == 3
    assert report.attempts["step-1"] == 3
    assert "step-1" in report.failed
