from __future__ import annotations

import pytest

from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_recovery import (
    AutonomousRecoveryDecision,
    AutonomousRecoveryExecutor,
    CallbackAutonomousRecoveryPolicy,
    NeverRetryAutonomousRecoveryPolicy,
)


class FakeExecutor:
    def __init__(
        self,
        results: list[AutonomousTaskResult],
    ) -> None:
        self.results = list(results)
        self.calls: list[
            tuple[str, str]
        ] = []

    def execute(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.calls.append(
            (
                run_id,
                task_id,
            )
        )

        if self.results:
            return self.results.pop(0)

        return AutonomousTaskResult(
            task_id=task_id,
            status="failed",
            error="no more results",
        )


def completed(
    task_id: str = "task-1",
) -> AutonomousTaskResult:
    return AutonomousTaskResult(
        task_id=task_id,
        status="completed",
        output="done",
    )


def failed(
    error: str = "temporary failure",
    task_id: str = "task-1",
) -> AutonomousTaskResult:
    return AutonomousTaskResult(
        task_id=task_id,
        status="failed",
        error=error,
    )


def test_successful_task_is_never_retried() -> None:
    executor = FakeExecutor(
        [
            completed()
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        executor,
        max_attempts=5,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "completed"
    assert executor.calls == [
        (
            "run-1",
            "task-1",
        )
    ]


def test_failed_task_is_retried_when_policy_allows() -> None:
    executor = FakeExecutor(
        [
            failed("temporary"),
            completed(),
        ]
    )

    seen_attempts: list[int] = []

    def callback(
        run_id: str,
        task_id: str,
        result: AutonomousTaskResult,
        attempt: int,
    ) -> AutonomousRecoveryDecision:
        seen_attempts.append(attempt)
        return AutonomousRecoveryDecision(
            retry=True,
            reason="transient failure",
        )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=CallbackAutonomousRecoveryPolicy(
            callback
        ),
        max_attempts=3,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "completed"
    assert executor.calls == [
        (
            "run-1",
            "task-1",
        ),
        (
            "run-1",
            "task-1",
        ),
    ]
    assert seen_attempts == [1]


def test_policy_can_stop_recovery() -> None:
    executor = FakeExecutor(
        [
            failed("permanent"),
            completed(),
        ]
    )

    def callback(
        run_id: str,
        task_id: str,
        result: AutonomousTaskResult,
        attempt: int,
    ) -> AutonomousRecoveryDecision:
        return AutonomousRecoveryDecision(
            retry=False,
            reason="not retryable",
        )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=CallbackAutonomousRecoveryPolicy(
            callback
        ),
        max_attempts=3,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "failed"
    assert result.error == "permanent"
    assert executor.calls == [
        (
            "run-1",
            "task-1",
        )
    ]


def test_max_attempts_bounds_retries() -> None:
    executor = FakeExecutor(
        [
            failed("failure-1"),
            failed("failure-2"),
            failed("failure-3"),
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda *args: AutonomousRecoveryDecision(
                retry=True
            )
        ),
        max_attempts=2,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "failed"
    assert result.error == "failure-2"
    assert len(executor.calls) == 2


def test_default_policy_does_not_retry() -> None:
    executor = FakeExecutor(
        [
            failed()
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=NeverRetryAutonomousRecoveryPolicy(),
        max_attempts=5,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "failed"
    assert executor.calls == [
        (
            "run-1",
            "task-1",
        )
    ]


def test_policy_exception_fails_closed() -> None:
    executor = FakeExecutor(
        [
            failed()
        ]
    )

    def callback(
        *args,
    ):
        raise RuntimeError(
            "recovery service unavailable"
        )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=CallbackAutonomousRecoveryPolicy(
            callback
        ),
        max_attempts=3,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "failed"
    assert (
        result.error
        == (
            "autonomous_recovery_policy_failed: "
            "recovery service unavailable"
        )
    )
    assert executor.calls == [
        (
            "run-1",
            "task-1",
        )
    ]


def test_invalid_policy_decision_fails_closed() -> None:
    executor = FakeExecutor(
        [
            failed()
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        executor,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda *args: {
                "retry": True,
            }
        ),
        max_attempts=3,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "failed"
    assert (
        result.error
        == (
            "autonomous_recovery_policy_failed: "
            "invalid recovery decision."
        )
    )
    assert len(executor.calls) == 1


def test_unexpected_status_fails_closed() -> None:
    executor = FakeExecutor(
        [
            AutonomousTaskResult(
                task_id="task-1",
                status="running",
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        executor,
        max_attempts=3,
    )

    result = recovery.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "failed"
    assert (
        result.error
        == (
            "autonomous_recovery_invalid_result: "
            "unsupported task status 'running'."
        )
    )
    assert len(executor.calls) == 1


def test_empty_run_id_is_rejected() -> None:
    recovery = AutonomousRecoveryExecutor(
        FakeExecutor(
            [
                completed()
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="run_id cannot be empty",
    ):
        recovery.execute(
            "   ",
            "task-1",
        )


def test_empty_task_id_is_rejected() -> None:
    recovery = AutonomousRecoveryExecutor(
        FakeExecutor(
            [
                completed()
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="task_id cannot be empty",
    ):
        recovery.execute(
            "run-1",
            "   ",
        )


def test_invalid_max_attempts_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="max_attempts must be at least 1",
    ):
        AutonomousRecoveryExecutor(
            FakeExecutor(
                [
                    completed()
                ]
            ),
            max_attempts=0,
        )


def test_executor_is_required() -> None:
    with pytest.raises(
        ValueError,
        match="executor is required",
    ):
        AutonomousRecoveryExecutor(
            None
        )


def test_callback_policy_requires_callable() -> None:
    with pytest.raises(
        TypeError,
        match="callback must be callable",
    ):
        CallbackAutonomousRecoveryPolicy(
            None
        )
