from __future__ import annotations

import pytest

from osa.observability import (
    InMemoryObservabilityLogger,
    ObservableAutonomousRecovery,
)
from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_recovery import (
    AutonomousRecoveryDecision,
    AutonomousRecoveryExecutor,
    NeverRetryAutonomousRecoveryPolicy,
)


class _Executor:
    def __init__(
        self,
        results: list[AutonomousTaskResult],
    ) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, str]] = []

    def execute(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.calls.append((run_id, task_id))

        if not self.results:
            raise RuntimeError("no result configured")

        return self.results.pop(0)


class _RetryPolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def decide(
        self,
        run_id: str,
        task_id: str,
        result: AutonomousTaskResult,
        attempt: int,
    ) -> AutonomousRecoveryDecision:
        self.calls.append(
            (run_id, task_id, attempt)
        )
        return AutonomousRecoveryDecision(
            retry=True,
            reason="retry_requested",
        )


def test_success_emits_started_and_completed() -> None:
    logger = InMemoryObservabilityLogger()

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-1",
                status="completed",
                output="done",
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
        max_attempts=1,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=logger,
    )

    result = observed.execute(
        "run-1",
        "task-1",
    )

    assert result.status == "completed"
    assert result.task_id == "task-1"
    assert task_executor.calls == [
        ("run-1", "task-1")
    ]

    events = logger.events()

    assert [event.event for event in events] == [
        "recovery.started",
        "recovery.completed",
    ]
    assert events[0].run_id == "run-1"
    assert events[0].task_id == "task-1"
    assert events[-1].run_id == "run-1"
    assert events[-1].task_id == "task-1"
    assert events[-1].status == "completed"


def test_failed_recovery_emits_exhausted() -> None:
    logger = InMemoryObservabilityLogger()

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-failed",
                status="failed",
                error="execution failed",
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
        max_attempts=1,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=logger,
    )

    result = observed.execute(
        "run-2",
        "task-failed",
    )

    assert result.status == "failed"
    assert result.error == "execution failed"

    events = logger.events()

    assert [event.event for event in events] == [
        "recovery.started",
        "recovery.exhausted",
    ]
    assert events[-1].error == "execution failed"
    assert events[-1].status == "failed"


def test_policy_failure_is_logged_as_recovery_failed() -> None:
    logger = InMemoryObservabilityLogger()

    class _BrokenPolicy:
        def decide(
            self,
            run_id,
            task_id,
            result,
            attempt,
        ):
            raise RuntimeError("policy exploded")

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-policy",
                status="failed",
                error="execution failed",
            ),
            AutonomousTaskResult(
                task_id="task-policy",
                status="completed",
            ),
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
        policy=_BrokenPolicy(),
        max_attempts=2,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=logger,
    )

    result = observed.execute(
        "run-3",
        "task-policy",
    )

    assert result.status == "failed"
    assert result.error.startswith(
        "autonomous_recovery_policy_failed:"
    )

    events = logger.events()

    assert [event.event for event in events] == [
        "recovery.started",
        "recovery.failed",
    ]
    assert events[-1].error == result.error


def test_retry_behavior_is_preserved() -> None:
    logger = InMemoryObservabilityLogger()
    policy = _RetryPolicy()

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-retry",
                status="failed",
                error="temporary failure",
            ),
            AutonomousTaskResult(
                task_id="task-retry",
                status="completed",
                output="recovered",
            ),
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
        policy=policy,
        max_attempts=2,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=logger,
    )

    result = observed.execute(
        "run-4",
        "task-retry",
    )

    assert result.status == "completed"
    assert result.output == "recovered"

    assert task_executor.calls == [
        ("run-4", "task-retry"),
        ("run-4", "task-retry"),
    ]

    assert policy.calls == [
        ("run-4", "task-retry", 1)
    ]

    events = logger.events()

    assert [event.event for event in events] == [
        "recovery.started",
        "recovery.completed",
    ]


def test_logger_identity_is_preserved() -> None:
    logger = InMemoryObservabilityLogger()

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-identity",
                status="completed",
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=logger,
    )

    assert observed.logger is logger


def test_logging_failure_does_not_break_recovery() -> None:
    class _BrokenLogger:
        def log(self, event, **data) -> None:
            raise RuntimeError("logger exploded")

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-safe",
                status="completed",
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=_BrokenLogger(),
    )

    result = observed.execute(
        "run-safe",
        "task-safe",
    )

    assert result.status == "completed"


def test_raw_output_is_never_logged() -> None:
    logger = InMemoryObservabilityLogger()

    secret = "TOP_SECRET_RECOVERY_OUTPUT"

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-secret",
                status="completed",
                output=secret,
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=logger,
    )

    result = observed.execute(
        "run-secret",
        "task-secret",
    )

    assert result.output == secret

    for event in logger.events():
        assert secret not in str(event.to_dict())

    completed = logger.events()[-1]

    assert completed.metadata["has_output"] is True
    assert "output" not in completed.metadata


def test_max_attempts_is_exposed_without_changing_executor() -> None:
    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-max",
                status="completed",
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
        max_attempts=3,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
    )

    assert observed.executor is recovery
    assert observed.policy is recovery.policy
    assert observed.max_attempts == 3


def test_none_executor_is_rejected() -> None:
    with pytest.raises(
        Exception,
        match="executor is required",
    ):
        ObservableAutonomousRecovery(None)


def test_default_policy_remains_fail_closed() -> None:
    logger = InMemoryObservabilityLogger()

    task_executor = _Executor(
        [
            AutonomousTaskResult(
                task_id="task-default",
                status="failed",
                error="failure",
            )
        ]
    )

    recovery = AutonomousRecoveryExecutor(
        task_executor,
        policy=NeverRetryAutonomousRecoveryPolicy(),
        max_attempts=2,
    )

    observed = ObservableAutonomousRecovery(
        recovery,
        logger=logger,
    )

    result = observed.execute(
        "run-default",
        "task-default",
    )

    assert result.status == "failed"
    assert task_executor.calls == [
        ("run-default", "task-default")
    ]
