from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryFailureKind,
    RecoveryRequest,
    RecoveryResult,
)
from osa.recovery_policy import NeverRetryRecoveryPolicy

from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_recovery_unification import (
    AutonomousRecoveryUnificationError,
    SafetyAwareAutonomousRecoveryPolicy,
    UnifiedAutonomousRecoveryBackend,
)


@dataclass(frozen=True)
class FakeAuthorization:
    allowed: bool
    reason: str | None = None


class FakeAuthorizer:
    def __init__(
        self,
        *,
        allowed: bool = True,
        reason: str | None = None,
    ) -> None:
        self.allowed = allowed
        self.reason = reason
        self.calls: list[tuple[str, str]] = []

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ) -> FakeAuthorization:
        self.calls.append(
            (run_id, task_id)
        )
        return FakeAuthorization(
            allowed=self.allowed,
            reason=self.reason,
        )


class RetryPolicy:
    def decide(
        self,
        request,
        attempt,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            retry=True,
            reason="retryable failure",
            failure_kind=RecoveryFailureKind.RETRYABLE,
            delay_seconds=0.0,
            next_attempt=attempt.attempt + 1,
        )


class FakeController:
    def __init__(
        self,
        result_factory,
    ) -> None:
        self._result_factory = result_factory
        self.requests: list[RecoveryRequest] = []

    def execute(
        self,
        request: RecoveryRequest,
    ) -> RecoveryResult:
        self.requests.append(request)
        return self._result_factory(request)


class FakeBackend:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, str]] = []
        self.results: list[AutonomousTaskResult] = []
        self.complete = False
        self.failed = False

    def create_run(
        self,
        goal: str,
    ) -> str:
        return "run-1"

    def ready_task_ids(
        self,
        run_id: str,
    ):
        return ("task-1",)

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        self.execute_calls.append(
            (run_id, task_id)
        )

        if self.results:
            return self.results.pop(0)

        return AutonomousTaskResult(
            task_id=task_id,
            status="completed",
            output="ok",
        )

    def is_complete(
        self,
        run_id: str,
    ) -> bool:
        return self.complete

    def has_failed(
        self,
        run_id: str,
    ) -> bool:
        return self.failed


def _retry_request() -> RecoveryRequest:
    return RecoveryRequest(
        run_id="run-1",
        task_id="task-1",
        attempt=1,
        max_attempts=3,
    )


def _retry_attempt() -> RecoveryAttempt:
    return RecoveryAttempt(
        attempt=1,
        max_attempts=3,
        success=False,
        failure_kind=RecoveryFailureKind.RETRYABLE,
        error="temporary",
        output_present=False,
        metadata={},
    )


def test_safety_aware_policy_rechecks_authorization_before_retry():
    authorizer = FakeAuthorizer(
        allowed=True
    )

    policy = SafetyAwareAutonomousRecoveryPolicy(
        RetryPolicy(),
        authorizer,
    )

    decision = policy.decide(
        _retry_request(),
        _retry_attempt(),
    )

    assert decision.retry is True
    assert authorizer.calls == [
        ("run-1", "task-1")
    ]


def test_safety_aware_policy_denies_retry_when_authorization_is_removed():
    authorizer = FakeAuthorizer(
        allowed=False,
        reason="permission revoked",
    )

    policy = SafetyAwareAutonomousRecoveryPolicy(
        RetryPolicy(),
        authorizer,
    )

    decision = policy.decide(
        _retry_request(),
        _retry_attempt(),
    )

    assert decision.retry is False
    assert (
        decision.failure_kind
        == RecoveryFailureKind.PERMISSION_DENIED
    )
    assert decision.next_attempt is None
    assert decision.reason == "permission revoked"


def test_safety_aware_policy_fails_closed_on_authorizer_exception():
    class RaisingAuthorizer:
        def authorize(
            self,
            run_id: str,
            task_id: str,
        ):
            raise RuntimeError(
                "authorization backend unavailable"
            )

    policy = SafetyAwareAutonomousRecoveryPolicy(
        RetryPolicy(),
        RaisingAuthorizer(),
    )

    decision = policy.decide(
        _retry_request(),
        _retry_attempt(),
    )

    assert decision.retry is False
    assert (
        decision.failure_kind
        == RecoveryFailureKind.PERMISSION_DENIED
    )
    assert decision.next_attempt is None


def test_unified_backend_builds_autonomous_recovery_request():
    backend = FakeBackend()

    task_result = AutonomousTaskResult(
        task_id="task-1",
        status="completed",
        output="done",
    )

    controller = FakeController(
        lambda request: RecoveryResult.succeeded(
            result=task_result,
        )
    )

    recovering = UnifiedAutonomousRecoveryBackend(
        backend,
        controller,
        max_attempts=4,
    )

    result = recovering.execute_task(
        "run-77",
        "task-1",
    )

    assert result is task_result
    assert len(controller.requests) == 1

    request = controller.requests[0]

    assert request.run_id == "run-77"
    assert request.task_id == "task-1"
    assert request.attempt == 1
    assert request.max_attempts == 4
    assert request.metadata["source"] == "autonomous"


def test_unified_backend_preserves_failed_autonomous_result():
    backend = FakeBackend()

    failed_result = AutonomousTaskResult(
        task_id="task-1",
        status="failed",
        error="temporary backend error",
    )

    controller = FakeController(
        lambda request: RecoveryResult.failed(
            attempts=(
                RecoveryAttempt(
                    attempt=1,
                    max_attempts=1,
                    success=False,
                    failure_kind=RecoveryFailureKind.BACKEND_ERROR,
                    error="temporary backend error",
                    output_present=False,
                    metadata={},
                ),
            ),
            failure_kind=RecoveryFailureKind.BACKEND_ERROR,
            error="temporary backend error",
            exhausted=True,
            result=failed_result,
        )
    )

    recovering = UnifiedAutonomousRecoveryBackend(
        backend,
        controller,
        max_attempts=1,
    )

    result = recovering.execute_task(
        "run-1",
        "task-1",
    )

    assert result is failed_result
    assert result.status == "failed"
    assert result.error == "temporary backend error"


def test_unified_backend_proxies_backend_lifecycle():
    backend = FakeBackend()

    controller = FakeController(
        lambda request: RecoveryResult.succeeded(
            result=AutonomousTaskResult(
                task_id=request.task_id,
                status="completed",
                output="ok",
            )
        )
    )

    recovering = UnifiedAutonomousRecoveryBackend(
        backend,
        controller,
    )

    assert recovering.create_run("goal") == "run-1"
    assert recovering.ready_task_ids("run-1") == (
        "task-1",
    )
    assert recovering.is_complete("run-1") is False
    assert recovering.has_failed("run-1") is False


def test_unified_backend_rejects_invalid_controller_result():
    backend = FakeBackend()

    controller = FakeController(
        lambda request: object()
    )

    recovering = UnifiedAutonomousRecoveryBackend(
        backend,
        controller,
    )

    with pytest.raises(
        AutonomousRecoveryUnificationError,
        match="invalid result",
    ):
        recovering.execute_task(
            "run-1",
            "task-1",
        )


def test_unified_backend_rejects_empty_identifiers():
    backend = FakeBackend()

    controller = FakeController(
        lambda request: RecoveryResult.succeeded(
            result=AutonomousTaskResult(
                task_id="task-1",
                status="completed",
            )
        )
    )

    recovering = UnifiedAutonomousRecoveryBackend(
        backend,
        controller,
    )

    with pytest.raises(
        AutonomousRecoveryUnificationError,
    ):
        recovering.execute_task(
            "",
            "task-1",
        )

    with pytest.raises(
        AutonomousRecoveryUnificationError,
    ):
        recovering.execute_task(
            "run-1",
            "",
        )


def test_never_retry_policy_remains_compatible_with_safety_wrapper():
    authorizer = FakeAuthorizer(
        allowed=True
    )

    policy = SafetyAwareAutonomousRecoveryPolicy(
        NeverRetryRecoveryPolicy(),
        authorizer,
    )

    decision = policy.decide(
        _retry_request(),
        _retry_attempt(),
    )

    assert decision.retry is False
    assert authorizer.calls == []
