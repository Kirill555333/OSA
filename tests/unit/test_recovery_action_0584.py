from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.recovery_action import (
    RecoverableActionExecutor,
    RecoveryActionIntegrationError,
)
from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryFailureKind,
)
from osa.recovery_policy import (
    DefaultRecoveryPolicy,
    RecoveryPolicyConfig,
)


@dataclass(frozen=True)
class FakeActionResult:
    success: bool
    output: str | None = None
    error: str | None = None


def _request() -> ActionRequest:
    return ActionRequest(
        request_id="req-1",
        kind=ActionKind.TOOL,
        name="demo_tool",
        arguments={
            "value": 42,
        },
    )


def _attempt_builder(
    request,
    result: dict[str, object],
) -> RecoveryAttempt:
    if result["ok"]:
        return RecoveryAttempt(
            attempt=request.attempt,
            max_attempts=request.max_attempts,
            success=True,
            output_present=True,
        )

    return RecoveryAttempt(
        attempt=request.attempt,
        max_attempts=request.max_attempts,
        success=False,
        failure_kind=RecoveryFailureKind.BACKEND_ERROR,
        error=str(result["error"]),
    )


def test_success_does_not_retry() -> None:
    calls: list[ActionRequest] = []

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        calls.append(request)

        return FakeActionResult(
            success=True,
            output="done",
        )

    recoverable = RecoverableActionExecutor(
        executor,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is True
    assert result.attempt_count == 1
    assert calls == [_request()]


def test_retry_reenters_same_action_executor() -> None:
    calls: list[ActionRequest] = []

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        calls.append(request)

        if len(calls) == 1:
            return FakeActionResult(
                success=False,
                error="temporary failure",
            )

        return FakeActionResult(
            success=True,
            output="done",
        )

    recoverable = RecoverableActionExecutor(
        executor,
        max_attempts=2,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset(
                    {
                        RecoveryFailureKind.BACKEND_ERROR,
                    }
                )
            )
        ),
    )

    request = _request()

    result = recoverable.execute(
        request,
    )

    assert result.success is True
    assert result.attempt_count == 2
    assert calls == [
        request,
        request,
    ]


def test_retry_preserves_request_identity_and_data() -> None:
    identities: list[int] = []
    arguments: list[dict[str, object]] = []

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        identities.append(id(request))
        arguments.append(dict(request.arguments))

        if len(identities) == 1:
            return FakeActionResult(
                success=False,
                error="temporary",
            )

        return FakeActionResult(
            success=True,
            output="ok",
        )

    request = _request()

    recoverable = RecoverableActionExecutor(
        executor,
        max_attempts=2,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset(
                    {
                        RecoveryFailureKind.BACKEND_ERROR,
                    }
                )
            )
        ),
    )

    result = recoverable.execute(
        request,
    )

    assert result.success is True
    assert identities == [
        id(request),
        id(request),
    ]
    assert arguments == [
        {
            "value": 42,
        },
        {
            "value": 42,
        },
    ]


def test_run_and_task_context_are_preserved() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=True,
            output="ok",
        )

    recoverable = RecoverableActionExecutor(
        executor,
        max_attempts=2,
    )

    result = recoverable.execute(
        _request(),
        run_id="run-7",
        task_id="task-9",
    )

    assert result.success is True
    assert result.attempt_count == 1


def test_custom_classifier_can_produce_permanent_failure() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=False,
            error="permission denied",
        )

    def classifier(
        recovery_request,
        result,
    ) -> RecoveryAttempt:
        return RecoveryAttempt(
            attempt=recovery_request.attempt,
            max_attempts=recovery_request.max_attempts,
            success=False,
            failure_kind=RecoveryFailureKind.PERMISSION_DENIED,
            error=result.error,
        )

    recoverable = RecoverableActionExecutor(
        executor,
        max_attempts=5,
        classifier=classifier,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset(
                    {
                        RecoveryFailureKind.BACKEND_ERROR,
                        RecoveryFailureKind.PERMISSION_DENIED,
                    }
                )
            )
        ),
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.attempt_count == 1
    assert result.failure_kind == (
        RecoveryFailureKind.PERMISSION_DENIED
    )


def test_classifier_failure_is_fail_closed() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=True,
            output="ok",
        )

    def broken_classifier(
        recovery_request,
        result,
    ):
        raise RuntimeError(
            "classifier exploded"
        )

    recoverable = RecoverableActionExecutor(
        executor,
        classifier=broken_classifier,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.INVALID_RESULT
    )


def test_invalid_action_request_is_rejected() -> None:
    recoverable = RecoverableActionExecutor(
        lambda request: FakeActionResult(
            success=True,
        ),
    )

    with pytest.raises(
        RecoveryActionIntegrationError,
        match="ActionRequest",
    ):
        recoverable.execute(
            "invalid",
        )


def test_invalid_action_result_is_fail_closed() -> None:
    recoverable = RecoverableActionExecutor(
        lambda request: object(),
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.INVALID_RESULT
    )


def test_backend_exception_is_preserved_as_failure() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        raise RuntimeError(
            "backend exploded"
        )

    recoverable = RecoverableActionExecutor(
        executor,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert result.error == (
        "backend exploded"
    )


def test_recovery_does_not_create_success_from_failed_result() -> None:
    calls = 0

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        nonlocal calls
        calls += 1

        return FakeActionResult(
            success=False,
            error="still failed",
        )

    recoverable = RecoverableActionExecutor(
        executor,
        max_attempts=3,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset()
            )
        ),
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert calls == 1
