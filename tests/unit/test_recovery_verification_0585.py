from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.recovery_contracts import (
    RecoveryFailureKind,
)
from osa.recovery_policy import (
    DefaultRecoveryPolicy,
    RecoveryPolicyConfig,
)
from osa.recovery_verification import (
    ActionVerificationResult,
    RecoveryVerificationError,
    VerificationAwareActionClassifier,
    create_verification_aware_executor,
)


@dataclass(frozen=True)
class FakeActionResult:
    success: bool
    output: str | None = None
    error: str | None = None


def _request() -> ActionRequest:
    return ActionRequest(
        request_id="req-verify-1",
        kind=ActionKind.TOOL,
        name="demo_tool",
        arguments={
            "value": 42,
        },
    )


def test_backend_success_and_verification_success_complete() -> None:
    calls: list[ActionRequest] = []

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        calls.append(request)

        return FakeActionResult(
            success=True,
            output="executed",
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        assert request == _request()
        assert result.output == "executed"

        return ActionVerificationResult(
            passed=True,
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=verifier,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is True
    assert result.attempt_count == 1
    assert calls == [_request()]


def test_verification_failure_becomes_recovery_failure() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=True,
            output="command accepted",
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        return ActionVerificationResult(
            passed=False,
            reason="window did not open",
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=verifier,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.attempt_count == 1
    assert result.failure_kind == (
        RecoveryFailureKind.VERIFICATION_FAILED
    )
    assert result.error == "window did not open"


def test_verification_failure_can_trigger_retry() -> None:
    calls: list[int] = []

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        calls.append(len(calls) + 1)

        return FakeActionResult(
            success=True,
            output=f"attempt-{len(calls)}",
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        if result.output == "attempt-1":
            return ActionVerificationResult(
                passed=False,
                reason="post-condition not reached",
            )

        return ActionVerificationResult(
            passed=True,
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=verifier,
        max_attempts=2,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset(
                    {
                        RecoveryFailureKind.VERIFICATION_FAILED,
                    }
                )
            )
        ),
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is True
    assert result.attempt_count == 2
    assert calls == [1, 2]
    assert result.attempts[0].failure_kind == (
        RecoveryFailureKind.VERIFICATION_FAILED
    )


def test_backend_failure_skips_verification() -> None:
    verification_calls = 0

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=False,
            error="backend failed",
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        nonlocal verification_calls
        verification_calls += 1

        return ActionVerificationResult(
            passed=True,
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=verifier,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert verification_calls == 0


def test_verifier_exception_is_fail_closed() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=True,
            output="executed",
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        raise RuntimeError(
            "verification exploded"
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=verifier,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.INVALID_RESULT
    )
    assert "verification exploded" in (
        result.error or ""
    )


def test_invalid_verifier_return_is_fail_closed() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=True,
            output="executed",
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=lambda request, result: "invalid",
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.INVALID_RESULT
    )


def test_verification_metadata_is_preserved() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=True,
            output="done",
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        return ActionVerificationResult(
            passed=True,
            metadata={
                "check": "window_exists",
            },
        )

    classifier = VerificationAwareActionClassifier(
        verifier,
    )

    recovery_request = (
        __import__(
            "osa.recovery_contracts",
            fromlist=["RecoveryRequest"],
        ).RecoveryRequest(
            request_id="req-verify-2",
            max_attempts=1,
            metadata={
                "_action_request": _request(),
            },
        )
    )

    attempt = classifier(
        recovery_request,
        FakeActionResult(
            success=True,
            output="done",
        ),
    )

    assert attempt.success is True
    assert attempt.metadata["verification_passed"] is True
    assert attempt.metadata["check"] == (
        "window_exists"
    )


def test_verification_result_rejects_invalid_passed_value() -> None:
    with pytest.raises(
        RecoveryVerificationError,
        match="passed",
    ):
        ActionVerificationResult(
            passed="yes",
        )


def test_verification_result_rejects_invalid_reason() -> None:
    with pytest.raises(
        RecoveryVerificationError,
        match="reason",
    ):
        ActionVerificationResult(
            passed=False,
            reason=123,
        )


def test_verification_result_copies_metadata() -> None:
    metadata = {
        "check": "exists",
    }

    result = ActionVerificationResult(
        passed=True,
        metadata=metadata,
    )

    metadata["changed"] = True

    assert result.metadata == {
        "check": "exists",
    }


def test_failed_verification_preserves_output_presence() -> None:
    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        return FakeActionResult(
            success=True,
            output="some output",
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        return ActionVerificationResult(
            passed=False,
            reason="expected state missing",
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=verifier,
    )

    result = recoverable.execute(
        _request(),
    )

    assert result.attempts[0].output_present is True


def test_retry_uses_same_action_request_after_verification_failure() -> None:
    requests: list[ActionRequest] = []

    def executor(
        request: ActionRequest,
    ) -> FakeActionResult:
        requests.append(request)

        return FakeActionResult(
            success=True,
            output=(
                "first"
                if len(requests) == 1
                else "second"
            ),
        )

    def verifier(
        request: ActionRequest,
        result: FakeActionResult,
    ) -> ActionVerificationResult:
        return ActionVerificationResult(
            passed=len(requests) > 1,
        )

    recoverable = create_verification_aware_executor(
        executor,
        verifier=verifier,
        max_attempts=2,
        policy=DefaultRecoveryPolicy(
            RecoveryPolicyConfig(
                retryable_kinds=frozenset(
                    {
                        RecoveryFailureKind.VERIFICATION_FAILED,
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
    assert requests == [
        request,
        request,
    ]
