from __future__ import annotations

import pytest

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryContractError,
    RecoveryDecision,
    RecoveryFailureKind,
    RecoveryRequest,
    RecoveryResult,
)


def test_recovery_request_supports_action_identifiers() -> None:
    request = RecoveryRequest(
        request_id="req-1",
        action_kind="browser",
        action_name="browser_click",
    )

    assert request.request_id == "req-1"
    assert request.action is True
    assert request.autonomous is False
    assert request.attempt == 1
    assert request.max_attempts == 1


def test_recovery_request_supports_autonomous_identifiers() -> None:
    request = RecoveryRequest(
        run_id="run-1",
        task_id="task-1",
        attempt=2,
        max_attempts=3,
    )

    assert request.run_id == "run-1"
    assert request.task_id == "task-1"
    assert request.autonomous is True
    assert request.action is False
    assert request.attempt == 2
    assert request.max_attempts == 3


def test_recovery_request_can_support_both_flows() -> None:
    request = RecoveryRequest(
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
        action_kind="desktop",
        action_name="desktop_click_point",
    )

    assert request.autonomous is True
    assert request.action is True


def test_recovery_request_requires_identifier() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="at least one recovery identifier",
    ):
        RecoveryRequest()


def test_recovery_request_rejects_attempt_above_maximum() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="cannot exceed",
    ):
        RecoveryRequest(
            request_id="req-1",
            attempt=3,
            max_attempts=2,
        )


def test_recovery_request_metadata_is_immutable() -> None:
    metadata = {
        "source": "test",
    }

    request = RecoveryRequest(
        request_id="req-1",
        metadata=metadata,
    )

    metadata["source"] = "changed"

    assert request.metadata["source"] == "test"

    with pytest.raises(TypeError):
        request.metadata["source"] = "changed"


def test_recovery_attempt_success_is_valid() -> None:
    attempt = RecoveryAttempt(
        attempt=1,
        max_attempts=3,
        success=True,
        output_present=True,
    )

    assert attempt.success is True
    assert attempt.failure_kind is None
    assert attempt.output_present is True


def test_recovery_attempt_failure_requires_kind() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="requires failure_kind",
    ):
        RecoveryAttempt(
            attempt=1,
            max_attempts=3,
            success=False,
        )


def test_recovery_attempt_success_cannot_have_failure_kind() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="cannot have failure_kind",
    ):
        RecoveryAttempt(
            attempt=1,
            max_attempts=3,
            success=True,
            failure_kind=RecoveryFailureKind.RETRYABLE,
        )


def test_recovery_attempt_accepts_string_failure_kind() -> None:
    attempt = RecoveryAttempt(
        attempt=1,
        max_attempts=2,
        success=False,
        failure_kind="backend_error",
        error="temporary failure",
    )

    assert attempt.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )


def test_recovery_decision_retry_requires_next_attempt() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="requires next_attempt",
    ):
        RecoveryDecision(
            retry=True,
            reason="retry",
        )


def test_recovery_decision_non_retry_cannot_have_next_attempt() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="cannot define next_attempt",
    ):
        RecoveryDecision(
            retry=False,
            next_attempt=2,
        )


def test_recovery_decision_retry_is_valid() -> None:
    decision = RecoveryDecision(
        retry=True,
        reason="temporary backend failure",
        failure_kind=RecoveryFailureKind.RETRYABLE,
        delay_seconds=0.25,
        next_attempt=2,
    )

    assert decision.retry is True
    assert decision.next_attempt == 2
    assert decision.delay_seconds == 0.25


def test_recovery_decision_no_retry_is_valid() -> None:
    decision = RecoveryDecision(
        retry=False,
        reason="permission denied",
        failure_kind=RecoveryFailureKind.PERMISSION_DENIED,
    )

    assert decision.retry is False
    assert decision.next_attempt is None
    assert decision.failure_kind == (
        RecoveryFailureKind.PERMISSION_DENIED
    )


def test_recovery_result_success_factory() -> None:
    attempt = RecoveryAttempt(
        attempt=1,
        max_attempts=1,
        success=True,
    )

    result = RecoveryResult.succeeded(
        result="action-result",
        attempts=(attempt,),
    )

    assert result.success is True
    assert result.attempt_count == 1
    assert result.last_attempt is attempt
    assert result.result == "action-result"
    assert result.failure_kind is None
    assert result.error is None
    assert result.exhausted is False


def test_recovery_result_failure_factory() -> None:
    attempt = RecoveryAttempt(
        attempt=1,
        max_attempts=1,
        success=False,
        failure_kind=RecoveryFailureKind.BACKEND_ERROR,
        error="backend failed",
    )

    result = RecoveryResult.failed(
        failure_kind=RecoveryFailureKind.MAX_ATTEMPTS,
        error="retries exhausted",
        attempts=(attempt,),
        exhausted=True,
        result="final-result",
    )

    assert result.success is False
    assert result.failure_kind == (
        RecoveryFailureKind.MAX_ATTEMPTS
    )
    assert result.error == "retries exhausted"
    assert result.exhausted is True
    assert result.result == "final-result"


def test_recovery_result_attempts_must_be_consecutive() -> None:
    first = RecoveryAttempt(
        attempt=1,
        max_attempts=3,
        success=False,
        failure_kind=RecoveryFailureKind.RETRYABLE,
    )
    third = RecoveryAttempt(
        attempt=3,
        max_attempts=3,
        success=False,
        failure_kind=RecoveryFailureKind.MAX_ATTEMPTS,
    )

    with pytest.raises(
        RecoveryContractError,
        match="ordered consecutively",
    ):
        RecoveryResult.failed(
            failure_kind=RecoveryFailureKind.MAX_ATTEMPTS,
            attempts=(first, third),
        )


def test_failed_result_requires_failure_kind() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="requires failure_kind",
    ):
        RecoveryResult(
            success=False,
        )


def test_successful_result_cannot_have_error() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="cannot have error",
    ):
        RecoveryResult(
            success=True,
            error="unexpected",
        )


def test_failure_kind_enum_contains_all_required_categories() -> None:
    assert set(RecoveryFailureKind) == {
        RecoveryFailureKind.RETRYABLE,
        RecoveryFailureKind.NON_RETRYABLE,
        RecoveryFailureKind.PERMISSION_DENIED,
        RecoveryFailureKind.CONFIRMATION_DENIED,
        RecoveryFailureKind.BACKEND_ERROR,
        RecoveryFailureKind.INVALID_RESULT,
        RecoveryFailureKind.VERIFICATION_FAILED,
        RecoveryFailureKind.MAX_ATTEMPTS,
    }


def test_contracts_are_immutable() -> None:
    request = RecoveryRequest(
        request_id="req-immutable",
    )

    with pytest.raises(
        AttributeError,
    ):
        request.request_id = "changed"


def test_negative_delay_is_rejected() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="non-negative",
    ):
        RecoveryDecision(
            retry=False,
            delay_seconds=-1,
        )


def test_boolean_attempt_is_rejected() -> None:
    with pytest.raises(
        RecoveryContractError,
        match="must be an integer",
    ):
        RecoveryRequest(
            request_id="req-bool",
            attempt=True,
        )
