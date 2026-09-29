from __future__ import annotations

import pytest

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryContractError,
    RecoveryDecision,
    RecoveryFailureKind,
    RecoveryRequest,
)
from osa.recovery_policy import (
    CallbackRecoveryPolicy,
    DefaultRecoveryPolicy,
    MappingRecoveryPolicy,
    NeverRetryRecoveryPolicy,
    RecoveryPolicyConfig,
    RecoveryPolicyError,
)


def _request(
    *,
    attempt: int = 1,
    max_attempts: int = 3,
) -> RecoveryRequest:
    return RecoveryRequest(
        request_id="req-1",
        attempt=attempt,
        max_attempts=max_attempts,
    )


def _failed_attempt(
    *,
    attempt: int = 1,
    max_attempts: int = 3,
    kind: RecoveryFailureKind = (
        RecoveryFailureKind.BACKEND_ERROR
    ),
) -> RecoveryAttempt:
    return RecoveryAttempt(
        attempt=attempt,
        max_attempts=max_attempts,
        success=False,
        failure_kind=kind,
        error="failure",
    )


def test_default_policy_is_fail_closed() -> None:
    policy = DefaultRecoveryPolicy()

    decision = policy.decide(
        _request(),
        _failed_attempt(),
    )

    assert decision.retry is False
    assert decision.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert decision.next_attempt is None


def test_default_policy_retries_only_explicitly_enabled_kind() -> None:
    policy = DefaultRecoveryPolicy(
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    RecoveryFailureKind.BACKEND_ERROR,
                }
            ),
            default_delay_seconds=0.25,
        )
    )

    decision = policy.decide(
        _request(),
        _failed_attempt(),
    )

    assert decision.retry is True
    assert decision.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert decision.next_attempt == 2
    assert decision.delay_seconds == 0.25


def test_default_policy_permanently_rejects_permission_denied_retry() -> None:
    policy = DefaultRecoveryPolicy(
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    RecoveryFailureKind.BACKEND_ERROR,
                    RecoveryFailureKind.PERMISSION_DENIED,
                }
            )
        )
    )

    decision = policy.decide(
        _request(),
        _failed_attempt(
            kind=RecoveryFailureKind.PERMISSION_DENIED
        ),
    )

    assert decision.retry is False
    assert decision.failure_kind == (
        RecoveryFailureKind.PERMISSION_DENIED
    )


def test_default_policy_allows_explicit_policy_choice() -> None:
    """
    Policy engine itself is configurable. Safety classification is supplied
    by the caller rather than silently overridden here.
    """
    policy = DefaultRecoveryPolicy(
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    RecoveryFailureKind.VERIFICATION_FAILED,
                }
            )
        )
    )

    decision = policy.decide(
        _request(),
        _failed_attempt(
            kind=RecoveryFailureKind.VERIFICATION_FAILED
        ),
    )

    assert decision.retry is True


def test_default_policy_stops_at_max_attempts() -> None:
    policy = DefaultRecoveryPolicy(
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    RecoveryFailureKind.BACKEND_ERROR,
                }
            )
        )
    )

    request = _request(
        attempt=3,
        max_attempts=3,
    )

    attempt = _failed_attempt(
        attempt=3,
        max_attempts=3,
    )

    decision = policy.decide(
        request,
        attempt,
    )

    assert decision.retry is False
    assert decision.failure_kind == (
        RecoveryFailureKind.MAX_ATTEMPTS
    )
    assert decision.reason == (
        "maximum_attempts_reached"
    )


def test_successful_attempt_never_retries() -> None:
    policy = DefaultRecoveryPolicy(
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    RecoveryFailureKind.BACKEND_ERROR,
                }
            )
        )
    )

    attempt = RecoveryAttempt(
        attempt=1,
        max_attempts=3,
        success=True,
    )

    decision = policy.decide(
        _request(),
        attempt,
    )

    assert decision.retry is False
    assert decision.reason == "attempt_succeeded"


def test_mapping_policy_defaults_missing_rule_to_no_retry() -> None:
    policy = MappingRecoveryPolicy(
        {
            RecoveryFailureKind.BACKEND_ERROR: True,
        }
    )

    decision = policy.decide(
        _request(),
        _failed_attempt(
            kind=RecoveryFailureKind.INVALID_RESULT
        ),
    )

    assert decision.retry is False


def test_mapping_policy_can_retry_selected_kind() -> None:
    policy = MappingRecoveryPolicy(
        {
            "backend_error": True,
            "verification_failed": False,
        },
        delay_seconds=0.5,
    )

    decision = policy.decide(
        _request(),
        _failed_attempt(
            kind=RecoveryFailureKind.BACKEND_ERROR
        ),
    )

    assert decision.retry is True
    assert decision.next_attempt == 2
    assert decision.delay_seconds == 0.5


def test_mapping_policy_can_disable_selected_kind() -> None:
    policy = MappingRecoveryPolicy(
        {
            "backend_error": False,
        }
    )

    decision = policy.decide(
        _request(),
        _failed_attempt(),
    )

    assert decision.retry is False
    assert decision.reason == (
        "failure_kind_not_retryable: backend_error"
    )


def test_callback_policy_returns_callback_decision() -> None:
    calls = []

    def callback(
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        calls.append(
            (
                request.request_id,
                attempt.attempt,
            )
        )

        return RecoveryDecision(
            retry=True,
            reason="callback_retry",
            failure_kind=(
                RecoveryFailureKind.RETRYABLE
            ),
            next_attempt=attempt.attempt + 1,
        )

    policy = CallbackRecoveryPolicy(
        callback
    )

    decision = policy.decide(
        _request(),
        _failed_attempt(),
    )

    assert decision.retry is True
    assert calls == [("req-1", 1)]


def test_callback_policy_rejects_invalid_return() -> None:
    policy = CallbackRecoveryPolicy(
        lambda request, attempt: "invalid"
    )

    with pytest.raises(
        RecoveryPolicyError,
        match="must return RecoveryDecision",
    ):
        policy.decide(
            _request(),
            _failed_attempt(),
        )


def test_callback_policy_rejects_out_of_range_next_attempt() -> None:
    policy = CallbackRecoveryPolicy(
        lambda request, attempt: RecoveryDecision(
            retry=True,
            next_attempt=99,
        )
    )

    with pytest.raises(
        RecoveryPolicyError,
        match="cannot exceed",
    ):
        policy.decide(
            _request(max_attempts=3),
            _failed_attempt(max_attempts=3),
        )


def test_never_retry_policy_never_retries() -> None:
    policy = NeverRetryRecoveryPolicy()

    decision = policy.decide(
        _request(),
        _failed_attempt(),
    )

    assert decision.retry is False
    assert decision.reason == "recovery_disabled"


def test_policy_config_normalizes_string_kinds() -> None:
    config = RecoveryPolicyConfig(
        retryable_kinds=frozenset(
            {
                "backend_error",
                RecoveryFailureKind.VERIFICATION_FAILED,
            }
        )
    )

    assert config.retryable_kinds == frozenset(
        {
            RecoveryFailureKind.BACKEND_ERROR,
            RecoveryFailureKind.VERIFICATION_FAILED,
        }
    )


def test_policy_config_rejects_negative_delay() -> None:
    with pytest.raises(
        RecoveryPolicyError,
        match="non-negative",
    ):
        RecoveryPolicyConfig(
            default_delay_seconds=-1
        )


def test_policy_config_rejects_invalid_kind() -> None:
    with pytest.raises(
        RecoveryPolicyError,
        match="invalid",
    ):
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    "not-a-real-kind",
                }
            )
        )


def test_policy_rejects_invalid_request() -> None:
    policy = DefaultRecoveryPolicy()

    with pytest.raises(
        RecoveryPolicyError,
        match="request must be",
    ):
        policy.decide(
            "invalid",
            _failed_attempt(),
        )


def test_policy_rejects_attempt_above_request_limit() -> None:
    policy = DefaultRecoveryPolicy()

    request = _request(
        attempt=1,
        max_attempts=2,
    )

    attempt = _failed_attempt(
        attempt=3,
        max_attempts=3,
    )

    with pytest.raises(
        RecoveryPolicyError,
        match="cannot exceed request.max_attempts",
    ):
        policy.decide(
            request,
            attempt,
        )


def test_mapping_policy_rejects_non_boolean_rule() -> None:
    with pytest.raises(
        RecoveryPolicyError,
        match="must be booleans",
    ):
        MappingRecoveryPolicy(
            {
                RecoveryFailureKind.BACKEND_ERROR: "yes",
            }
        )


def test_mapping_policy_rejects_negative_delay() -> None:
    with pytest.raises(
        RecoveryPolicyError,
        match="non-negative",
    ):
        MappingRecoveryPolicy(
            {},
            delay_seconds=-0.1,
        )


def test_callback_policy_requires_callable() -> None:
    with pytest.raises(
        RecoveryPolicyError,
        match="callable",
    ):
        CallbackRecoveryPolicy(
            "not-callable"
        )


def test_retry_decision_always_points_to_next_attempt() -> None:
    policy = DefaultRecoveryPolicy(
        RecoveryPolicyConfig(
            retryable_kinds=frozenset(
                {
                    RecoveryFailureKind.BACKEND_ERROR,
                }
            )
        )
    )

    decision = policy.decide(
        _request(attempt=2, max_attempts=4),
        _failed_attempt(attempt=2, max_attempts=4),
    )

    assert decision.retry is True
    assert decision.next_attempt == 3
