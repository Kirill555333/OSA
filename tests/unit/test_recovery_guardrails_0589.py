from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryContractError,
    RecoveryDecision,
    RecoveryFailureKind,
    RecoveryRequest,
)
from osa.recovery_guardrails import (
    GuardedRecoveryPolicy,
    RecoveryGuardrailConfig,
    RecoveryGuardrailError,
    RecoverySafetyGuard,
)
from osa.recovery_policy import (
    NeverRetryRecoveryPolicy,
)


def _request() -> RecoveryRequest:
    return RecoveryRequest(
        run_id="run-1",
        task_id="task-1",
        request_id="req-1",
        action_kind="tool",
        action_name="demo_tool",
        attempt=1,
        max_attempts=3,
    )


def _attempt(
    *,
    attempt: int = 1,
    max_attempts: int = 3,
    success: bool = False,
    failure_kind: RecoveryFailureKind = (
        RecoveryFailureKind.RETRYABLE
    ),
) -> RecoveryAttempt:
    if success:
        return RecoveryAttempt(
            attempt=attempt,
            max_attempts=max_attempts,
            success=True,
            failure_kind=None,
            error=None,
            output_present=True,
            metadata={},
        )

    return RecoveryAttempt(
        attempt=attempt,
        max_attempts=max_attempts,
        success=False,
        failure_kind=failure_kind,
        error="temporary failure",
        output_present=False,
        metadata={},
    )


def _decision(
    *,
    retry: bool = True,
    failure_kind: RecoveryFailureKind = (
        RecoveryFailureKind.RETRYABLE
    ),
    delay_seconds: float = 0.0,
    next_attempt: int | None = 2,
) -> RecoveryDecision:
    return RecoveryDecision(
        retry=retry,
        reason=(
            "retry requested"
            if retry
            else "do not retry"
        ),
        failure_kind=failure_kind,
        delay_seconds=delay_seconds,
        next_attempt=next_attempt,
    )


@dataclass
class AllowingAuthorizer:
    calls: int = 0

    def authorize(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> bool:
        self.calls += 1
        return True


@dataclass
class DenyingAuthorizer:
    calls: int = 0

    def authorize(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> bool:
        self.calls += 1
        return False


class RaisingAuthorizer:
    def authorize(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> bool:
        raise RuntimeError(
            "authorization unavailable"
        )


class RetryPolicy:
    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        return _decision()


class PermanentRetryPolicy:
    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        return _decision(
            failure_kind=attempt.failure_kind
        )


def test_valid_retry_is_allowed():
    authorizer = AllowingAuthorizer()

    guard = RecoverySafetyGuard(
        authorizer=authorizer,
    )

    attempt = _attempt()

    decision = guard.validate(
        _request(),
        attempt,
        _decision(),
    )

    assert decision.retry is True
    assert decision.next_attempt == 2
    assert authorizer.calls == 1


def test_permanent_failure_is_never_retryable():
    guard = RecoverySafetyGuard()

    for failure_kind in (
        RecoveryFailureKind.PERMISSION_DENIED,
        RecoveryFailureKind.CONFIRMATION_DENIED,
        RecoveryFailureKind.INVALID_RESULT,
        RecoveryFailureKind.MAX_ATTEMPTS,
    ):
        attempt = _attempt(
            failure_kind=failure_kind,
        )

        decision = guard.validate(
            _request(),
            attempt,
            _decision(
                failure_kind=failure_kind,
            ),
        )

        assert decision.retry is False
        assert decision.next_attempt is None
        assert decision.failure_kind == failure_kind


def test_successful_attempt_is_never_retryable():
    guard = RecoverySafetyGuard()

    attempt = _attempt(
        success=True,
    )

    decision = guard.validate(
        _request(),
        attempt,
        _decision(),
    )

    assert decision.retry is False
    assert decision.next_attempt is None


def test_max_attempts_is_never_retryable():
    guard = RecoverySafetyGuard()

    attempt = _attempt(
        attempt=3,
        max_attempts=3,
    )

    decision = guard.validate(
        _request(),
        attempt,
        _decision(
            next_attempt=4,
        ),
    )

    assert decision.retry is False
    assert decision.next_attempt is None


def test_retry_must_be_sequential():
    guard = RecoverySafetyGuard()

    attempt = _attempt(
        attempt=1,
        max_attempts=4,
    )

    decision = guard.validate(
        _request(),
        attempt,
        _decision(
            next_attempt=3,
        ),
    )

    assert decision.retry is False
    assert decision.next_attempt is None
    assert "exactly one" in (
        decision.reason or ""
    )


def test_retry_cannot_exceed_budget():
    guard = RecoverySafetyGuard(
        config=RecoveryGuardrailConfig(
            require_sequential_attempts=False,
        ),
    )

    attempt = _attempt(
        attempt=1,
        max_attempts=2,
    )

    decision = guard.validate(
        _request(),
        attempt,
        _decision(
            next_attempt=5,
        ),
    )

    assert decision.retry is False
    assert decision.next_attempt is None


@pytest.mark.parametrize(
    "delay",
    [
        -1.0,
        float("-inf"),
    ],
)
def test_invalid_negative_delays_are_rejected_by_contract(
    delay: float,
):
    with pytest.raises(
        RecoveryContractError,
        match="delay_seconds",
    ):
        _decision(
            delay_seconds=delay,
        )


@pytest.mark.parametrize(
    "delay",
    [
        float("inf"),
        float("nan"),
    ],
)
def test_invalid_non_finite_delays_are_rejected_by_guard(
    delay: float,
):
    guard = RecoverySafetyGuard()

    decision = guard.validate(
        _request(),
        _attempt(),
        _decision(
            delay_seconds=delay,
        ),
    )

    assert decision.retry is False
    assert decision.next_attempt is None


def test_delay_above_configured_limit_is_rejected():
    guard = RecoverySafetyGuard(
        config=RecoveryGuardrailConfig(
            max_delay_seconds=1.0,
        ),
    )

    decision = guard.validate(
        _request(),
        _attempt(),
        _decision(
            delay_seconds=1.01,
        ),
    )

    assert decision.retry is False
    assert decision.next_attempt is None


def test_failure_kind_mismatch_is_rejected():
    guard = RecoverySafetyGuard()

    attempt = _attempt(
        failure_kind=RecoveryFailureKind.BACKEND_ERROR,
    )

    decision = guard.validate(
        _request(),
        attempt,
        _decision(
            failure_kind=RecoveryFailureKind.RETRYABLE,
        ),
    )

    assert decision.retry is False
    assert decision.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )


def test_missing_next_attempt_is_rejected_by_contract():
    with pytest.raises(
        RecoveryContractError,
        match="next_attempt",
    ):
        _decision(
            retry=True,
            next_attempt=None,
        )


def test_denying_authorizer_blocks_retry():
    authorizer = DenyingAuthorizer()

    guard = RecoverySafetyGuard(
        authorizer=authorizer,
    )

    decision = guard.validate(
        _request(),
        _attempt(),
        _decision(),
    )

    assert decision.retry is False
    assert decision.next_attempt is None
    assert authorizer.calls == 1


def test_authorizer_exception_fails_closed():
    guard = RecoverySafetyGuard(
        authorizer=RaisingAuthorizer(),
    )

    decision = guard.validate(
        _request(),
        _attempt(),
        _decision(),
    )

    assert decision.retry is False
    assert decision.next_attempt is None


def test_no_retry_decision_is_normalized():
    guard = RecoverySafetyGuard()

    decision = guard.validate(
        _request(),
        _attempt(),
        _decision(
            retry=False,
            next_attempt=None,
        ),
    )

    assert decision.retry is False
    assert decision.next_attempt is None
    assert decision.delay_seconds == 0.0
    assert decision.failure_kind == (
        RecoveryFailureKind.RETRYABLE
    )


def test_guarded_policy_allows_safe_retry():
    authorizer = AllowingAuthorizer()

    policy = GuardedRecoveryPolicy(
        RetryPolicy(),
        guard=RecoverySafetyGuard(
            authorizer=authorizer,
        ),
    )

    decision = policy.decide(
        _request(),
        _attempt(),
    )

    assert decision.retry is True
    assert decision.next_attempt == 2
    assert authorizer.calls == 1


def test_guarded_policy_blocks_bad_policy_decision():
    class BadPolicy:
        def decide(
            self,
            request: RecoveryRequest,
            attempt: RecoveryAttempt,
        ) -> RecoveryDecision:
            return _decision(
                next_attempt=99,
            )

    policy = GuardedRecoveryPolicy(
        BadPolicy(),
    )

    decision = policy.decide(
        _request(),
        _attempt(),
    )

    assert decision.retry is False
    assert decision.next_attempt is None


def test_guarded_policy_fails_closed_on_policy_exception():
    class RaisingPolicy:
        def decide(
            self,
            request: RecoveryRequest,
            attempt: RecoveryAttempt,
        ) -> RecoveryDecision:
            raise RuntimeError(
                "policy unavailable"
            )

    policy = GuardedRecoveryPolicy(
        RaisingPolicy(),
    )

    decision = policy.decide(
        _request(),
        _attempt(
            failure_kind=RecoveryFailureKind.BACKEND_ERROR,
        ),
    )

    assert decision.retry is False
    assert decision.failure_kind == (
        RecoveryFailureKind.BACKEND_ERROR
    )
    assert "failed closed" in (
        decision.reason or ""
    )


def test_guarded_policy_rejects_invalid_policy_result():
    class InvalidPolicy:
        def decide(
            self,
            request: RecoveryRequest,
            attempt: RecoveryAttempt,
        ):
            return object()

    policy = GuardedRecoveryPolicy(
        InvalidPolicy(),
    )

    decision = policy.decide(
        _request(),
        _attempt(),
    )

    assert decision.retry is False
    assert decision.next_attempt is None
    assert "invalid decision" in (
        decision.reason or ""
    )


def test_invalid_guardrail_config_is_rejected():
    with pytest.raises(
        RecoveryGuardrailError,
        match="finite",
    ):
        RecoveryGuardrailConfig(
            max_delay_seconds=float("inf"),
        )

    with pytest.raises(
        RecoveryGuardrailError,
        match="negative",
    ):
        RecoveryGuardrailConfig(
            max_delay_seconds=-1,
        )


def test_guard_rejects_invalid_types():
    guard = RecoverySafetyGuard()

    with pytest.raises(
        TypeError,
        match="RecoveryRequest",
    ):
        guard.validate(
            object(),
            _attempt(),
            _decision(),
        )

    with pytest.raises(
        TypeError,
        match="RecoveryAttempt",
    ):
        guard.validate(
            _request(),
            object(),
            _decision(),
        )

    with pytest.raises(
        TypeError,
        match="RecoveryDecision",
    ):
        guard.validate(
            _request(),
            _attempt(),
            object(),
        )


def test_never_retry_policy_remains_compatible_with_guard():
    policy = GuardedRecoveryPolicy(
        NeverRetryRecoveryPolicy(),
    )

    decision = policy.decide(
        _request(),
        _attempt(),
    )

    assert decision.retry is False
    assert decision.next_attempt is None
    assert decision.failure_kind == (
        RecoveryFailureKind.RETRYABLE
    )
