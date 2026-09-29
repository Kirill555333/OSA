from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryRequest,
)
from osa.recovery_policy import (
    PERMANENT_FAILURE_KINDS,
    RecoveryPolicy,
)


class RecoveryGuardrailError(ValueError):
    """Raised when recovery safety guardrail configuration is invalid."""


class RecoveryRetryAuthorizer(Protocol):
    """Authorize a recovery retry before it is accepted."""

    def authorize(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> bool:
        ...


@dataclass(frozen=True)
class RecoveryGuardrailConfig:
    """Immutable safety limits for retry decisions."""

    max_delay_seconds: float = 60.0
    require_sequential_attempts: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.max_delay_seconds,
            (int, float),
        ):
            raise RecoveryGuardrailError(
                "max_delay_seconds must be numeric."
            )

        if not math.isfinite(
            float(self.max_delay_seconds)
        ):
            raise RecoveryGuardrailError(
                "max_delay_seconds must be finite."
            )

        if self.max_delay_seconds < 0:
            raise RecoveryGuardrailError(
                "max_delay_seconds cannot be negative."
            )

        if not isinstance(
            self.require_sequential_attempts,
            bool,
        ):
            raise RecoveryGuardrailError(
                "require_sequential_attempts must be boolean."
            )


class RecoverySafetyGuard:
    """
    Validate recovery decisions before retry is accepted.

    The guard is deliberately fail-closed. It never executes a backend,
    changes a RecoveryResult, or mutates the request/attempt.
    """

    def __init__(
        self,
        *,
        config: RecoveryGuardrailConfig | None = None,
        authorizer: RecoveryRetryAuthorizer | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else RecoveryGuardrailConfig()
        )
        self._authorizer = authorizer

    @property
    def config(self) -> RecoveryGuardrailConfig:
        return self._config

    @property
    def authorizer(
        self,
    ) -> RecoveryRetryAuthorizer | None:
        return self._authorizer

    def validate(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
        decision: RecoveryDecision,
    ) -> RecoveryDecision:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise TypeError(
                "request must be a RecoveryRequest."
            )

        if not isinstance(
            attempt,
            RecoveryAttempt,
        ):
            raise TypeError(
                "attempt must be a RecoveryAttempt."
            )

        if not isinstance(
            decision,
            RecoveryDecision,
        ):
            raise TypeError(
                "decision must be a RecoveryDecision."
            )

        if attempt.success:
            return self._deny_retry(
                attempt,
                "Successful recovery attempts cannot be retried.",
            )

        if attempt.failure_kind in PERMANENT_FAILURE_KINDS:
            return self._deny_retry(
                attempt,
                (
                    "Permanent recovery failure kinds "
                    "cannot be retried."
                ),
            )

        if decision.failure_kind != attempt.failure_kind:
            return self._deny_retry(
                attempt,
                (
                    "Recovery decision failure_kind does not "
                    "match the recovery attempt."
                ),
            )

        if not decision.retry:
            return self._normalize_no_retry(
                attempt,
                decision,
            )

        if attempt.attempt >= attempt.max_attempts:
            return self._deny_retry(
                attempt,
                "Maximum recovery attempts have been reached.",
            )

        try:
            delay = float(
                decision.delay_seconds
            )
        except (TypeError, ValueError):
            return self._deny_retry(
                attempt,
                "Recovery retry delay must be numeric.",
            )

        if not math.isfinite(delay):
            return self._deny_retry(
                attempt,
                "Recovery retry delay must be finite.",
            )

        if delay < 0:
            return self._deny_retry(
                attempt,
                "Recovery retry delay cannot be negative.",
            )

        if delay > self._config.max_delay_seconds:
            return self._deny_retry(
                attempt,
                (
                    "Recovery retry delay exceeds the "
                    "configured safety limit."
                ),
            )

        next_attempt = decision.next_attempt

        if next_attempt is None:
            return self._deny_retry(
                attempt,
                "Retry decisions must define next_attempt.",
            )

        if (
            self._config.require_sequential_attempts
            and next_attempt != attempt.attempt + 1
        ):
            return self._deny_retry(
                attempt,
                (
                    "Recovery retries must advance exactly "
                    "one attempt at a time."
                ),
            )

        if next_attempt <= attempt.attempt:
            return self._deny_retry(
                attempt,
                "next_attempt must be greater than the current attempt.",
            )

        if next_attempt > attempt.max_attempts:
            return self._deny_retry(
                attempt,
                "next_attempt cannot exceed max_attempts.",
            )

        if self._authorizer is not None:
            try:
                authorized = self._authorizer.authorize(
                    request,
                    attempt,
                )
            except Exception:
                return self._deny_retry(
                    attempt,
                    "Recovery retry authorization failed closed.",
                )

            if not authorized:
                return self._deny_retry(
                    attempt,
                    "Recovery retry authorization was denied.",
                )

        return decision

    @staticmethod
    def _deny_retry(
        attempt: RecoveryAttempt,
        reason: str,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            retry=False,
            reason=reason,
            failure_kind=attempt.failure_kind,
            delay_seconds=0.0,
            next_attempt=None,
        )

    @staticmethod
    def _normalize_no_retry(
        attempt: RecoveryAttempt,
        decision: RecoveryDecision,
    ) -> RecoveryDecision:
        reason = decision.reason or "Recovery retry was not requested."

        return RecoveryDecision(
            retry=False,
            reason=reason,
            failure_kind=attempt.failure_kind,
            delay_seconds=0.0,
            next_attempt=None,
        )


class GuardedRecoveryPolicy:
    """
    Wrap an existing recovery policy with non-bypassable safety checks.
    """

    def __init__(
        self,
        policy: RecoveryPolicy,
        *,
        guard: RecoverySafetyGuard | None = None,
    ) -> None:
        if policy is None:
            raise ValueError(
                "policy is required."
            )

        self._policy = policy
        self._guard = (
            guard
            if guard is not None
            else RecoverySafetyGuard()
        )

    @property
    def policy(self) -> RecoveryPolicy:
        return self._policy

    @property
    def guard(self) -> RecoverySafetyGuard:
        return self._guard

    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise TypeError(
                "request must be a RecoveryRequest."
            )

        if not isinstance(
            attempt,
            RecoveryAttempt,
        ):
            raise TypeError(
                "attempt must be a RecoveryAttempt."
            )

        try:
            decision = self._policy.decide(
                request,
                attempt,
            )
        except Exception:
            return RecoveryDecision(
                retry=False,
                reason="Recovery policy failed closed.",
                failure_kind=attempt.failure_kind,
                delay_seconds=0.0,
                next_attempt=None,
            )

        if not isinstance(
            decision,
            RecoveryDecision,
        ):
            return RecoveryDecision(
                retry=False,
                reason="Recovery policy returned an invalid decision.",
                failure_kind=attempt.failure_kind,
                delay_seconds=0.0,
                next_attempt=None,
            )

        return self._guard.validate(
            request,
            attempt,
            decision,
        )
