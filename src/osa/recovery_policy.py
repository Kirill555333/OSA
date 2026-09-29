from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryFailureKind,
    RecoveryRequest,
)


class RecoveryPolicyError(ValueError):
    """Raised when a recovery policy is invalid."""


PERMANENT_FAILURE_KINDS = frozenset(
    {
        RecoveryFailureKind.PERMISSION_DENIED,
        RecoveryFailureKind.CONFIRMATION_DENIED,
        RecoveryFailureKind.INVALID_RESULT,
        RecoveryFailureKind.MAX_ATTEMPTS,
    }
)


class RecoveryPolicy(Protocol):
    """Decide whether the current failed attempt may be retried."""

    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        ...


@dataclass(frozen=True, slots=True)
class RecoveryPolicyConfig:
    """
    Configuration for the default bounded recovery policy.

    Only explicitly configured retryable categories may be retried, except
    permanent failure categories which are always fail-closed.
    """

    retryable_kinds: frozenset[RecoveryFailureKind] = frozenset()
    default_delay_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(
            self.retryable_kinds,
            frozenset,
        ):
            raise RecoveryPolicyError(
                "retryable_kinds must be a frozenset"
            )

        normalized: set[RecoveryFailureKind] = set()

        for item in self.retryable_kinds:
            if isinstance(item, RecoveryFailureKind):
                normalized.add(item)
                continue

            try:
                normalized.add(
                    RecoveryFailureKind(item)
                )
            except (TypeError, ValueError) as exc:
                raise RecoveryPolicyError(
                    "retryable_kinds contains an invalid "
                    "failure kind"
                ) from exc

        if isinstance(
            self.default_delay_seconds,
            bool,
        ) or not isinstance(
            self.default_delay_seconds,
            (int, float),
        ):
            raise RecoveryPolicyError(
                "default_delay_seconds must be a number"
            )

        delay = float(
            self.default_delay_seconds
        )

        if delay < 0.0:
            raise RecoveryPolicyError(
                "default_delay_seconds must be non-negative"
            )

        object.__setattr__(
            self,
            "retryable_kinds",
            frozenset(normalized),
        )
        object.__setattr__(
            self,
            "default_delay_seconds",
            delay,
        )


class DefaultRecoveryPolicy:
    """
    Fail-closed recovery policy.

    Permanent failure categories can never be retried, even if a caller
    accidentally includes them in retryable_kinds.
    """

    def __init__(
        self,
        config: RecoveryPolicyConfig | None = None,
    ) -> None:
        self._config = (
            config
            if config is not None
            else RecoveryPolicyConfig()
        )

    @property
    def config(self) -> RecoveryPolicyConfig:
        return self._config

    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        self._validate_inputs(
            request,
            attempt,
        )

        if attempt.success:
            return RecoveryDecision(
                retry=False,
                reason="attempt_succeeded",
            )

        failure_kind = attempt.failure_kind

        if failure_kind is None:
            raise RecoveryPolicyError(
                "failed attempt must have failure_kind"
            )

        if attempt.attempt >= request.max_attempts:
            return RecoveryDecision(
                retry=False,
                reason="maximum_attempts_reached",
                failure_kind=RecoveryFailureKind.MAX_ATTEMPTS,
            )

        if failure_kind in PERMANENT_FAILURE_KINDS:
            return RecoveryDecision(
                retry=False,
                reason=(
                    "failure_kind_permanently_non_retryable: "
                    f"{failure_kind.value}"
                ),
                failure_kind=failure_kind,
            )

        if (
            failure_kind
            not in self._config.retryable_kinds
        ):
            return RecoveryDecision(
                retry=False,
                reason=(
                    "failure_kind_not_retryable: "
                    f"{failure_kind.value}"
                ),
                failure_kind=failure_kind,
            )

        next_attempt = attempt.attempt + 1

        if next_attempt > request.max_attempts:
            return RecoveryDecision(
                retry=False,
                reason="maximum_attempts_reached",
                failure_kind=RecoveryFailureKind.MAX_ATTEMPTS,
            )

        return RecoveryDecision(
            retry=True,
            reason=(
                "failure_kind_retryable: "
                f"{failure_kind.value}"
            ),
            failure_kind=failure_kind,
            delay_seconds=(
                self._config.default_delay_seconds
            ),
            next_attempt=next_attempt,
        )

    @staticmethod
    def _validate_inputs(
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> None:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise RecoveryPolicyError(
                "request must be a RecoveryRequest"
            )

        if not isinstance(
            attempt,
            RecoveryAttempt,
        ):
            raise RecoveryPolicyError(
                "attempt must be a RecoveryAttempt"
            )

        if attempt.attempt > request.max_attempts:
            raise RecoveryPolicyError(
                "attempt cannot exceed request.max_attempts"
            )


class MappingRecoveryPolicy:
    """
    Explicit failure-kind policy.

    Permanent failure categories remain fail-closed regardless of mappings.
    """

    def __init__(
        self,
        rules: Mapping[
            RecoveryFailureKind | str,
            bool,
        ],
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        if not isinstance(
            rules,
            Mapping,
        ):
            raise RecoveryPolicyError(
                "rules must be a mapping"
            )

        normalized_rules: dict[
            RecoveryFailureKind,
            bool,
        ] = {}

        for key, value in rules.items():
            if isinstance(
                key,
                RecoveryFailureKind,
            ):
                failure_kind = key
            else:
                try:
                    failure_kind = RecoveryFailureKind(
                        key
                    )
                except (TypeError, ValueError) as exc:
                    raise RecoveryPolicyError(
                        f"invalid recovery failure kind: {key!r}"
                    ) from exc

            if not isinstance(
                value,
                bool,
            ):
                raise RecoveryPolicyError(
                    "recovery rule values must be booleans"
                )

            normalized_rules[failure_kind] = value

        if isinstance(
            delay_seconds,
            bool,
        ) or not isinstance(
            delay_seconds,
            (int, float),
        ):
            raise RecoveryPolicyError(
                "delay_seconds must be a number"
            )

        normalized_delay = float(
            delay_seconds
        )

        if normalized_delay < 0.0:
            raise RecoveryPolicyError(
                "delay_seconds must be non-negative"
            )

        self._rules = normalized_rules
        self._delay_seconds = normalized_delay

    @property
    def rules(self) -> Mapping[
        RecoveryFailureKind,
        bool,
    ]:
        return dict(self._rules)

    @property
    def delay_seconds(self) -> float:
        return self._delay_seconds

    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise RecoveryPolicyError(
                "request must be a RecoveryRequest"
            )

        if not isinstance(
            attempt,
            RecoveryAttempt,
        ):
            raise RecoveryPolicyError(
                "attempt must be a RecoveryAttempt"
            )

        if attempt.success:
            return RecoveryDecision(
                retry=False,
                reason="attempt_succeeded",
            )

        failure_kind = attempt.failure_kind

        if failure_kind is None:
            raise RecoveryPolicyError(
                "failed attempt must have failure_kind"
            )

        if attempt.attempt >= request.max_attempts:
            return RecoveryDecision(
                retry=False,
                reason="maximum_attempts_reached",
                failure_kind=RecoveryFailureKind.MAX_ATTEMPTS,
            )

        if failure_kind in PERMANENT_FAILURE_KINDS:
            return RecoveryDecision(
                retry=False,
                reason=(
                    "failure_kind_permanently_non_retryable: "
                    f"{failure_kind.value}"
                ),
                failure_kind=failure_kind,
            )

        retry = self._rules.get(
            failure_kind,
            False,
        )

        if not retry:
            return RecoveryDecision(
                retry=False,
                reason=(
                    "failure_kind_not_retryable: "
                    f"{failure_kind.value}"
                ),
                failure_kind=failure_kind,
            )

        return RecoveryDecision(
            retry=True,
            reason=(
                "failure_kind_retryable: "
                f"{failure_kind.value}"
            ),
            failure_kind=failure_kind,
            delay_seconds=self._delay_seconds,
            next_attempt=attempt.attempt + 1,
        )


class CallbackRecoveryPolicy:
    """Callback-backed recovery policy."""

    def __init__(
        self,
        callback: Callable[
            [RecoveryRequest, RecoveryAttempt],
            RecoveryDecision,
        ],
    ) -> None:
        if not callable(callback):
            raise RecoveryPolicyError(
                "callback must be callable"
            )

        self._callback = callback

    @property
    def callback(self):
        return self._callback

    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise RecoveryPolicyError(
                "request must be a RecoveryRequest"
            )

        if not isinstance(
            attempt,
            RecoveryAttempt,
        ):
            raise RecoveryPolicyError(
                "attempt must be a RecoveryAttempt"
            )

        decision = self._callback(
            request,
            attempt,
        )

        if not isinstance(
            decision,
            RecoveryDecision,
        ):
            raise RecoveryPolicyError(
                "callback must return RecoveryDecision"
            )

        if decision.retry:
            if decision.next_attempt is None:
                raise RecoveryPolicyError(
                    "retry decision requires next_attempt"
                )

            if (
                attempt.failure_kind
                in PERMANENT_FAILURE_KINDS
            ):
                raise RecoveryPolicyError(
                    "permanent failure category cannot be retried"
                )

            if (
                decision.next_attempt
                > request.max_attempts
            ):
                raise RecoveryPolicyError(
                    "next_attempt cannot exceed "
                    "request.max_attempts"
                )

        return decision


class NeverRetryRecoveryPolicy:
    """Explicit fail-closed policy that never requests a retry."""

    def decide(
        self,
        request: RecoveryRequest,
        attempt: RecoveryAttempt,
    ) -> RecoveryDecision:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise RecoveryPolicyError(
                "request must be a RecoveryRequest"
            )

        if not isinstance(
            attempt,
            RecoveryAttempt,
        ):
            raise RecoveryPolicyError(
                "attempt must be a RecoveryAttempt"
            )

        return RecoveryDecision(
            retry=False,
            reason="recovery_disabled",
            failure_kind=(
                attempt.failure_kind
                if not attempt.success
                else None
            ),
        )
