from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from time import sleep
from typing import Any

from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryDecision,
    RecoveryFailureKind,
    RecoveryRequest,
    RecoveryResult,
)
from osa.recovery_policy import (
    DefaultRecoveryPolicy,
    RecoveryPolicy,
)


class RecoveryControllerError(RuntimeError):
    """Raised when the recovery controller cannot continue safely."""


RecoveryExecutor = Callable[
    [RecoveryRequest],
    Any,
]

RecoveryAttemptBuilder = Callable[
    [RecoveryRequest, Any],
    RecoveryAttempt,
]

RecoverySleeper = Callable[
    [float],
    None,
]


class RecoveryController:
    """
    Bounded retry controller.

    The controller owns retry orchestration only. It never executes a
    backend directly. Every retry calls the same injected executor again.
    """

    def __init__(
        self,
        executor: RecoveryExecutor,
        *,
        attempt_builder: RecoveryAttemptBuilder,
        policy: RecoveryPolicy | None = None,
        sleeper: RecoverySleeper | None = None,
    ) -> None:
        if not callable(executor):
            raise RecoveryControllerError(
                "executor must be callable"
            )

        if not callable(attempt_builder):
            raise RecoveryControllerError(
                "attempt_builder must be callable"
            )

        if sleeper is None:
            sleeper = sleep

        if not callable(sleeper):
            raise RecoveryControllerError(
                "sleeper must be callable"
            )

        self._executor = executor
        self._attempt_builder = attempt_builder
        self._policy = (
            policy
            if policy is not None
            else DefaultRecoveryPolicy()
        )
        self._sleeper = sleeper

    @property
    def executor(self) -> RecoveryExecutor:
        return self._executor

    @property
    def attempt_builder(
        self,
    ) -> RecoveryAttemptBuilder:
        return self._attempt_builder

    @property
    def policy(self) -> RecoveryPolicy:
        return self._policy

    @property
    def sleeper(self) -> RecoverySleeper:
        return self._sleeper

    def execute(
        self,
        request: RecoveryRequest,
    ) -> RecoveryResult:
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise RecoveryControllerError(
                "request must be a RecoveryRequest"
            )

        current_request = request
        attempts: list[RecoveryAttempt] = []
        last_result: Any | None = None

        while True:
            execution_failed = False

            try:
                raw_result = self._executor(
                    current_request
                )
            except Exception as exc:
                execution_failed = True
                raw_result = None

                attempt = RecoveryAttempt(
                    attempt=current_request.attempt,
                    max_attempts=current_request.max_attempts,
                    success=False,
                    failure_kind=(
                        RecoveryFailureKind.BACKEND_ERROR
                    ),
                    error=str(exc),
                    output_present=False,
                )
            else:
                last_result = raw_result

                try:
                    attempt = self._attempt_builder(
                        current_request,
                        raw_result,
                    )
                except Exception as exc:
                    return RecoveryResult.failed(
                        failure_kind=(
                            RecoveryFailureKind.INVALID_RESULT
                        ),
                        error=(
                            "recovery_attempt_builder_failed: "
                            f"{exc}"
                        ),
                        attempts=tuple(attempts),
                        exhausted=False,
                        result=raw_result,
                    )

                if not isinstance(
                    attempt,
                    RecoveryAttempt,
                ):
                    return RecoveryResult.failed(
                        failure_kind=(
                            RecoveryFailureKind.INVALID_RESULT
                        ),
                        error=(
                            "recovery_attempt_builder_failed: "
                            "invalid RecoveryAttempt."
                        ),
                        attempts=tuple(attempts),
                        exhausted=False,
                        result=raw_result,
                    )

            attempts.append(attempt)

            if attempt.success:
                return RecoveryResult.succeeded(
                    result=raw_result,
                    attempts=tuple(attempts),
                )

            if current_request.attempt >= (
                current_request.max_attempts
            ):
                return RecoveryResult.failed(
                    failure_kind=(
                        attempt.failure_kind
                        or RecoveryFailureKind.NON_RETRYABLE
                    ),
                    error=(
                        attempt.error
                        or (
                            "maximum_attempts_reached"
                        )
                    ),
                    attempts=tuple(attempts),
                    exhausted=True,
                    result=(
                        last_result
                        if raw_result is None
                        else raw_result
                    ),
                )

            try:
                decision = self._policy.decide(
                    current_request,
                    attempt,
                )
            except Exception as exc:
                return RecoveryResult.failed(
                    failure_kind=(
                        RecoveryFailureKind.NON_RETRYABLE
                    ),
                    error=(
                        "recovery_policy_failed: "
                        f"{exc}"
                    ),
                    attempts=tuple(attempts),
                    exhausted=False,
                    result=(
                        last_result
                        if raw_result is None
                        else raw_result
                    ),
                )

            if not isinstance(
                decision,
                RecoveryDecision,
            ):
                return RecoveryResult.failed(
                    failure_kind=(
                        RecoveryFailureKind.NON_RETRYABLE
                    ),
                    error=(
                        "recovery_policy_failed: "
                        "invalid RecoveryDecision."
                    ),
                    attempts=tuple(attempts),
                    exhausted=False,
                    result=(
                        last_result
                        if raw_result is None
                        else raw_result
                    ),
                )

            if not decision.retry:
                return RecoveryResult.failed(
                    failure_kind=(
                        attempt.failure_kind
                        or RecoveryFailureKind.NON_RETRYABLE
                    ),
                    error=(
                        attempt.error
                        or decision.reason
                        or "recovery_stopped"
                    ),
                    attempts=tuple(attempts),
                    exhausted=False,
                    result=(
                        last_result
                        if raw_result is None
                        else raw_result
                    ),
                )

            if decision.next_attempt is None:
                return RecoveryResult.failed(
                    failure_kind=(
                        RecoveryFailureKind.NON_RETRYABLE
                    ),
                    error=(
                        "recovery_policy_failed: "
                        "retry decision has no next_attempt."
                    ),
                    attempts=tuple(attempts),
                    exhausted=False,
                    result=(
                        last_result
                        if raw_result is None
                        else raw_result
                    ),
                )

            if decision.next_attempt <= (
                current_request.attempt
            ):
                return RecoveryResult.failed(
                    failure_kind=(
                        RecoveryFailureKind.NON_RETRYABLE
                    ),
                    error=(
                        "recovery_policy_failed: "
                        "next_attempt must increase."
                    ),
                    attempts=tuple(attempts),
                    exhausted=False,
                    result=(
                        last_result
                        if raw_result is None
                        else raw_result
                    ),
                )

            if decision.next_attempt > (
                current_request.max_attempts
            ):
                return RecoveryResult.failed(
                    failure_kind=(
                        RecoveryFailureKind.NON_RETRYABLE
                    ),
                    error=(
                        "recovery_policy_failed: "
                        "next_attempt exceeds max_attempts."
                    ),
                    attempts=tuple(attempts),
                    exhausted=False,
                    result=(
                        last_result
                        if raw_result is None
                        else raw_result
                    ),
                )

            if decision.delay_seconds > 0.0:
                try:
                    self._sleeper(
                        decision.delay_seconds
                    )
                except Exception as exc:
                    return RecoveryResult.failed(
                        failure_kind=(
                            RecoveryFailureKind.NON_RETRYABLE
                        ),
                        error=(
                            "recovery_sleeper_failed: "
                            f"{exc}"
                        ),
                        attempts=tuple(attempts),
                        exhausted=False,
                        result=(
                            last_result
                            if raw_result is None
                            else raw_result
                        ),
                    )

            current_request = replace(
                current_request,
                attempt=decision.next_attempt,
            )

            if execution_failed:
                last_result = None
