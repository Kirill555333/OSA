from __future__ import annotations

from typing import Any, Protocol

from osa.actions.contracts import ActionRequest
from osa.recovery_action import (
    ActionResultClassifier,
    RecoverableActionExecutor,
)
from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryFailureKind,
    RecoveryRequest,
)
from osa.recovery_policy import RecoveryPolicy


class RecoveryVerificationError(RuntimeError):
    """Raised when verification integration is invalid."""


class ActionVerificationResult:
    """
    Immutable-like verification result used by recovery integration.

    ``passed`` is the authoritative signal that the requested action
    achieved its expected post-condition.
    """

    __slots__ = (
        "passed",
        "reason",
        "metadata",
    )

    def __init__(
        self,
        *,
        passed: bool,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(
            passed,
            bool,
        ):
            raise RecoveryVerificationError(
                "passed must be a boolean"
            )

        if reason is not None and not isinstance(
            reason,
            str,
        ):
            raise RecoveryVerificationError(
                "reason must be a string or None"
            )

        if metadata is None:
            metadata = {}

        if not isinstance(
            metadata,
            dict,
        ):
            raise RecoveryVerificationError(
                "metadata must be a dictionary"
            )

        self.passed = passed
        self.reason = reason
        self.metadata = dict(metadata)

    def __repr__(self) -> str:
        return (
            "ActionVerificationResult("
            f"passed={self.passed!r}, "
            f"reason={self.reason!r}, "
            f"metadata={self.metadata!r}"
            ")"
        )


class ActionVerifier(Protocol):
    """Verify that an action achieved its expected post-condition."""

    def __call__(
        self,
        request: ActionRequest,
        result: Any,
    ) -> ActionVerificationResult:
        ...


class VerificationAwareActionClassifier:
    """
    Decorator for an existing action-result classifier.

    Backend success is checked first. When the backend reports success,
    the verifier decides whether the actual action outcome is successful.

    Verification failure is never treated as backend success.
    """

    def __init__(
        self,
        verifier: ActionVerifier,
        *,
        base_classifier: ActionResultClassifier | None = None,
    ) -> None:
        if not callable(verifier):
            raise RecoveryVerificationError(
                "verifier must be callable"
            )

        self._verifier = verifier
        self._base_classifier = base_classifier

    @property
    def verifier(self) -> ActionVerifier:
        return self._verifier

    @property
    def base_classifier(
        self,
    ) -> ActionResultClassifier | None:
        return self._base_classifier

    def __call__(
        self,
        recovery_request: RecoveryRequest,
        result: Any,
    ) -> RecoveryAttempt:
        request = recovery_request.metadata.get(
            "_action_request"
        )

        if not isinstance(
            request,
            ActionRequest,
        ):
            raise RecoveryVerificationError(
                "verification context does not contain "
                "the original ActionRequest"
            )

        base_attempt = self._base_attempt(
            recovery_request,
            result,
        )

        if not base_attempt.success:
            return base_attempt

        try:
            verification = self._verifier(
                request,
                result,
            )
        except Exception as exc:
            raise RecoveryVerificationError(
                "action verification failed: "
                f"{exc}"
            ) from exc

        if not isinstance(
            verification,
            ActionVerificationResult,
        ):
            raise RecoveryVerificationError(
                "verifier must return "
                "ActionVerificationResult"
            )

        if verification.passed:
            return RecoveryAttempt(
                attempt=base_attempt.attempt,
                max_attempts=base_attempt.max_attempts,
                success=True,
                output_present=base_attempt.output_present,
                metadata={
                    **base_attempt.metadata,
                    "verification_passed": True,
                    **verification.metadata,
                },
            )

        return RecoveryAttempt(
            attempt=base_attempt.attempt,
            max_attempts=base_attempt.max_attempts,
            success=False,
            failure_kind=(
                RecoveryFailureKind.VERIFICATION_FAILED
            ),
            error=(
                verification.reason
                or "action_verification_failed"
            ),
            output_present=base_attempt.output_present,
            metadata={
                **base_attempt.metadata,
                "verification_passed": False,
                **verification.metadata,
            },
        )

    def _base_attempt(
        self,
        recovery_request: RecoveryRequest,
        result: Any,
    ) -> RecoveryAttempt:
        if self._base_classifier is None:
            base_attempt = RecoveryAttempt(
                attempt=recovery_request.attempt,
                max_attempts=recovery_request.max_attempts,
                success=(
                    getattr(
                        result,
                        "success",
                        None,
                    )
                    is True
                ),
                failure_kind=(
                    None
                    if getattr(
                        result,
                        "success",
                        None,
                    )
                    is True
                    else RecoveryFailureKind.BACKEND_ERROR
                ),
                error=getattr(
                    result,
                    "error",
                    None,
                ),
                output_present=(
                    getattr(
                        result,
                        "output",
                        None,
                    )
                    is not None
                ),
            )
        else:
            base_attempt = self._base_classifier(
                recovery_request,
                result,
            )

        if not isinstance(
            base_attempt,
            RecoveryAttempt,
        ):
            raise RecoveryVerificationError(
                "base classifier must return "
                "RecoveryAttempt"
            )

        return base_attempt


def create_verification_aware_executor(
    action_executor,
    *,
    verifier: ActionVerifier,
    max_attempts: int = 1,
    policy: RecoveryPolicy | None = None,
    base_classifier: ActionResultClassifier | None = None,
    sleeper=None,
) -> RecoverableActionExecutor:
    """
    Create a recoverable action executor with post-condition verification.

    The original action executor remains untouched. Verification only
    changes how successful backend results are classified for recovery.
    """

    classifier = VerificationAwareActionClassifier(
        verifier,
        base_classifier=base_classifier,
    )

    return RecoverableActionExecutor(
        action_executor,
        max_attempts=max_attempts,
        policy=policy,
        classifier=classifier,
        sleeper=sleeper,
    )
