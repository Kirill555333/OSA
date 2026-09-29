from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class RecoveryContractError(ValueError):
    """Raised when a unified recovery contract is invalid."""


class RecoveryFailureKind(str, Enum):
    """Canonical categories used by the unified recovery layer."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"
    PERMISSION_DENIED = "permission_denied"
    CONFIRMATION_DENIED = "confirmation_denied"
    BACKEND_ERROR = "backend_error"
    INVALID_RESULT = "invalid_result"
    VERIFICATION_FAILED = "verification_failed"
    MAX_ATTEMPTS = "max_attempts"


def _normalize_identifier(
    value: str | None,
    field_name: str,
    *,
    required: bool,
) -> str | None:
    if value is None:
        if required:
            raise RecoveryContractError(
                f"{field_name} is required"
            )
        return None

    if not isinstance(value, str):
        raise RecoveryContractError(
            f"{field_name} must be a string"
        )

    normalized = value.strip()

    if not normalized:
        if required:
            raise RecoveryContractError(
                f"{field_name} cannot be empty"
            )
        return None

    return normalized


def _normalize_metadata(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise RecoveryContractError(
            "metadata must be a mapping"
        )

    return MappingProxyType(dict(value))


def _normalize_positive_int(
    value: int,
    field_name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryContractError(
            f"{field_name} must be an integer"
        )

    if value < 1:
        raise RecoveryContractError(
            f"{field_name} must be at least 1"
        )

    return value


def _normalize_non_negative_float(
    value: float,
    field_name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise RecoveryContractError(
            f"{field_name} must be a number"
        )

    normalized = float(value)

    if normalized < 0.0:
        raise RecoveryContractError(
            f"{field_name} must be non-negative"
        )

    return normalized


@dataclass(frozen=True, slots=True)
class RecoveryRequest:
    """
    Immutable request describing one recovery operation.

    The identifier set supports both action recovery and autonomous task
    recovery:
    - action flow normally supplies request_id;
    - autonomous flow normally supplies run_id + task_id.
    """

    run_id: str | None = None
    task_id: str | None = None
    request_id: str | None = None
    action_kind: str | None = None
    action_name: str | None = None
    attempt: int = 1
    max_attempts: int = 1
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        run_id = _normalize_identifier(
            self.run_id,
            "run_id",
            required=False,
        )
        task_id = _normalize_identifier(
            self.task_id,
            "task_id",
            required=False,
        )
        request_id = _normalize_identifier(
            self.request_id,
            "request_id",
            required=False,
        )
        action_kind = _normalize_identifier(
            self.action_kind,
            "action_kind",
            required=False,
        )
        action_name = _normalize_identifier(
            self.action_name,
            "action_name",
            required=False,
        )

        if (
            run_id is None
            and task_id is None
            and request_id is None
        ):
            raise RecoveryContractError(
                "at least one recovery identifier is required"
            )

        attempt = _normalize_positive_int(
            self.attempt,
            "attempt",
        )
        max_attempts = _normalize_positive_int(
            self.max_attempts,
            "max_attempts",
        )

        if attempt > max_attempts:
            raise RecoveryContractError(
                "attempt cannot exceed max_attempts"
            )

        object.__setattr__(
            self,
            "run_id",
            run_id,
        )
        object.__setattr__(
            self,
            "task_id",
            task_id,
        )
        object.__setattr__(
            self,
            "request_id",
            request_id,
        )
        object.__setattr__(
            self,
            "action_kind",
            action_kind,
        )
        object.__setattr__(
            self,
            "action_name",
            action_name,
        )
        object.__setattr__(
            self,
            "attempt",
            attempt,
        )
        object.__setattr__(
            self,
            "max_attempts",
            max_attempts,
        )
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata),
        )

    @property
    def autonomous(self) -> bool:
        return (
            self.run_id is not None
            or self.task_id is not None
        )

    @property
    def action(self) -> bool:
        return (
            self.request_id is not None
            or self.action_name is not None
        )


@dataclass(frozen=True, slots=True)
class RecoveryAttempt:
    """
    Immutable description of one completed recovery/execution attempt.
    """

    attempt: int
    max_attempts: int
    success: bool
    failure_kind: RecoveryFailureKind | None = None
    error: str | None = None
    output_present: bool = False
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        attempt = _normalize_positive_int(
            self.attempt,
            "attempt",
        )
        max_attempts = _normalize_positive_int(
            self.max_attempts,
            "max_attempts",
        )

        if attempt > max_attempts:
            raise RecoveryContractError(
                "attempt cannot exceed max_attempts"
            )

        if not isinstance(self.success, bool):
            raise RecoveryContractError(
                "success must be a boolean"
            )

        if self.failure_kind is not None and not isinstance(
            self.failure_kind,
            RecoveryFailureKind,
        ):
            try:
                failure_kind = RecoveryFailureKind(
                    self.failure_kind
                )
            except (TypeError, ValueError) as exc:
                raise RecoveryContractError(
                    "failure_kind is invalid"
                ) from exc
        else:
            failure_kind = self.failure_kind

        if self.error is not None:
            if not isinstance(self.error, str):
                raise RecoveryContractError(
                    "error must be a string or None"
                )

            error = self.error.strip() or None
        else:
            error = None

        if not isinstance(
            self.output_present,
            bool,
        ):
            raise RecoveryContractError(
                "output_present must be a boolean"
            )

        if self.success:
            if failure_kind is not None:
                raise RecoveryContractError(
                    "successful attempt cannot have failure_kind"
                )
        elif failure_kind is None:
            raise RecoveryContractError(
                "failed attempt requires failure_kind"
            )

        object.__setattr__(
            self,
            "attempt",
            attempt,
        )
        object.__setattr__(
            self,
            "max_attempts",
            max_attempts,
        )
        object.__setattr__(
            self,
            "failure_kind",
            failure_kind,
        )
        object.__setattr__(
            self,
            "error",
            error,
        )
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """
    Immutable policy decision for the next recovery step.
    """

    retry: bool
    reason: str | None = None
    failure_kind: RecoveryFailureKind | None = None
    delay_seconds: float = 0.0
    next_attempt: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.retry, bool):
            raise RecoveryContractError(
                "retry must be a boolean"
            )

        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise RecoveryContractError(
                    "reason must be a string or None"
                )

            reason = self.reason.strip() or None
        else:
            reason = None

        if self.failure_kind is not None and not isinstance(
            self.failure_kind,
            RecoveryFailureKind,
        ):
            try:
                failure_kind = RecoveryFailureKind(
                    self.failure_kind
                )
            except (TypeError, ValueError) as exc:
                raise RecoveryContractError(
                    "failure_kind is invalid"
                ) from exc
        else:
            failure_kind = self.failure_kind

        delay_seconds = _normalize_non_negative_float(
            self.delay_seconds,
            "delay_seconds",
        )

        next_attempt = self.next_attempt

        if next_attempt is not None:
            next_attempt = _normalize_positive_int(
                next_attempt,
                "next_attempt",
            )

        if self.retry and next_attempt is None:
            raise RecoveryContractError(
                "retry decision requires next_attempt"
            )

        if not self.retry and next_attempt is not None:
            raise RecoveryContractError(
                "non-retry decision cannot define next_attempt"
            )

        object.__setattr__(
            self,
            "reason",
            reason,
        )
        object.__setattr__(
            self,
            "failure_kind",
            failure_kind,
        )
        object.__setattr__(
            self,
            "delay_seconds",
            delay_seconds,
        )
        object.__setattr__(
            self,
            "next_attempt",
            next_attempt,
        )


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """
    Immutable aggregate result of a bounded recovery operation.

    `result` intentionally remains opaque so the contract can carry both
    ActionResult and AutonomousTaskResult without duplicating those models.
    """

    success: bool
    attempts: tuple[RecoveryAttempt, ...] = ()
    failure_kind: RecoveryFailureKind | None = None
    error: str | None = None
    exhausted: bool = False
    result: Any | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.success,
            bool,
        ):
            raise RecoveryContractError(
                "success must be a boolean"
            )

        if not isinstance(self.attempts, tuple):
            raise RecoveryContractError(
                "attempts must be a tuple"
            )

        normalized_attempts: list[RecoveryAttempt] = []

        for item in self.attempts:
            if not isinstance(
                item,
                RecoveryAttempt,
            ):
                raise RecoveryContractError(
                    "attempts must contain RecoveryAttempt values"
                )

            normalized_attempts.append(item)

        for index, item in enumerate(
            normalized_attempts,
            start=1,
        ):
            if item.attempt != index:
                raise RecoveryContractError(
                    "attempts must be ordered consecutively"
                )

        failure_kind = self.failure_kind

        if failure_kind is not None and not isinstance(
            failure_kind,
            RecoveryFailureKind,
        ):
            try:
                failure_kind = RecoveryFailureKind(
                    failure_kind
                )
            except (TypeError, ValueError) as exc:
                raise RecoveryContractError(
                    "failure_kind is invalid"
                ) from exc

        if self.error is not None:
            if not isinstance(self.error, str):
                raise RecoveryContractError(
                    "error must be a string or None"
                )

            error = self.error.strip() or None
        else:
            error = None

        if not isinstance(
            self.exhausted,
            bool,
        ):
            raise RecoveryContractError(
                "exhausted must be a boolean"
            )

        if self.success:
            if failure_kind is not None:
                raise RecoveryContractError(
                    "successful result cannot have failure_kind"
                )

            if error is not None:
                raise RecoveryContractError(
                    "successful result cannot have error"
                )

            if self.exhausted:
                raise RecoveryContractError(
                    "successful result cannot be exhausted"
                )
        else:
            if failure_kind is None:
                raise RecoveryContractError(
                    "failed result requires failure_kind"
                )

        object.__setattr__(
            self,
            "attempts",
            tuple(normalized_attempts),
        )
        object.__setattr__(
            self,
            "failure_kind",
            failure_kind,
        )
        object.__setattr__(
            self,
            "error",
            error,
        )
        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(self.metadata),
        )

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def last_attempt(self) -> RecoveryAttempt | None:
        if not self.attempts:
            return None

        return self.attempts[-1]

    @classmethod
    def succeeded(
        cls,
        *,
        result: Any | None = None,
        attempts: tuple[RecoveryAttempt, ...] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RecoveryResult:
        return cls(
            success=True,
            attempts=attempts,
            result=result,
            metadata=(
                {}
                if metadata is None
                else metadata
            ),
        )

    @classmethod
    def failed(
        cls,
        *,
        failure_kind: RecoveryFailureKind,
        error: str | None = None,
        attempts: tuple[RecoveryAttempt, ...] = (),
        exhausted: bool = False,
        result: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RecoveryResult:
        return cls(
            success=False,
            attempts=attempts,
            failure_kind=failure_kind,
            error=error,
            exhausted=exhausted,
            result=result,
            metadata=(
                {}
                if metadata is None
                else metadata
            ),
        )
