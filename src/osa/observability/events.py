from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from osa.actions.contracts import ActionKind


class ObservabilityEventError(ValueError):
    """Raised when an observability event is invalid."""


def _normalize_optional_identifier(
    value: str | None,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ObservabilityEventError(
            f"{field_name} must be a string or None."
        )

    normalized = value.strip()

    if not normalized:
        raise ObservabilityEventError(
            f"{field_name} cannot be empty."
        )

    return normalized


def _normalize_optional_string(
    value: str | None,
    field_name: str,
) -> str | None:
    return _normalize_optional_identifier(
        value,
        field_name,
    )


def _normalize_action_kind(
    value: ActionKind | str | None,
) -> str | None:
    if value is None:
        return None

    if isinstance(value, ActionKind):
        return value.value

    if not isinstance(value, str):
        raise ObservabilityEventError(
            "action_kind must be an ActionKind, string, or None."
        )

    normalized = value.strip()

    if not normalized:
        raise ObservabilityEventError(
            "action_kind cannot be empty."
        )

    return normalized


def _normalize_timestamp(
    value: datetime | None,
) -> datetime:
    if value is None:
        return datetime.now(
            timezone.utc
        )

    if not isinstance(value, datetime):
        raise ObservabilityEventError(
            "timestamp must be a datetime or None."
        )

    if value.tzinfo is None:
        raise ObservabilityEventError(
            "timestamp must be timezone-aware."
        )

    return value


def _normalize_non_negative_number(
    value: int | float | None,
    field_name: str,
) -> int | float | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ObservabilityEventError(
            f"{field_name} must be a number or None."
        )

    if value < 0:
        raise ObservabilityEventError(
            f"{field_name} cannot be negative."
        )

    return value


def _normalize_positive_integer(
    value: int | None,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise ObservabilityEventError(
            f"{field_name} must be an integer or None."
        )

    if value < 1:
        raise ObservabilityEventError(
            f"{field_name} must be at least 1."
        )

    return value


@dataclass(frozen=True)
class ObservabilityEvent:
    """
    Immutable structured event shared by OSA execution layers.

    Correlation identifiers are optional because not every event belongs to
    an autonomous task or action request. When present, they preserve the
    relationship between run, task, and action execution.
    """

    event: str
    source: str

    event_id: str = field(
        default_factory=lambda: str(uuid4())
    )
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    request_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None

    action_kind: ActionKind | str | None = None
    action_name: str | None = None

    status: str | None = None
    duration_ms: int | float | None = None
    decision: str | None = None

    attempt: int | None = None
    max_attempts: int | None = None

    error: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.event_id,
            str,
        ):
            raise ObservabilityEventError(
                "event_id must be a string."
            )

        normalized_event_id = self.event_id.strip()

        if not normalized_event_id:
            raise ObservabilityEventError(
                "event_id cannot be empty."
            )

        if not isinstance(
            self.event,
            str,
        ):
            raise ObservabilityEventError(
                "event must be a string."
            )

        normalized_event = self.event.strip()

        if not normalized_event:
            raise ObservabilityEventError(
                "event cannot be empty."
            )

        if not isinstance(
            self.source,
            str,
        ):
            raise ObservabilityEventError(
                "source must be a string."
            )

        normalized_source = self.source.strip()

        if not normalized_source:
            raise ObservabilityEventError(
                "source cannot be empty."
            )

        normalized_request_id = (
            _normalize_optional_identifier(
                self.request_id,
                "request_id",
            )
        )

        normalized_run_id = (
            _normalize_optional_identifier(
                self.run_id,
                "run_id",
            )
        )

        normalized_task_id = (
            _normalize_optional_identifier(
                self.task_id,
                "task_id",
            )
        )

        normalized_action_kind = _normalize_action_kind(
            self.action_kind
        )

        normalized_action_name = (
            _normalize_optional_string(
                self.action_name,
                "action_name",
            )
        )

        normalized_status = _normalize_optional_string(
            self.status,
            "status",
        )

        normalized_decision = _normalize_optional_string(
            self.decision,
            "decision",
        )

        normalized_error = _normalize_optional_string(
            self.error,
            "error",
        )

        normalized_duration_ms = (
            _normalize_non_negative_number(
                self.duration_ms,
                "duration_ms",
            )
        )

        normalized_attempt = (
            _normalize_positive_integer(
                self.attempt,
                "attempt",
            )
        )

        normalized_max_attempts = (
            _normalize_positive_integer(
                self.max_attempts,
                "max_attempts",
            )
        )

        if (
            normalized_attempt is not None
            and normalized_max_attempts is not None
            and normalized_attempt > normalized_max_attempts
        ):
            raise ObservabilityEventError(
                "attempt cannot exceed max_attempts."
            )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise ObservabilityEventError(
                "metadata must be a mapping."
            )

        metadata = dict(
            self.metadata
        )

        object.__setattr__(
            self,
            "event_id",
            normalized_event_id,
        )
        object.__setattr__(
            self,
            "event",
            normalized_event,
        )
        object.__setattr__(
            self,
            "source",
            normalized_source,
        )
        object.__setattr__(
            self,
            "timestamp",
            _normalize_timestamp(
                self.timestamp
            ),
        )
        object.__setattr__(
            self,
            "request_id",
            normalized_request_id,
        )
        object.__setattr__(
            self,
            "run_id",
            normalized_run_id,
        )
        object.__setattr__(
            self,
            "task_id",
            normalized_task_id,
        )
        object.__setattr__(
            self,
            "action_kind",
            normalized_action_kind,
        )
        object.__setattr__(
            self,
            "action_name",
            normalized_action_name,
        )
        object.__setattr__(
            self,
            "status",
            normalized_status,
        )
        object.__setattr__(
            self,
            "duration_ms",
            normalized_duration_ms,
        )
        object.__setattr__(
            self,
            "decision",
            normalized_decision,
        )
        object.__setattr__(
            self,
            "attempt",
            normalized_attempt,
        )
        object.__setattr__(
            self,
            "max_attempts",
            normalized_max_attempts,
        )
        object.__setattr__(
            self,
            "error",
            normalized_error,
        )
        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(
                metadata
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly event representation."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event": self.event,
            "source": self.source,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "action_kind": self.action_kind,
            "action_name": self.action_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "decision": self.decision,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "metadata": dict(
                self.metadata
            ),
        }
