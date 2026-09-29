from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from osa.recovery_contracts import RecoveryRequest


class RecoveryContextError(ValueError):
    """Raised when recovery context data is invalid."""


@dataclass(frozen=True)
class RecoveryContext:
    """
    Immutable runtime context shared across recovery layers.

    The context contains correlation and action identity only. Raw action
    arguments and outputs are intentionally not part of this object.
    """

    run_id: str | None = None
    task_id: str | None = None
    request_id: str | None = None
    action_kind: Any = None
    action_name: str | None = None
    attempt: int = 1
    max_attempts: int = 1
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}),
        repr=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        normalized_run_id = self._normalize_identifier(
            self.run_id,
            "run_id",
        )
        normalized_task_id = self._normalize_identifier(
            self.task_id,
            "task_id",
        )
        normalized_request_id = self._normalize_identifier(
            self.request_id,
            "request_id",
        )

        if (
            normalized_run_id is None
            and normalized_task_id is None
            and normalized_request_id is None
        ):
            raise RecoveryContextError(
                "At least one recovery identifier is required."
            )

        if not isinstance(
            self.attempt,
            int,
        ):
            raise RecoveryContextError(
                "attempt must be an integer."
            )

        if not isinstance(
            self.max_attempts,
            int,
        ):
            raise RecoveryContextError(
                "max_attempts must be an integer."
            )

        if self.attempt < 1:
            raise RecoveryContextError(
                "attempt must be at least 1."
            )

        if self.max_attempts < 1:
            raise RecoveryContextError(
                "max_attempts must be at least 1."
            )

        if self.attempt > self.max_attempts:
            raise RecoveryContextError(
                "attempt cannot exceed max_attempts."
            )

        normalized_action_name = self._normalize_action_name(
            self.action_name
        )

        normalized_kind = self._enum_value(
            self.action_kind
        )

        if not isinstance(
            self.metadata,
            Mapping,
        ):
            raise RecoveryContextError(
                "metadata must be a mapping."
            )

        immutable_metadata = MappingProxyType(
            dict(self.metadata)
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
            "request_id",
            normalized_request_id,
        )
        object.__setattr__(
            self,
            "action_name",
            normalized_action_name,
        )
        object.__setattr__(
            self,
            "action_kind",
            normalized_kind,
        )
        object.__setattr__(
            self,
            "metadata",
            immutable_metadata,
        )

    @classmethod
    def from_request(
        cls,
        request: RecoveryRequest,
    ) -> RecoveryContext:
        """Create immutable recovery context from a recovery request."""
        if not isinstance(
            request,
            RecoveryRequest,
        ):
            raise TypeError(
                "request must be a RecoveryRequest."
            )

        return cls(
            run_id=request.run_id,
            task_id=request.task_id,
            request_id=request.request_id,
            action_kind=request.action_kind,
            action_name=request.action_name,
            attempt=request.attempt,
            max_attempts=request.max_attempts,
            metadata=request.metadata,
        )

    @property
    def is_autonomous(self) -> bool:
        """Return whether the context has autonomous run/task correlation."""
        return (
            self.run_id is not None
            or self.task_id is not None
        )

    @property
    def is_action(self) -> bool:
        """Return whether the context identifies an action request."""
        return (
            self.request_id is not None
            or self.action_kind is not None
            or self.action_name is not None
        )

    @property
    def exhausted(self) -> bool:
        """Return whether the current attempt is the final allowed attempt."""
        return self.attempt >= self.max_attempts

    @property
    def correlation_id(self) -> str | None:
        """
        Return the most specific available correlation identifier.

        Request ID takes precedence over task ID, then run ID.
        """
        if self.request_id is not None:
            return self.request_id

        if self.task_id is not None:
            return self.task_id

        return self.run_id

    def next_attempt(
        self,
        attempt: int | None = None,
    ) -> RecoveryContext:
        """
        Return a new context for the next recovery attempt.

        The current context is never mutated.
        """
        next_attempt = (
            self.attempt + 1
            if attempt is None
            else attempt
        )

        if next_attempt <= self.attempt:
            raise RecoveryContextError(
                "next attempt must be greater than "
                "the current attempt."
            )

        if next_attempt > self.max_attempts:
            raise RecoveryContextError(
                "next attempt cannot exceed max_attempts."
            )

        return RecoveryContext(
            run_id=self.run_id,
            task_id=self.task_id,
            request_id=self.request_id,
            action_kind=self.action_kind,
            action_name=self.action_name,
            attempt=next_attempt,
            max_attempts=self.max_attempts,
            metadata=self.metadata,
        )

    def with_metadata(
        self,
        **updates: Any,
    ) -> RecoveryContext:
        """
        Return a new context with merged metadata.

        The existing context and its metadata remain unchanged.
        """
        merged = dict(self.metadata)
        merged.update(updates)

        return RecoveryContext(
            run_id=self.run_id,
            task_id=self.task_id,
            request_id=self.request_id,
            action_kind=self.action_kind,
            action_name=self.action_name,
            attempt=self.attempt,
            max_attempts=self.max_attempts,
            metadata=merged,
        )

    def without_metadata(
        self,
        *keys: str,
    ) -> RecoveryContext:
        """Return a new context with selected metadata removed."""
        remaining = {
            key: value
            for key, value in self.metadata.items()
            if key not in keys
        }

        return RecoveryContext(
            run_id=self.run_id,
            task_id=self.task_id,
            request_id=self.request_id,
            action_kind=self.action_kind,
            action_name=self.action_name,
            attempt=self.attempt,
            max_attempts=self.max_attempts,
            metadata=remaining,
        )

    @staticmethod
    def _normalize_identifier(
        value: str | None,
        field_name: str,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            raise RecoveryContextError(
                f"{field_name} must be a string or None."
            )

        normalized = value.strip()

        if not normalized:
            raise RecoveryContextError(
                f"{field_name} cannot be empty."
            )

        return normalized

    @staticmethod
    def _normalize_action_name(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):
            raise RecoveryContextError(
                "action_name must be a string or None."
            )

        normalized = value.strip()

        if not normalized:
            raise RecoveryContextError(
                "action_name cannot be empty."
            )

        return normalized

    @staticmethod
    def _enum_value(
        value: Any,
    ) -> Any:
        return getattr(
            value,
            "value",
            value,
        )


EMPTY_RECOVERY_CONTEXT = RecoveryContext(
    request_id="empty-context",
)
