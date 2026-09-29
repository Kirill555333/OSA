from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from osa.observability.events import ObservabilityEvent
from osa.observability.logger import ObservabilityLogger


class ObservabilityRedactionError(ValueError):
    """Raised when observability redaction configuration is invalid."""


DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "token",
        "access_token",
        "refresh_token",
        "api_key",
        "apikey",
        "secret",
        "authorization",
        "cookie",
        "set-cookie",
        "session",
        "session_id",
        "credential",
        "credentials",
        "private_key",
        "client_secret",
    }
)


class ObservabilityRedactor:
    """
    Produce sanitized copies of observability data.

    The redactor never mutates caller-owned objects.
    """

    def __init__(
        self,
        *,
        sensitive_keys: set[str] | frozenset[str] | None = None,
        replacement: str = "[REDACTED]",
        max_string_length: int = 4096,
    ) -> None:
        if sensitive_keys is None:
            sensitive_keys = DEFAULT_SENSITIVE_KEYS

        if not isinstance(sensitive_keys, (set, frozenset)):
            raise ObservabilityRedactionError(
                "sensitive_keys must be a set or frozenset"
            )

        normalized_keys: set[str] = set()

        for key in sensitive_keys:
            if not isinstance(key, str):
                raise ObservabilityRedactionError(
                    "sensitive_keys must contain strings"
                )

            normalized = key.strip().lower()

            if not normalized:
                raise ObservabilityRedactionError(
                    "sensitive_keys cannot contain empty strings"
                )

            normalized_keys.add(normalized)

        if not isinstance(replacement, str):
            raise ObservabilityRedactionError(
                "replacement must be a string"
            )

        if not replacement:
            raise ObservabilityRedactionError(
                "replacement cannot be empty"
            )

        if (
            isinstance(max_string_length, bool)
            or not isinstance(max_string_length, int)
        ):
            raise ObservabilityRedactionError(
                "max_string_length must be an integer"
            )

        if max_string_length < 1:
            raise ObservabilityRedactionError(
                "max_string_length must be at least 1"
            )

        self._sensitive_keys = frozenset(
            normalized_keys
        )
        self._replacement = replacement
        self._max_string_length = max_string_length

    @property
    def sensitive_keys(self) -> frozenset[str]:
        return self._sensitive_keys

    @property
    def replacement(self) -> str:
        return self._replacement

    @property
    def max_string_length(self) -> int:
        return self._max_string_length

    def redact_event(
        self,
        event: ObservabilityEvent,
    ) -> ObservabilityEvent:
        if not isinstance(event, ObservabilityEvent):
            raise ObservabilityRedactionError(
                "event must be an ObservabilityEvent"
            )

        return ObservabilityEvent(
            event=event.event,
            source=event.source,
            request_id=event.request_id,
            run_id=event.run_id,
            task_id=event.task_id,
            action_kind=event.action_kind,
            action_name=event.action_name,
            status=event.status,
            duration_ms=event.duration_ms,
            decision=event.decision,
            attempt=event.attempt,
            max_attempts=event.max_attempts,
            error=self._redact_string(event.error),
            metadata=self.redact_mapping(
                event.metadata
            ),
            timestamp=event.timestamp,
            event_id=event.event_id,
        )

    def redact_mapping(
        self,
        value: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ObservabilityRedactionError(
                "value must be a mapping"
            )

        result: dict[str, Any] = {}

        for key, item in value.items():
            if isinstance(key, str):
                normalized_key = key.strip().lower()
            else:
                normalized_key = ""

            if normalized_key in self._sensitive_keys:
                result[key] = self._replacement
                continue

            result[key] = self.redact_value(item)

        return result

    def redact_value(
        self,
        value: Any,
    ) -> Any:
        if isinstance(value, str):
            return self._redact_string(value)

        if isinstance(value, Mapping):
            return self.redact_mapping(value)

        if isinstance(value, tuple):
            return tuple(
                self.redact_value(item)
                for item in value
            )

        if isinstance(value, list):
            return [
                self.redact_value(item)
                for item in value
            ]

        if isinstance(value, set):
            return {
                self.redact_value(item)
                for item in value
            }

        return value

    def _redact_string(
        self,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            return value

        if len(value) <= self._max_string_length:
            return value

        return (
            value[: self._max_string_length]
            + "…[TRUNCATED]"
        )


class RedactingObservabilityLogger:
    """
    Logger decorator that sanitizes events before forwarding them.

    Logger failures remain the responsibility of the wrapped logger.
    """

    def __init__(
        self,
        logger: ObservabilityLogger,
        *,
        redactor: ObservabilityRedactor | None = None,
    ) -> None:
        if logger is None:
            raise ObservabilityRedactionError(
                "logger is required"
            )

        self._logger = logger
        self._redactor = (
            ObservabilityRedactor()
            if redactor is None
            else redactor
        )

    @property
    def logger(self) -> ObservabilityLogger:
        return self._logger

    @property
    def redactor(self) -> ObservabilityRedactor:
        return self._redactor

    def log(
        self,
        event: ObservabilityEvent | str,
        **data: Any,
    ) -> None:
        if isinstance(event, ObservabilityEvent):
            self._logger.log(
                self._redactor.redact_event(event)
            )
            return

        if not isinstance(event, str):
            raise ObservabilityRedactionError(
                "event must be an ObservabilityEvent or string"
            )

        sanitized_data = self._redactor.redact_mapping(
            data
        )

        self._logger.log(
            event,
            **sanitized_data,
        )
