from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Any, Protocol

from osa.observability.events import ObservabilityEvent


class ObservabilityLoggerError(RuntimeError):
    """Raised when an observability logger cannot process an event."""


class ObservabilityLogger(Protocol):
    """Minimal sink interface for structured OSA observability events."""

    def log(
        self,
        event: ObservabilityEvent | str,
        **data: Any,
    ) -> ObservabilityEvent | None:
        """Record an event without affecting execution semantics."""
        ...


class NullObservabilityLogger:
    """
    No-op observability sink.

    It intentionally returns None and never stores any event.
    """

    def log(
        self,
        event: ObservabilityEvent | str,
        **data: Any,
    ) -> None:
        return None


class InMemoryObservabilityLogger:
    """
    Thread-safe in-memory event sink for tests, diagnostics and local UI work.
    """

    _EVENT_FIELDS = {
        "event_id",
        "timestamp",
        "request_id",
        "run_id",
        "task_id",
        "action_kind",
        "action_name",
        "status",
        "duration_ms",
        "decision",
        "attempt",
        "max_attempts",
        "error",
        "metadata",
    }

    def __init__(self) -> None:
        self._events: list[ObservabilityEvent] = []
        self._lock = RLock()

    def log(
        self,
        event: ObservabilityEvent | str,
        **data: Any,
    ) -> ObservabilityEvent:
        normalized = self._coerce_event(
            event,
            data,
        )

        with self._lock:
            self._events.append(normalized)

        return normalized

    def events(self) -> tuple[ObservabilityEvent, ...]:
        """Return an immutable snapshot of all recorded events."""
        with self._lock:
            return tuple(self._events)

    def clear(self) -> None:
        """Remove all recorded events."""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    @classmethod
    def _coerce_event(
        cls,
        event: ObservabilityEvent | str,
        data: Mapping[str, Any],
    ) -> ObservabilityEvent:
        if isinstance(
            event,
            ObservabilityEvent,
        ):
            if data:
                raise ObservabilityLoggerError(
                    "Additional data is not allowed when logging "
                    "an ObservabilityEvent instance."
                )

            return event

        if not isinstance(
            event,
            str,
        ):
            raise ObservabilityLoggerError(
                "event must be an ObservabilityEvent or string."
            )

        normalized_event = event.strip()

        if not normalized_event:
            raise ObservabilityLoggerError(
                "event cannot be empty."
            )

        payload = dict(data)

        source = payload.pop(
            "source",
            "legacy",
        )

        metadata_value = payload.pop(
            "metadata",
            None,
        )

        if metadata_value is None:
            metadata: dict[str, Any] = {}
        elif isinstance(
            metadata_value,
            Mapping,
        ):
            metadata = dict(metadata_value)
        else:
            raise ObservabilityLoggerError(
                "metadata must be a mapping when supplied."
            )

        event_kwargs: dict[str, Any] = {
            "event": normalized_event,
            "source": source,
        }

        for field_name in cls._EVENT_FIELDS:
            if field_name == "metadata":
                continue

            if field_name in payload:
                event_kwargs[field_name] = payload.pop(
                    field_name
                )

        metadata.update(payload)

        if metadata:
            event_kwargs["metadata"] = metadata

        return ObservabilityEvent(
            **event_kwargs,
        )


__all__ = [
    "InMemoryObservabilityLogger",
    "NullObservabilityLogger",
    "ObservabilityLogger",
    "ObservabilityLoggerError",
]
