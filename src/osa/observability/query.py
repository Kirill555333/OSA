from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from osa.observability.events import ObservabilityEvent


class ObservabilityQueryError(ValueError):
    """Raised when an observability query is invalid."""


class ObservabilityEventSource(Protocol):
    """Minimal event source required by ObservabilityEventQuery."""

    def events(self) -> list[ObservabilityEvent]:
        """Return available observability events."""
        ...


@dataclass(frozen=True, slots=True)
class ObservabilityEventQuery:
    """
    Immutable query definition for observability events.

    Any non-None field participates in matching. Tuple-valued filters use
    membership semantics.
    """

    event: str | tuple[str, ...] | None = None
    source: str | tuple[str, ...] | None = None
    status: str | tuple[str, ...] | None = None
    request_id: str | tuple[str, ...] | None = None
    run_id: str | tuple[str, ...] | None = None
    task_id: str | tuple[str, ...] | None = None
    action_kind: str | tuple[str, ...] | None = None
    action_name: str | tuple[str, ...] | None = None
    decision: str | tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )
    limit: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "event",
            self._normalize_filter(self.event, "event"),
        )
        object.__setattr__(
            self,
            "source",
            self._normalize_filter(self.source, "source"),
        )
        object.__setattr__(
            self,
            "status",
            self._normalize_filter(self.status, "status"),
        )
        object.__setattr__(
            self,
            "request_id",
            self._normalize_filter(
                self.request_id,
                "request_id",
            ),
        )
        object.__setattr__(
            self,
            "run_id",
            self._normalize_filter(
                self.run_id,
                "run_id",
            ),
        )
        object.__setattr__(
            self,
            "task_id",
            self._normalize_filter(
                self.task_id,
                "task_id",
            ),
        )
        object.__setattr__(
            self,
            "action_kind",
            self._normalize_filter(
                self.action_kind,
                "action_kind",
            ),
        )
        object.__setattr__(
            self,
            "action_name",
            self._normalize_filter(
                self.action_name,
                "action_name",
            ),
        )
        object.__setattr__(
            self,
            "decision",
            self._normalize_filter(
                self.decision,
                "decision",
            ),
        )

        if not isinstance(self.metadata, Mapping):
            raise ObservabilityQueryError(
                "metadata must be a mapping"
            )

        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

        if self.limit is not None:
            if isinstance(self.limit, bool) or not isinstance(
                self.limit,
                int,
            ):
                raise ObservabilityQueryError(
                    "limit must be an integer or None"
                )

            if self.limit < 1:
                raise ObservabilityQueryError(
                    "limit must be at least 1"
                )

    @staticmethod
    def _normalize_filter(
        value: str | tuple[str, ...] | None,
        field_name: str,
    ) -> str | tuple[str, ...] | None:
        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip()

            if not normalized:
                raise ObservabilityQueryError(
                    f"{field_name} cannot be empty"
                )

            return normalized

        if not isinstance(value, tuple):
            raise ObservabilityQueryError(
                f"{field_name} must be a string, tuple, or None"
            )

        normalized_items: list[str] = []

        for item in value:
            if not isinstance(item, str):
                raise ObservabilityQueryError(
                    f"{field_name} tuple items must be strings"
                )

            normalized = item.strip()

            if not normalized:
                raise ObservabilityQueryError(
                    f"{field_name} tuple items cannot be empty"
                )

            normalized_items.append(normalized)

        if not normalized_items:
            raise ObservabilityQueryError(
                f"{field_name} tuple cannot be empty"
            )

        return tuple(normalized_items)

    @staticmethod
    def _matches_filter(
        value: str | None,
        query: str | tuple[str, ...] | None,
    ) -> bool:
        if query is None:
            return True

        if value is None:
            return False

        if isinstance(query, str):
            return value == query

        return value in query

    def matches(
        self,
        event: ObservabilityEvent,
    ) -> bool:
        if not isinstance(event, ObservabilityEvent):
            return False

        if not self._matches_filter(
            event.event,
            self.event,
        ):
            return False

        if not self._matches_filter(
            event.source,
            self.source,
        ):
            return False

        if not self._matches_filter(
            event.status,
            self.status,
        ):
            return False

        if not self._matches_filter(
            event.request_id,
            self.request_id,
        ):
            return False

        if not self._matches_filter(
            event.run_id,
            self.run_id,
        ):
            return False

        if not self._matches_filter(
            event.task_id,
            self.task_id,
        ):
            return False

        if not self._matches_filter(
            event.action_kind,
            self.action_kind,
        ):
            return False

        if not self._matches_filter(
            event.action_name,
            self.action_name,
        ):
            return False

        if not self._matches_filter(
            event.decision,
            self.decision,
        ):
            return False

        for key, expected in self.metadata.items():
            if event.metadata.get(key) != expected:
                return False

        return True


class ObservabilityEventQueryAPI:
    """
    Read-only query API over an observability event source.

    The API does not mutate or reorder the source. Returned results preserve
    the event order supplied by the source, except for latest(), which returns
    the most recent matching events based on source order.
    """

    def __init__(
        self,
        source: ObservabilityEventSource,
    ) -> None:
        if source is None:
            raise ObservabilityQueryError(
                "source is required"
            )

        self._source = source

    @property
    def source(self) -> ObservabilityEventSource:
        return self._source

    def query(
        self,
        query: ObservabilityEventQuery | None = None,
        *,
        event: str | tuple[str, ...] | None = None,
        source: str | tuple[str, ...] | None = None,
        status: str | tuple[str, ...] | None = None,
        request_id: str | tuple[str, ...] | None = None,
        run_id: str | tuple[str, ...] | None = None,
        task_id: str | tuple[str, ...] | None = None,
        action_kind: str | tuple[str, ...] | None = None,
        action_name: str | tuple[str, ...] | None = None,
        decision: str | tuple[str, ...] | None = None,
        metadata: Mapping[str, Any] | None = None,
        limit: int | None = None,
    ) -> tuple[ObservabilityEvent, ...]:
        if query is not None and any(
            value is not None
            for value in (
                event,
                source,
                status,
                request_id,
                run_id,
                task_id,
                action_kind,
                action_name,
                decision,
                metadata,
                limit,
            )
        ):
            raise ObservabilityQueryError(
                "query object cannot be combined with "
                "inline query filters"
            )

        if query is None:
            query = ObservabilityEventQuery(
                event=event,
                source=source,
                status=status,
                request_id=request_id,
                run_id=run_id,
                task_id=task_id,
                action_kind=action_kind,
                action_name=action_name,
                decision=decision,
                metadata=(
                    {}
                    if metadata is None
                    else metadata
                ),
                limit=limit,
            )

        raw_events = self._source.events()

        if not isinstance(raw_events, Iterable):
            raise ObservabilityQueryError(
                "event source returned a non-iterable collection"
            )

        matched: list[ObservabilityEvent] = []

        for item in raw_events:
            if not isinstance(item, ObservabilityEvent):
                continue

            if not query.matches(item):
                continue

            matched.append(item)

            if (
                query.limit is not None
                and len(matched) >= query.limit
            ):
                break

        return tuple(matched)

    def count(
        self,
        query: ObservabilityEventQuery | None = None,
        **filters: Any,
    ) -> int:
        return len(
            self.query(
                query,
                **filters,
            )
        )

    def latest(
        self,
        limit: int = 1,
        query: ObservabilityEventQuery | None = None,
        **filters: Any,
    ) -> tuple[ObservabilityEvent, ...]:
        if isinstance(limit, bool) or not isinstance(
            limit,
            int,
        ):
            raise ObservabilityQueryError(
                "limit must be an integer"
            )

        if limit < 1:
            raise ObservabilityQueryError(
                "limit must be at least 1"
            )

        if query is not None:
            if filters:
                raise ObservabilityQueryError(
                    "query object cannot be combined with "
                    "inline query filters"
                )

            matched = self.query(query)
        else:
            matched = self.query(**filters)

        return tuple(
            reversed(
                matched[-limit:]
            )
        )
