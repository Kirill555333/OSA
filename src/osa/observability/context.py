from __future__ import annotations

from dataclasses import dataclass, replace


class ObservabilityContextError(ValueError):
    """Raised when an observability correlation context is invalid."""


def _normalize_identifier(
    value: str | None,
    field_name: str,
) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        raise ObservabilityContextError(
            f"{field_name} must be a string or None."
        )

    normalized = value.strip()

    if not normalized:
        raise ObservabilityContextError(
            f"{field_name} cannot be empty."
        )

    return normalized


@dataclass(frozen=True)
class ObservabilityContext:
    """
    Immutable correlation context shared across OSA execution layers.

    Any subset of run_id, task_id and request_id is valid. This allows the
    same context object to be used for autonomous, task, action and ordinary
    agent events.
    """

    run_id: str | None = None
    task_id: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "run_id",
            _normalize_identifier(
                self.run_id,
                "run_id",
            ),
        )
        object.__setattr__(
            self,
            "task_id",
            _normalize_identifier(
                self.task_id,
                "task_id",
            ),
        )
        object.__setattr__(
            self,
            "request_id",
            _normalize_identifier(
                self.request_id,
                "request_id",
            ),
        )

    @property
    def empty(self) -> bool:
        """Return True when no correlation identifiers are present."""
        return (
            self.run_id is None
            and self.task_id is None
            and self.request_id is None
        )

    @property
    def autonomous(self) -> bool:
        """Return True when this context belongs to an autonomous run."""
        return self.run_id is not None

    @property
    def action(self) -> bool:
        """Return True when this context identifies an action request."""
        return self.request_id is not None

    def with_run(
        self,
        run_id: str,
    ) -> ObservabilityContext:
        """Return a new context with the supplied run identifier."""
        return replace(
            self,
            run_id=run_id,
        )

    def with_task(
        self,
        task_id: str,
    ) -> ObservabilityContext:
        """Return a new context with the supplied task identifier."""
        return replace(
            self,
            task_id=task_id,
        )

    def with_request(
        self,
        request_id: str,
    ) -> ObservabilityContext:
        """Return a new context with the supplied request identifier."""
        return replace(
            self,
            request_id=request_id,
        )

    def clear_request(self) -> ObservabilityContext:
        """Return a new context without an action request identifier."""
        return replace(
            self,
            request_id=None,
        )

    def clear_task(self) -> ObservabilityContext:
        """Return a new context without a task identifier."""
        return replace(
            self,
            task_id=None,
            request_id=None,
        )

    def clear_run(self) -> ObservabilityContext:
        """Return a new empty correlation context."""
        return replace(
            self,
            run_id=None,
            task_id=None,
            request_id=None,
        )

    def to_dict(self) -> dict[str, str | None]:
        """Return the context in event-compatible dictionary form."""
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "request_id": self.request_id,
        }


EMPTY_OBSERVABILITY_CONTEXT = ObservabilityContext()


__all__ = [
    "EMPTY_OBSERVABILITY_CONTEXT",
    "ObservabilityContext",
    "ObservabilityContextError",
]
