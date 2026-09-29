"""Canonical execution context used by OSA internal action infrastructure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from osa.actions.contracts import ActionRequest


class ExecutionContextError(ValueError):
    """Raised when an execution context is invalid."""


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
        raise ExecutionContextError(
            f"{field_name} must be a string or None."
        )

    normalized = value.strip()

    if not normalized:
        raise ExecutionContextError(
            f"{field_name} cannot be empty."
        )

    return normalized


def _normalize_round(
    value: int | None,
) -> int | None:
    if value is None:
        return None

    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        int,
    ):
        raise ExecutionContextError(
            "round_number must be an integer or None."
        )

    if value < 1:
        raise ExecutionContextError(
            "round_number must be at least 1."
        )

    return value


def _normalize_metadata(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})

    if not isinstance(
        value,
        Mapping,
    ):
        raise ExecutionContextError(
            "metadata must be a mapping."
        )

    return MappingProxyType(
        dict(value)
    )


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """
    Immutable correlation context surrounding one execution.

    `request_id` identifies the unified action.
    `run_id` and `task_id` identify a broader execution scope.
    `round_number` is present only for genuine model/tool rounds.
    `source` identifies the execution entry path.
    `metadata` contains only residual, non-correlation metadata.
    """

    request_id: str
    run_id: str | None = None
    task_id: str | None = None
    round_number: int | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate and normalize the immutable context."""
        request_id = _normalize_identifier(
            self.request_id,
            "request_id",
        )

        if request_id is None:
            raise ExecutionContextError(
                "request_id is required."
            )

        object.__setattr__(
            self,
            "request_id",
            request_id,
        )

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
            "round_number",
            _normalize_round(
                self.round_number
            ),
        )

        object.__setattr__(
            self,
            "source",
            _normalize_identifier(
                self.source,
                "source",
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _normalize_metadata(
                self.metadata
            ),
        )

    @property
    def autonomous(self) -> bool:
        """Return whether broader run/task execution context is present."""
        return (
            self.run_id is not None
            or self.task_id is not None
        )

    @property
    def round(self) -> int | None:
        """Return the model round when one exists."""
        return self.round_number

    @classmethod
    def from_action_request(
        cls,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        source: str | None = None,
    ) -> "ExecutionContext":
        """
        Extract one canonical context from an ActionRequest.

        Explicit run/task/source values supplied by the caller take
        precedence over corresponding request metadata. Request identity
        always comes from ActionRequest.

        Recognized correlation fields are consumed into canonical fields and
        are not duplicated in residual metadata.
        """
        from osa.actions.contracts import ActionRequest

        if not isinstance(
            request,
            ActionRequest,
        ):
            raise ExecutionContextError(
                "request must be an ActionRequest."
            )

        request_metadata = dict(
            request.metadata
        )

        request_metadata.pop(
            "request_id",
            None,
        )

        metadata_run_id = request_metadata.pop(
            "run_id",
            None,
        )

        metadata_task_id = request_metadata.pop(
            "task_id",
            None,
        )

        voice_task_id = request_metadata.pop(
            "voice_task_id",
            None,
        )

        metadata_round = request_metadata.pop(
            "round",
            None,
        )

        metadata_source = request_metadata.pop(
            "source",
            None,
        )

        resolved_run_id = (
            run_id
            if run_id is not None
            else metadata_run_id
        )

        resolved_task_id = (
            task_id
            if task_id is not None
            else (
                metadata_task_id
                if metadata_task_id is not None
                else voice_task_id
            )
        )

        resolved_source = (
            source
            if source is not None
            else metadata_source
        )

        return cls(
            request_id=request.request_id,
            run_id=resolved_run_id,
            task_id=resolved_task_id,
            round_number=metadata_round,
            source=resolved_source,
            metadata=request_metadata,
        )

    @classmethod
    def from_autonomous_action_request(
        cls,
        request: ActionRequest,
        *,
        run_id: str,
        task_id: str,
    ) -> "ExecutionContext":
        """
        Build the canonical context for one autonomous action.

        Autonomous run/task identifiers are authoritative inputs from the
        autonomous orchestration layer. The ActionRequest itself remains
        unchanged.
        """
        normalized_run_id = _normalize_identifier(
            run_id,
            "run_id",
        )

        if normalized_run_id is None:
            raise ExecutionContextError(
                "run_id is required for autonomous context."
            )

        normalized_task_id = _normalize_identifier(
            task_id,
            "task_id",
        )

        if normalized_task_id is None:
            raise ExecutionContextError(
                "task_id is required for autonomous context."
            )

        context = cls.from_action_request(
            request,
            run_id=normalized_run_id,
            task_id=normalized_task_id,
            source="autonomous",
        )

        if context.round_number is not None:
            raise ExecutionContextError(
                "autonomous context cannot contain round_number."
            )

        return context


__all__ = [
    "ExecutionContext",
    "ExecutionContextError",
]
