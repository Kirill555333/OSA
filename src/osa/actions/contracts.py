"""Unified action contracts for OSA."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import uuid4


class ActionContractError(ValueError):
    """Raised when an action contract is invalid."""


class ActionKind(str, Enum):
    """Supported execution backends."""

    BROWSER = "browser"
    DESKTOP = "desktop"
    TOOL = "tool"


def _validate_non_empty(value: str, field_name: str) -> str:
    """Validate and normalize a required string field."""
    if not isinstance(value, str):
        raise ActionContractError(
            f"{field_name} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ActionContractError(
            f"{field_name} cannot be empty."
        )

    return normalized


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate and freeze a string-keyed mapping."""
    if value is None:
        return MappingProxyType({})

    if not isinstance(value, Mapping):
        raise ActionContractError(
            f"{field_name} must be a mapping."
        )

    normalized: dict[str, Any] = {}

    for key, item in value.items():
        if not isinstance(key, str):
            raise ActionContractError(
                f"{field_name} keys must be strings."
            )

        normalized[key] = item

    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """A backend-neutral request for one executable action."""

    kind: ActionKind
    name: str
    arguments: Mapping[str, Any] = field(
        default_factory=dict
    )
    request_id: str = field(
        default_factory=lambda: uuid4().hex
    )
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate the action request."""
        kind = self.kind

        if not isinstance(kind, ActionKind):
            try:
                kind = ActionKind(kind)
            except (TypeError, ValueError) as exc:
                raise ActionContractError(
                    "kind must be a valid ActionKind."
                ) from exc

            object.__setattr__(
                self,
                "kind",
                kind,
            )

        object.__setattr__(
            self,
            "name",
            _validate_non_empty(
                self.name,
                "name",
            ),
        )

        object.__setattr__(
            self,
            "request_id",
            _validate_non_empty(
                self.request_id,
                "request_id",
            ),
        )

        object.__setattr__(
            self,
            "arguments",
            _freeze_mapping(
                self.arguments,
                "arguments",
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(
                self.metadata,
                "metadata",
            ),
        )

    @property
    def backend(self) -> str:
        """Return the backend identifier."""
        return self.kind.value


@dataclass(frozen=True, slots=True)
class ActionResult:
    """A normalized result returned by an action backend."""

    request_id: str
    success: bool
    output: str = ""
    error: str | None = None
    data: Mapping[str, Any] = field(
        default_factory=dict
    )
    metadata: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate and normalize the action result."""
        object.__setattr__(
            self,
            "request_id",
            _validate_non_empty(
                self.request_id,
                "request_id",
            ),
        )

        if not isinstance(self.success, bool):
            raise ActionContractError(
                "success must be a boolean."
            )

        if not isinstance(self.output, str):
            raise ActionContractError(
                "output must be a string."
            )

        if self.error is not None:
            object.__setattr__(
                self,
                "error",
                _validate_non_empty(
                    self.error,
                    "error",
                ),
            )

        object.__setattr__(
            self,
            "data",
            _freeze_mapping(
                self.data,
                "data",
            ),
        )

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(
                self.metadata,
                "metadata",
            ),
        )

        if self.success and self.error is not None:
            raise ActionContractError(
                "successful action results cannot contain an error."
            )

        if not self.success and self.error is None:
            raise ActionContractError(
                "failed action results must contain an error."
            )

    @classmethod
    def succeeded(
        cls,
        request_id: str,
        output: str = "",
        *,
        data: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionResult":
        """Create a successful action result."""
        return cls(
            request_id=request_id,
            success=True,
            output=output,
            data={} if data is None else data,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def failed(
        cls,
        request_id: str,
        error: str,
        *,
        output: str = "",
        data: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ActionResult":
        """Create a failed action result."""
        return cls(
            request_id=request_id,
            success=False,
            output=output,
            error=error,
            data={} if data is None else data,
            metadata={} if metadata is None else metadata,
        )
