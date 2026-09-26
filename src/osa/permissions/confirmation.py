"""Confirmation handling for OSA permissions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


ConfirmationCallback = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class ConfirmationRequest:
    """Information presented when a tool requires confirmation."""

    tool_name: str
    description: str


class ConfirmationHandler:
    """Handle permission confirmations."""

    def __init__(
        self,
        callback: ConfirmationCallback | None = None,
    ) -> None:
        self._callback = callback

    def request(
        self,
        confirmation: ConfirmationRequest,
    ) -> bool:
        """Request confirmation for an operation."""
        if self._callback is None:
            return False

        return bool(
            self._callback(
                confirmation.description
            )
        )
