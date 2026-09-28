from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

from osa.permissions import PermissionLevel, PermissionPolicy


class AutonomousSafetyError(RuntimeError):
    """Raised when autonomous execution is not authorized."""


class TaskToolResolver(Protocol):
    """Resolve an internal task id to the tool that will execute it."""

    def __call__(self, run_id: str, task_id: str) -> str | None:
        ...


class TaskConfirmationCallback(Protocol):
    """Request explicit approval for a confirmation-level action."""

    def __call__(
        self,
        run_id: str,
        task_id: str,
        tool_name: str,
    ) -> bool:
        ...


@dataclass(frozen=True)
class AutonomousAuthorization:
    allowed: bool
    tool_name: str | None = None
    reason: str | None = None


class AutonomousSafetyGate:
    """
    Permission gate for autonomous task execution.

    Safety is fail-closed:
    - DENY -> never execute
    - CONFIRM -> execute only after explicit confirmation
    - ALLOW -> execute
    - unknown/unresolvable tool -> deny
    """

    def __init__(
        self,
        permissions: PermissionPolicy,
        *,
        resolver: TaskToolResolver,
        confirmation: TaskConfirmationCallback | None = None,
        additional_denied_tools: Iterable[str] = (),
    ) -> None:
        self._permissions = permissions
        self._resolver = resolver
        self._confirmation = confirmation
        self._additional_denied_tools = frozenset(
            additional_denied_tools
        )

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousAuthorization:
        tool_name = self._resolver(run_id, task_id)

        if not tool_name:
            return AutonomousAuthorization(
                allowed=False,
                reason="tool_resolution_failed",
            )

        if tool_name in self._additional_denied_tools:
            return AutonomousAuthorization(
                allowed=False,
                tool_name=tool_name,
                reason="tool_blocked_by_autonomous_policy",
            )

        decision = self._permissions.decide(tool_name)

        if decision.level == PermissionLevel.DENY:
            return AutonomousAuthorization(
                allowed=False,
                tool_name=tool_name,
                reason="permission_denied",
            )

        if decision.level == PermissionLevel.CONFIRM:
            if self._confirmation is None:
                return AutonomousAuthorization(
                    allowed=False,
                    tool_name=tool_name,
                    reason="confirmation_required",
                )

            confirmed = self._confirmation(
                run_id,
                task_id,
                tool_name,
            )

            if not confirmed:
                return AutonomousAuthorization(
                    allowed=False,
                    tool_name=tool_name,
                    reason="confirmation_rejected",
                )

        return AutonomousAuthorization(
            allowed=True,
            tool_name=tool_name,
            reason="authorized",
        )


class AllowlistedAutonomousSafetyGate:
    """
    Simple fail-closed gate useful for highly restricted autonomous jobs.
    """

    def __init__(self, allowed_tools: Iterable[str]) -> None:
        self._allowed_tools = frozenset(allowed_tools)

    def authorize(
        self,
        run_id: str,
        task_id: str,
        *,
        tool_name: str | None,
    ) -> AutonomousAuthorization:
        if not tool_name:
            return AutonomousAuthorization(
                allowed=False,
                reason="tool_resolution_failed",
            )

        if tool_name not in self._allowed_tools:
            return AutonomousAuthorization(
                allowed=False,
                tool_name=tool_name,
                reason="tool_not_allowlisted",
            )

        return AutonomousAuthorization(
            allowed=True,
            tool_name=tool_name,
            reason="tool_allowlisted",
        )
