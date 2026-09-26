"""Permission policy for OSA tool execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping


class PermissionLevel(StrEnum):
    """Available permission levels for tool execution."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """Decision returned by the permission policy."""

    level: PermissionLevel
    reason: str


class PermissionPolicy:
    """Determine whether OSA may execute a specific tool."""

    def __init__(
        self,
        rules: Mapping[str, PermissionLevel] | None = None,
        *,
        default: PermissionLevel = PermissionLevel.DENY,
    ) -> None:
        self._rules = dict(rules or {})
        self._default = default

    def set_rule(
        self,
        tool_name: str,
        level: PermissionLevel,
    ) -> None:
        """Set the permission level for a tool."""
        normalized_name = tool_name.strip()

        if not normalized_name:
            raise ValueError("Tool name cannot be empty.")

        self._rules[normalized_name] = level

    def decide(self, tool_name: str) -> PermissionDecision:
        """Return the permission decision for a tool."""
        normalized_name = tool_name.strip()

        if not normalized_name:
            return PermissionDecision(
                level=PermissionLevel.DENY,
                reason="Tool name cannot be empty.",
            )

        level = self._rules.get(
            normalized_name,
            self._default,
        )

        reasons = {
            PermissionLevel.ALLOW: "Tool is explicitly allowed.",
            PermissionLevel.CONFIRM: "Tool requires user confirmation.",
            PermissionLevel.DENY: "Tool is not permitted by policy.",
        }

        return PermissionDecision(
            level=level,
            reason=reasons[level],
        )
