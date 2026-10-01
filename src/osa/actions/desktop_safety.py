"""Safety policies for desktop automation actions in OSA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from osa.actions.contracts import ActionKind, ActionRequest
from osa.actions.pipeline import (
    ActionDecision,
    ActionPolicy,
    ActionPolicyDecision,
)


@dataclass(frozen=True, slots=True)
class DesktopSafetyConfig:
    """Configuration for desktop action safety rules."""

    allow_mutation: bool = True
    confirm_app_termination: bool = True
    confirm_window_closing: bool = True
    confirm_destructive_hotkeys: bool = True
    protected_applications: frozenset[str] = frozenset(
        {
            "terminal",
            "iterm2",
            "iterm",
            "bash",
            "zsh",
            "powershell",
            "cmd",
            "code",
        }
    )
    max_screen_bounds: tuple[int, int] | None = None


class DesktopActionSafetyPolicy:
    """Safety evaluation policy for desktop actions, enforcing confirmation and guardrails."""

    READ_ONLY_ACTIONS = frozenset(
        {
            "desktop_screenshot",
            "desktop_applications",
            "desktop_windows",
            "desktop_find_element",
            "desktop_health_check",
        }
    )

    MUTATING_ACTIONS = frozenset(
        {
            "desktop_launch_application",
            "desktop_close_application",
            "desktop_activate_window",
            "desktop_close_window",
            "desktop_click_element",
            "desktop_click_point",
            "desktop_type",
            "desktop_press",
            "desktop_hotkey",
        }
    )

    DESTRUCTIVE_HOTKEYS = frozenset(
        {
            frozenset({"command", "q"}),
            frozenset({"cmd", "q"}),
            frozenset({"alt", "f4"}),
            frozenset({"ctrl", "q"}),
            frozenset({"control", "q"}),
            frozenset({"command", "shift", "q"}),
            frozenset({"command", "option", "esc"}),
        }
    )

    BLOCKED_HOTKEYS = frozenset(
        {
            frozenset({"ctrl", "alt", "delete"}),
            frozenset({"control", "alt", "delete"}),
            frozenset({"ctrl", "alt", "del"}),
        }
    )

    def __init__(self, config: DesktopSafetyConfig | None = None) -> None:
        self._config = config or DesktopSafetyConfig()

    @property
    def config(self) -> DesktopSafetyConfig:
        """Return the active safety configuration."""
        return self._config

    def evaluate(self, request: ActionRequest) -> ActionPolicyDecision:
        """Evaluate action against desktop safety invariants."""
        if request.kind is not ActionKind.DESKTOP:
            return ActionPolicyDecision(ActionDecision.ALLOW)

        action_name = request.name

        # 1. Read-Only mode restriction
        if not self._config.allow_mutation and action_name in self.MUTATING_ACTIONS:
            return ActionPolicyDecision(
                ActionDecision.DENY,
                f"Desktop mutating action '{action_name}' is forbidden in read-only mode.",
            )

        # 2. Coordinate boundary verification
        if action_name == "desktop_click_point":
            x = request.arguments.get("x")
            y = request.arguments.get("y")
            if isinstance(x, int) and isinstance(y, int) and not isinstance(x, bool) and not isinstance(y, bool):
                if x < 0 or y < 0:
                    return ActionPolicyDecision(
                        ActionDecision.DENY,
                        f"Screen coordinates cannot be negative: ({x}, {y}).",
                    )
                if self._config.max_screen_bounds:
                    max_w, max_h = self._config.max_screen_bounds
                    if x > max_w or y > max_h:
                        return ActionPolicyDecision(
                            ActionDecision.DENY,
                            f"Click coordinates ({x}, {y}) exceed screen bounds ({max_w}, {max_h}).",
                        )

        # 3. Application termination guardrails
        if action_name == "desktop_close_application":
            app_name = str(request.arguments.get("name", "")).strip().casefold()
            if app_name in self._config.protected_applications:
                return ActionPolicyDecision(
                    ActionDecision.CONFIRM,
                    f"Closing protected application '{app_name}' requires explicit confirmation.",
                )
            if self._config.confirm_app_termination:
                return ActionPolicyDecision(
                    ActionDecision.CONFIRM,
                    f"Terminating application '{app_name or 'unnamed'}' requires confirmation.",
                )

        # 4. Window closing guardrails
        if action_name == "desktop_close_window":
            title = str(request.arguments.get("title", "")).strip()
            if self._config.confirm_window_closing:
                return ActionPolicyDecision(
                    ActionDecision.CONFIRM,
                    f"Closing desktop window '{title or 'unnamed'}' requires confirmation.",
                )

        # 5. Hotkey safety checks
        if action_name == "desktop_hotkey":
            raw_keys = request.arguments.get("keys", ())
            if isinstance(raw_keys, (list, tuple)):
                normalized_keys = frozenset(
                    str(k).strip().casefold() for k in raw_keys if str(k).strip()
                )
                if normalized_keys in self.BLOCKED_HOTKEYS:
                    return ActionPolicyDecision(
                        ActionDecision.DENY,
                        f"Hotkey combination '{'+'.join(sorted(normalized_keys))}' is blocked for system safety.",
                    )
                if self._config.confirm_destructive_hotkeys and normalized_keys in self.DESTRUCTIVE_HOTKEYS:
                    return ActionPolicyDecision(
                        ActionDecision.CONFIRM,
                        f"Executing application termination hotkey '{'+'.join(sorted(normalized_keys))}' requires confirmation.",
                    )

        return ActionPolicyDecision(ActionDecision.ALLOW)


class CompositeActionPolicy:
    """Evaluates multiple policies in sequence; DENY overrides CONFIRM, which overrides ALLOW."""

    def __init__(self, policies: Sequence[ActionPolicy]) -> None:
        self._policies = tuple(policies)

    def evaluate(self, request: ActionRequest) -> ActionPolicyDecision:
        """Evaluate request across all chained policies."""
        final_decision = ActionPolicyDecision(ActionDecision.ALLOW)

        for policy in self._policies:
            res = policy.evaluate(request)
            if res.decision is ActionDecision.DENY:
                return res
            if res.decision is ActionDecision.CONFIRM and final_decision.decision is not ActionDecision.CONFIRM:
                final_decision = res

        return final_decision
