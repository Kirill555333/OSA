"""Safety policy that applies desktop rules to model-facing desktop tools (OSA 1.5.1.4)."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from osa.actions.contracts import ActionKind, ActionRequest
from osa.actions.desktop_safety import DesktopActionSafetyPolicy
from osa.actions.pipeline import ActionDecision, ActionPolicyDecision

INPUT_TOOLS = frozenset({"type_text", "press_key", "hotkey", "click_mouse"})


def _as_int(value: Any) -> Any:
    """Return an int for int-like values (including numeric strings), else the value."""
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float) and value.is_integer():
        return int(value)

    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return value

    return value


class ForegroundProbe:
    """Ask the desktop backend which application is in front (None when unknown).

    The backend is created lazily. A backend without ``foreground_application`` (not
    every platform has it yet) simply reports "unknown".
    """

    def __init__(self, backend: Any | None = None) -> None:
        self._backend = backend

    def __call__(self) -> str | None:
        try:
            if self._backend is None:
                from osa.desktop.factory import create_desktop_automation

                self._backend = create_desktop_automation()

            query = getattr(self._backend, "foreground_application", None)

            if not callable(query):
                return None

            name = query()
        except Exception:
            return None

        text = str(name).strip() if name else ""

        return text or None


class DesktopToolSafetyPolicy:
    """Evaluate desktop TOOL requests with the DESKTOP safety rules.

    The agent sends every model tool call as ``ActionKind.TOOL``, so the rules in
    ``DesktopActionSafetyPolicy`` (confirmation for closing applications, protected
    applications, destructive and blocked hotkeys, coordinate checks) would never run
    for the tools the model actually calls. This policy translates the model-facing
    tool name and arguments into the equivalent desktop action and reuses those rules
    without duplicating them. Requests for any other tool are allowed unchanged.

    When a ``foreground`` provider is given, typing, key presses, hotkeys and clicks
    aimed at a protected application (a terminal, a code editor) need confirmation.
    An unknown foreground application does not block anything: it is an extra layer.
    Tools listed in ``skip_tools`` are left to other policies, so a user is not asked
    twice for the same action.
    """

    def __init__(
        self,
        desktop_policy: DesktopActionSafetyPolicy | None = None,
        *,
        foreground: Callable[[], str | None] | None = None,
        skip_tools: frozenset[str] = frozenset(),
    ) -> None:
        self._desktop_policy = desktop_policy or DesktopActionSafetyPolicy()
        self._foreground = foreground
        self._skip_tools = frozenset(skip_tools)

    @property
    def desktop_policy(self) -> DesktopActionSafetyPolicy:
        """Return the wrapped desktop safety policy."""
        return self._desktop_policy

    def evaluate(self, request: ActionRequest) -> ActionPolicyDecision:
        """Evaluate one action request."""
        if request.kind is not ActionKind.TOOL or request.name in self._skip_tools:
            return ActionPolicyDecision(ActionDecision.ALLOW)

        desktop_request = self._to_desktop_request(request)

        if desktop_request is not None:
            decision = self._desktop_policy.evaluate(desktop_request)

            if decision.decision is not ActionDecision.ALLOW:
                return decision

        return self._check_foreground(request)

    def _check_foreground(self, request: ActionRequest) -> ActionPolicyDecision:
        if self._foreground is None or request.name not in INPUT_TOOLS:
            return ActionPolicyDecision(ActionDecision.ALLOW)

        try:
            application = self._foreground()
        except Exception:
            application = None

        if not application:
            return ActionPolicyDecision(ActionDecision.ALLOW)

        protected = self._desktop_policy.config.protected_applications

        if application.strip().casefold() in protected:
            return ActionPolicyDecision(
                ActionDecision.CONFIRM,
                (
                    f"Input into protected application '{application}' "
                    f"({request.name}) requires confirmation."
                ),
            )

        return ActionPolicyDecision(ActionDecision.ALLOW)

    @staticmethod
    def _to_desktop_request(request: ActionRequest) -> ActionRequest | None:
        arguments: Mapping[str, Any] = request.arguments

        if request.name == "close_application":
            return ActionRequest(
                kind=ActionKind.DESKTOP,
                name="desktop_close_application",
                arguments={"name": str(arguments.get("application_name", ""))},
                request_id=request.request_id,
            )

        if request.name == "hotkey":
            return ActionRequest(
                kind=ActionKind.DESKTOP,
                name="desktop_hotkey",
                arguments={"keys": arguments.get("keys", ())},
                request_id=request.request_id,
            )

        if request.name == "click_mouse":
            return ActionRequest(
                kind=ActionKind.DESKTOP,
                name="desktop_click_point",
                arguments={
                    "x": _as_int(arguments.get("x")),
                    "y": _as_int(arguments.get("y")),
                },
                request_id=request.request_id,
            )

        return None
