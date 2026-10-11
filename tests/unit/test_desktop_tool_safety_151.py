"""Tests for DesktopToolSafetyPolicy (OSA 1.5.1.2a)."""

from __future__ import annotations

import pytest

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.actions.desktop_safety import DesktopActionSafetyPolicy, DesktopSafetyConfig
from osa.actions.desktop_tool_safety import DesktopToolSafetyPolicy
from osa.actions.pipeline import (
    ActionDecision,
    ActionSafetyPipeline,
    CallbackActionConfirmationHandler,
)
from osa.actions.router import ActionRouter


def _tool_request(name: str, **arguments: object) -> ActionRequest:
    return ActionRequest(kind=ActionKind.TOOL, name=name, arguments=arguments)


class _RecordingHandler:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, request: ActionRequest) -> ActionResult:
        self.executed.append(request.name)
        return ActionResult.succeeded(request.request_id, output="done")


# --- close_application ---------------------------------------------------------------


def test_close_application_requires_confirmation() -> None:
    decision = DesktopToolSafetyPolicy().evaluate(
        _tool_request("close_application", application_name="Google Chrome")
    )

    assert decision.decision is ActionDecision.CONFIRM
    assert "google chrome" in decision.reason.lower()


@pytest.mark.parametrize("name", ["Terminal", "iTerm2", "Code", "zsh"])
def test_close_protected_application_requires_confirmation(name: str) -> None:
    decision = DesktopToolSafetyPolicy().evaluate(
        _tool_request("close_application", application_name=name)
    )

    assert decision.decision is ActionDecision.CONFIRM
    assert "protected" in decision.reason.lower()


def test_relaxed_config_still_protects_terminal_but_not_other_apps() -> None:
    policy = DesktopToolSafetyPolicy(
        DesktopActionSafetyPolicy(DesktopSafetyConfig(confirm_app_termination=False))
    )

    relaxed = policy.evaluate(_tool_request("close_application", application_name="Notes"))
    protected = policy.evaluate(_tool_request("close_application", application_name="Terminal"))

    assert relaxed.decision is ActionDecision.ALLOW
    assert protected.decision is ActionDecision.CONFIRM


# --- hotkey --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keys",
    [
        ["command", "q"],
        ["cmd", "q"],
        ["Command", "Q"],
        ["q", "command"],
        ["alt", "f4"],
        ["ctrl", "q"],
    ],
)
def test_destructive_hotkeys_require_confirmation(keys: list[str]) -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_tool_request("hotkey", keys=keys))

    assert decision.decision is ActionDecision.CONFIRM


@pytest.mark.parametrize(
    "keys",
    [["ctrl", "alt", "delete"], ["control", "alt", "delete"], ["ctrl", "alt", "del"]],
)
def test_blocked_hotkeys_are_denied(keys: list[str]) -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_tool_request("hotkey", keys=keys))

    assert decision.decision is ActionDecision.DENY


@pytest.mark.parametrize("keys", [["command", "c"], ["command", "k"], ["ctrl", "f"]])
def test_ordinary_hotkeys_are_allowed(keys: list[str]) -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_tool_request("hotkey", keys=keys))

    assert decision.decision is ActionDecision.ALLOW


# --- click_mouse ---------------------------------------------------------------------


def test_negative_click_coordinates_are_denied() -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_tool_request("click_mouse", x=-1, y=10))

    assert decision.decision is ActionDecision.DENY


def test_negative_click_coordinates_given_as_strings_are_denied() -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_tool_request("click_mouse", x="-5", y="10"))

    assert decision.decision is ActionDecision.DENY


def test_valid_click_coordinates_are_allowed() -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_tool_request("click_mouse", x=100, y=200))

    assert decision.decision is ActionDecision.ALLOW


# --- everything else is untouched ------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["open_url", "launch_application", "type_text", "press_key", "calculator", "read_file"],
)
def test_other_tools_are_allowed(name: str) -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_tool_request(name, anything="value"))

    assert decision.decision is ActionDecision.ALLOW


def test_non_tool_requests_are_not_evaluated() -> None:
    request = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_close_application",
        arguments={"name": "Notes"},
    )

    assert DesktopToolSafetyPolicy().evaluate(request).decision is ActionDecision.ALLOW


# --- integration with the real safety pipeline -------------------------------------------


def _pipeline(handler: _RecordingHandler, confirm: bool | None) -> ActionSafetyPipeline:
    confirmation = None

    if confirm is not None:
        confirmation = CallbackActionConfirmationHandler(lambda request, reason: confirm)

    return ActionSafetyPipeline(
        ActionRouter({ActionKind.TOOL: handler}),
        safety_policy=DesktopToolSafetyPolicy(),
        confirmation_handler=confirmation,
    )


def test_pipeline_blocks_close_application_when_not_confirmed() -> None:
    handler = _RecordingHandler()

    result = _pipeline(handler, confirm=False).dispatch(
        _tool_request("close_application", application_name="Notes")
    )

    assert result.success is False
    assert result.metadata["pipeline_error"] == "not_confirmed"
    assert handler.executed == []


def test_pipeline_runs_close_application_when_confirmed() -> None:
    handler = _RecordingHandler()

    result = _pipeline(handler, confirm=True).dispatch(
        _tool_request("close_application", application_name="Notes")
    )

    assert result.success is True
    assert handler.executed == ["close_application"]


def test_pipeline_fails_closed_without_confirmation_handler() -> None:
    handler = _RecordingHandler()

    result = _pipeline(handler, confirm=None).dispatch(
        _tool_request("close_application", application_name="Notes")
    )

    assert result.success is False
    assert result.metadata["pipeline_error"] == "confirmation_not_configured"
    assert handler.executed == []


def test_pipeline_denies_blocked_hotkey_without_asking() -> None:
    handler = _RecordingHandler()

    result = _pipeline(handler, confirm=True).dispatch(
        _tool_request("hotkey", keys=["ctrl", "alt", "delete"])
    )

    assert result.success is False
    assert result.metadata["pipeline_error"] == "denied"
    assert handler.executed == []


def test_pipeline_does_not_interfere_with_ordinary_tools() -> None:
    handler = _RecordingHandler()

    result = _pipeline(handler, confirm=False).dispatch(
        _tool_request("open_url", url="https://example.com")
    )

    assert result.success is True
    assert handler.executed == ["open_url"]
