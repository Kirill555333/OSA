"""Tests for the protected foreground application rule (OSA 1.5.1.4)."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

from osa.actions.contracts import ActionKind, ActionRequest, ActionResult
from osa.actions.desktop_tool_safety import (
    INPUT_TOOLS,
    DesktopToolSafetyPolicy,
    ForegroundProbe,
)
from osa.actions.pipeline import (
    ActionDecision,
    ActionSafetyPipeline,
    CallbackActionConfirmationHandler,
)
from osa.actions.router import ActionRouter


def _request(name: str, **arguments: object) -> ActionRequest:
    return ActionRequest(kind=ActionKind.TOOL, name=name, arguments=arguments)


def _arguments_for(name: str) -> dict[str, object]:
    return {
        "type_text": {"text": "ls -la"},
        "press_key": {"key": "enter"},
        "hotkey": {"keys": ["command", "c"]},
        "click_mouse": {"x": 100, "y": 100},
    }[name]


def _policy(application: str | None, **kwargs) -> DesktopToolSafetyPolicy:
    return DesktopToolSafetyPolicy(foreground=lambda: application, **kwargs)


# --- protected foreground application ----------------------------------------------


@pytest.mark.parametrize("tool", sorted(INPUT_TOOLS))
@pytest.mark.parametrize("application", ["Terminal", "iTerm2", "Code", "PowerShell", "zsh"])
def test_input_into_a_protected_application_requires_confirmation(
    tool: str,
    application: str,
) -> None:
    decision = _policy(application).evaluate(_request(tool, **_arguments_for(tool)))

    assert decision.decision is ActionDecision.CONFIRM
    assert application in decision.reason
    assert tool in decision.reason


@pytest.mark.parametrize("tool", sorted(INPUT_TOOLS))
@pytest.mark.parametrize("application", ["Safari", "Notes", "Google Chrome", "Telegram"])
def test_input_into_an_ordinary_application_is_allowed(tool: str, application: str) -> None:
    decision = _policy(application).evaluate(_request(tool, **_arguments_for(tool)))

    assert decision.decision is ActionDecision.ALLOW


def test_application_names_are_compared_case_insensitively() -> None:
    decision = _policy("  TERMINAL ").evaluate(_request("type_text", text="x"))

    assert decision.decision is ActionDecision.CONFIRM


@pytest.mark.parametrize("application", [None, "", "   "])
def test_unknown_foreground_application_does_not_block_input(application) -> None:
    decision = _policy(application).evaluate(_request("type_text", text="x"))

    assert decision.decision is ActionDecision.ALLOW


def test_failing_foreground_provider_does_not_block_input() -> None:
    def broken() -> str | None:
        raise RuntimeError("no accessibility permission")

    policy = DesktopToolSafetyPolicy(foreground=broken)

    assert policy.evaluate(_request("type_text", text="x")).decision is ActionDecision.ALLOW


def test_without_a_foreground_provider_the_rule_is_inactive() -> None:
    decision = DesktopToolSafetyPolicy().evaluate(_request("type_text", text="x"))

    assert decision.decision is ActionDecision.ALLOW


@pytest.mark.parametrize("tool", ["open_url", "launch_application", "read_file", "calculator"])
def test_tools_that_do_not_type_are_not_affected_by_the_foreground(tool: str) -> None:
    decision = _policy("Terminal").evaluate(_request(tool, anything="value"))

    assert decision.decision is ActionDecision.ALLOW


def test_provider_is_not_asked_for_tools_that_do_not_type() -> None:
    asked: list[str] = []

    def provider() -> str:
        asked.append("asked")
        return "Terminal"

    DesktopToolSafetyPolicy(foreground=provider).evaluate(_request("open_url", url="x"))

    assert asked == []


# --- precedence --------------------------------------------------------------------


def test_blocked_hotkey_is_denied_even_in_an_ordinary_application() -> None:
    decision = _policy("Notes").evaluate(_request("hotkey", keys=["ctrl", "alt", "delete"]))

    assert decision.decision is ActionDecision.DENY


def test_blocked_hotkey_is_denied_not_confirmed_in_a_protected_application() -> None:
    decision = _policy("Terminal").evaluate(_request("hotkey", keys=["ctrl", "alt", "delete"]))

    assert decision.decision is ActionDecision.DENY


def test_destructive_hotkey_asks_about_termination_in_an_ordinary_application() -> None:
    decision = _policy("Notes").evaluate(_request("hotkey", keys=["command", "q"]))

    assert decision.decision is ActionDecision.CONFIRM
    assert "termination" in decision.reason


def test_negative_click_is_denied_in_a_protected_application() -> None:
    decision = _policy("Terminal").evaluate(_request("click_mouse", x=-5, y=10))

    assert decision.decision is ActionDecision.DENY


# --- skip list ---------------------------------------------------------------------


def test_skipped_tool_is_left_to_other_policies() -> None:
    policy = _policy("Terminal", skip_tools=frozenset({"close_application"}))

    closing = policy.evaluate(_request("close_application", application_name="Notes"))
    typing = policy.evaluate(_request("type_text", text="x"))

    assert closing.decision is ActionDecision.ALLOW
    assert typing.decision is ActionDecision.CONFIRM


def test_close_application_still_asks_when_it_is_not_skipped() -> None:
    decision = _policy("Notes").evaluate(_request("close_application", application_name="Notes"))

    assert decision.decision is ActionDecision.CONFIRM


# --- pipeline ----------------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, request: ActionRequest) -> ActionResult:
        self.executed.append(request.name)
        return ActionResult.succeeded(request.request_id, output="done")


def _pipeline(recorder: _Recorder, application: str | None, confirm: bool) -> ActionSafetyPipeline:
    return ActionSafetyPipeline(
        ActionRouter({ActionKind.TOOL: recorder}),
        safety_policy=_policy(application),
        confirmation_handler=CallbackActionConfirmationHandler(lambda request, reason: confirm),
    )


def test_typing_into_a_terminal_is_blocked_when_not_confirmed() -> None:
    recorder = _Recorder()

    result = _pipeline(recorder, "Terminal", confirm=False).dispatch(
        _request("type_text", text="rm -rf ~")
    )

    assert result.success is False
    assert result.metadata["pipeline_error"] == "not_confirmed"
    assert recorder.executed == []


def test_typing_into_a_terminal_runs_when_confirmed() -> None:
    recorder = _Recorder()

    result = _pipeline(recorder, "Terminal", confirm=True).dispatch(
        _request("type_text", text="ls")
    )

    assert result.success is True
    assert recorder.executed == ["type_text"]


def test_typing_into_an_ordinary_application_needs_no_confirmation() -> None:
    recorder = _Recorder()

    result = _pipeline(recorder, "Notes", confirm=False).dispatch(
        _request("type_text", text="hello")
    )

    assert result.success is True
    assert recorder.executed == ["type_text"]


# --- foreground probe --------------------------------------------------------------


class _Backend:
    def __init__(self, name) -> None:
        self._name = name

    def foreground_application(self):
        if isinstance(self._name, Exception):
            raise self._name

        return self._name


def test_probe_returns_the_name_reported_by_the_backend() -> None:
    assert ForegroundProbe(_Backend("Terminal"))() == "Terminal"
    assert ForegroundProbe(_Backend("  Safari \n"))() == "Safari"


@pytest.mark.parametrize("reported", [None, "", "   "])
def test_probe_reports_unknown_for_empty_answers(reported) -> None:
    assert ForegroundProbe(_Backend(reported))() is None


def test_probe_reports_unknown_when_the_backend_fails() -> None:
    assert ForegroundProbe(_Backend(RuntimeError("osascript failed")))() is None


def test_probe_reports_unknown_when_the_backend_has_no_such_capability() -> None:
    class _Plain:
        pass

    assert ForegroundProbe(_Plain())() is None


# --- wiring ------------------------------------------------------------------------


def test_agent_accepts_an_action_safety_policy() -> None:
    try:
        spec = importlib.util.find_spec("osa.core.agent")
    except ModuleNotFoundError:
        return

    if spec is None:
        return

    from osa.core.agent import Agent

    assert "action_safety_policy" in inspect.signature(Agent.__init__).parameters


def test_main_wires_the_desktop_tool_safety_policy_into_the_agent() -> None:
    try:
        spec = importlib.util.find_spec("osa.main")
    except ModuleNotFoundError:
        return

    if spec is None or spec.origin is None:
        return

    source = Path(spec.origin).read_text(encoding="utf-8")

    assert "DesktopToolSafetyPolicy(" in source
    assert "action_safety_policy=action_safety_policy" in source
    assert 'skip_tools=frozenset({"close_application"})' in source
