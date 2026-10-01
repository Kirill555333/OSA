"""Unit tests for DesktopActionSafetyPolicy and composite policies (0.9.2)."""

from __future__ import annotations

import pytest

from osa.actions import (
    ActionConfirmationHandler,
    ActionDecision,
    ActionKind,
    ActionRequest,
    ActionRouter,
    ActionSafetyPipeline,
    AllowAllActionPolicy,
    CompositeActionPolicy,
    DesktopActionSafetyPolicy,
    DesktopSafetyConfig,
)
from osa.actions.desktop import DesktopActionAdapter
from osa.desktop.automation import create_fake_desktop_automation


class MockConfirmationHandler(ActionConfirmationHandler):
    """Predictable confirmation handler for tests."""

    def __init__(self, should_confirm: bool = True) -> None:
        self.should_confirm = should_confirm
        self.invocations: list[tuple[ActionRequest, str]] = []

    def confirm(self, request: ActionRequest, reason: str) -> bool:
        self.invocations.append((request, reason))
        return self.should_confirm


def test_non_desktop_action_allowed() -> None:
    policy = DesktopActionSafetyPolicy()
    req = ActionRequest(
        kind=ActionKind.TOOL,
        name="calculator",
        arguments={"expression": "1+1"},
        request_id="req-tool",
    )
    decision = policy.evaluate(req)
    assert decision.decision is ActionDecision.ALLOW


def test_read_only_actions_allowed() -> None:
    policy = DesktopActionSafetyPolicy()
    for action in ("desktop_screenshot", "desktop_applications", "desktop_windows", "desktop_health_check"):
        req = ActionRequest(
            kind=ActionKind.DESKTOP,
            name=action,
            arguments={},
            request_id=f"req-{action}",
        )
        assert policy.evaluate(req).decision is ActionDecision.ALLOW


def test_read_only_mode_blocks_mutations() -> None:
    config = DesktopSafetyConfig(allow_mutation=False)
    policy = DesktopActionSafetyPolicy(config)

    # Screenshot is read-only -> ALLOW
    req_shot = ActionRequest(kind=ActionKind.DESKTOP, name="desktop_screenshot", arguments={}, request_id="r1")
    assert policy.evaluate(req_shot).decision is ActionDecision.ALLOW

    # Click is mutation -> DENY
    req_click = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_click_point",
        arguments={"x": 100, "y": 100},
        request_id="r2",
    )
    res_click = policy.evaluate(req_click)
    assert res_click.decision is ActionDecision.DENY
    assert "read-only" in res_click.reason


def test_click_point_coordinate_validation() -> None:
    config = DesktopSafetyConfig(max_screen_bounds=(1920, 1080))
    policy = DesktopActionSafetyPolicy(config)

    # Valid coordinates
    req_ok = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_click_point",
        arguments={"x": 500, "y": 400},
        request_id="r_ok",
    )
    assert policy.evaluate(req_ok).decision is ActionDecision.ALLOW

    # Negative coordinates
    req_neg = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_click_point",
        arguments={"x": -10, "y": 100},
        request_id="r_neg",
    )
    res_neg = policy.evaluate(req_neg)
    assert res_neg.decision is ActionDecision.DENY
    assert "cannot be negative" in res_neg.reason

    # Out of bounds
    req_oob = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_click_point",
        arguments={"x": 2000, "y": 100},
        request_id="r_oob",
    )
    res_oob = policy.evaluate(req_oob)
    assert res_oob.decision is ActionDecision.DENY
    assert "exceed screen bounds" in res_oob.reason


def test_app_termination_and_window_closing_require_confirmation() -> None:
    policy = DesktopActionSafetyPolicy()

    # Close regular application
    req_close = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_close_application",
        arguments={"name": "Calculator"},
        request_id="r_close",
    )
    res_close = policy.evaluate(req_close)
    assert res_close.decision is ActionDecision.CONFIRM

    # Close protected application (terminal)
    req_prot = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_close_application",
        arguments={"name": "Terminal"},
        request_id="r_prot",
    )
    res_prot = policy.evaluate(req_prot)
    assert res_prot.decision is ActionDecision.CONFIRM
    assert "protected application" in res_prot.reason

    # Close window
    req_win = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_close_window",
        arguments={"title": "Document 1"},
        request_id="r_win",
    )
    assert policy.evaluate(req_win).decision is ActionDecision.CONFIRM


def test_hotkey_safety_blocked_and_confirmation() -> None:
    policy = DesktopActionSafetyPolicy()

    # Destructive hotkey (Cmd+Q)
    req_quit = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_hotkey",
        arguments={"keys": ["command", "q"]},
        request_id="r_quit",
    )
    res_quit = policy.evaluate(req_quit)
    assert res_quit.decision is ActionDecision.CONFIRM

    # Blocked system hotkey (Ctrl+Alt+Del)
    req_block = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_hotkey",
        arguments={"keys": ["ctrl", "alt", "delete"]},
        request_id="r_block",
    )
    res_block = policy.evaluate(req_block)
    assert res_block.decision is ActionDecision.DENY
    assert "blocked" in res_block.reason


def test_composite_action_policy() -> None:
    desktop_policy = DesktopActionSafetyPolicy()
    allow_all = AllowAllActionPolicy()

    composite = CompositeActionPolicy((allow_all, desktop_policy))

    req_blocked = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_hotkey",
        arguments={"keys": ["ctrl", "alt", "delete"]},
        request_id="r1",
    )
    assert composite.evaluate(req_blocked).decision is ActionDecision.DENY

    req_confirm = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_close_window",
        arguments={"title": "Main"},
        request_id="r2",
    )
    assert composite.evaluate(req_confirm).decision is ActionDecision.CONFIRM


def test_pipeline_integration_with_desktop_safety() -> None:
    fake_backend = create_fake_desktop_automation()
    adapter = DesktopActionAdapter(fake_backend)
    router = ActionRouter()
    router.register(ActionKind.DESKTOP, adapter)

    policy = DesktopActionSafetyPolicy()
    confirm_handler = MockConfirmationHandler(should_confirm=True)

    pipeline = ActionSafetyPipeline(
        safety_policy=policy,
        confirmation_handler=confirm_handler,
        router=router,
    )

    # 1. Close application with confirmed user -> succeeds
    launched_app = fake_backend.launch_application("test-app")
    req_close = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_close_application",
        arguments={"name": launched_app.application_id},
        request_id="r_exec",
    )
    res = pipeline.dispatch(req_close)

    assert res.success is True
    assert len(confirm_handler.invocations) == 1

    # 2. Blocked hotkey -> denied without calling confirmation
    req_block = ActionRequest(
        kind=ActionKind.DESKTOP,
        name="desktop_hotkey",
        arguments={"keys": ["ctrl", "alt", "delete"]},
        request_id="r_block",
    )
    res_block = pipeline.dispatch(req_block)

    assert res_block.success is False
    assert "blocked" in (res_block.error or "")
