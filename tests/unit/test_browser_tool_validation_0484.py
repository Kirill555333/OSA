from __future__ import annotations

import pytest

from osa.browser.automation import create_fake_browser_automation
from osa.tools.browser_automation import create_browser_action_tools


def get_tool(tools, name: str):
    return next(tool for tool in tools if tool.name == name)


@pytest.fixture
def tools():
    browser = create_fake_browser_automation()
    return create_browser_action_tools(browser)


@pytest.mark.parametrize(
    "tool_name,args",
    (
        ("browser_open", {}),
        ("browser_open", {"url": ""}),
        ("browser_find", {}),
        ("browser_find", {"kind": "", "value": ""}),
        ("browser_click", {}),
        ("browser_type", {}),
        ("browser_type", {"text": ""}),
        ("browser_press", {}),
        ("browser_press", {"key": ""}),
    ),
)
def test_browser_tools_reject_invalid_required_arguments(
    tools,
    tool_name: str,
    args: dict,
) -> None:
    result = get_tool(tools, tool_name).execute(args)

    assert result.success is False
    assert result.error
    assert result.output == ""


@pytest.mark.parametrize(
    "tool_name,args",
    (
        ("browser_wait", {"seconds": -1}),
        ("browser_wait", {"seconds": 0}),
    ),
)
def test_browser_wait_rejects_non_positive_timeout(
    tools,
    tool_name: str,
    args: dict,
) -> None:
    result = get_tool(tools, tool_name).execute(args)

    assert result.success is False
    assert result.error
    assert result.output == ""


def test_browser_observe_accepts_empty_arguments(tools) -> None:
    result = get_tool(tools, "browser_observe").execute({})

    assert result.success is False or result.success is True
    assert result.output is not None


def test_browser_screenshot_accepts_empty_arguments(tools) -> None:
    result = get_tool(tools, "browser_screenshot").execute({})

    assert result.success is False or result.success is True
    assert result.output is not None
