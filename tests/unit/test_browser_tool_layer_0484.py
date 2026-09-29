from __future__ import annotations

from osa.browser.automation import (
    BrowserAutomationError,
    create_fake_browser_automation,
)
from osa.tools.browser_automation import create_browser_action_tools


EXPECTED_TOOL_NAMES = (
    "browser_open",
    "browser_find",
    "browser_click",
    "browser_type",
    "browser_press",
    "browser_read",
    "browser_wait",
    "browser_observe",
    "browser_screenshot",
)


def test_browser_tool_layer_has_stable_public_contract() -> None:
    browser = create_fake_browser_automation()
    tools = create_browser_action_tools(browser)

    assert tuple(tool.name for tool in tools) == EXPECTED_TOOL_NAMES
    assert len({tool.name for tool in tools}) == len(tools)

    for tool in tools:
        assert isinstance(tool.name, str)
        assert tool.name.strip()
        assert isinstance(tool.description, str)
        assert tool.description.strip()
        assert callable(tool.execute)


def test_browser_open_and_observe_share_backend_state() -> None:
    browser = create_fake_browser_automation()
    tools = create_browser_action_tools(browser)

    open_tool = next(
        tool for tool in tools if tool.name == "browser_open"
    )
    observe_tool = next(
        tool for tool in tools if tool.name == "browser_observe"
    )

    opened = open_tool.execute(
        {"url": "https://example.com"}
    )

    assert opened.success is True
    assert opened.error is None

    observed = observe_tool.execute({})

    assert observed.success is True
    assert observed.error is None
    assert "example.com" in observed.output


def test_browser_tool_layer_reports_backend_errors_as_failures() -> None:
    browser = create_fake_browser_automation()

    def failing_get_text(*args, **kwargs) -> str:
        raise BrowserAutomationError(
            "simulated browser backend failure"
        )

    browser.get_text = failing_get_text

    tools = create_browser_action_tools(browser)

    read_tool = next(
        tool for tool in tools if tool.name == "browser_read"
    )

    result = read_tool.execute({})

    assert result.success is False
    assert result.error
    assert "simulated browser backend failure" in result.error
    assert result.output == ""


def test_browser_safety_failure_does_not_execute_navigation() -> None:
    browser = create_fake_browser_automation()
    tools = create_browser_action_tools(browser)

    open_tool = next(
        tool for tool in tools if tool.name == "browser_open"
    )

    result = open_tool.execute(
        {"url": "file:///etc/hosts"}
    )

    assert result.success is False
    assert result.error
    assert browser.tabs() == ()
