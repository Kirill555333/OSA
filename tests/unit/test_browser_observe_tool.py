from __future__ import annotations

from osa.browser import create_fake_browser_automation
from osa.tools.browser_automation import (
    BrowserObserveTool,
    create_browser_action_tools,
)


def test_browser_observe_tool() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    tool = BrowserObserveTool(browser)

    result = tool.execute({})

    assert result.success is True
    assert "URL: https://example.com" in result.output
    assert "Title: (untitled)" in result.output
    assert "Page text:" in result.output
    assert result.metadata["tab_id"] == "tab-1"


def test_browser_observe_limits() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    tool = BrowserObserveTool(browser)

    result = tool.execute(
        {
            "max_text_characters": 1,
            "max_elements": 1,
        }
    )

    assert result.success is True
    assert result.metadata["text_truncated"] is True


def test_browser_observe_rejects_invalid_limit() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    tool = BrowserObserveTool(browser)

    result = tool.execute(
        {
            "max_elements": 0,
        }
    )

    assert result.success is False
    assert "between 1 and 500" in result.error


def test_browser_observe_is_in_action_factory() -> None:
    browser = create_fake_browser_automation()

    tools = create_browser_action_tools(browser)

    assert "browser_observe" in [
        tool.name
        for tool in tools
    ]
