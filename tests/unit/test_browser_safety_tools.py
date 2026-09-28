from __future__ import annotations

from osa.browser import (
    BrowserSafetyPolicy,
    create_fake_browser_automation,
)
from osa.tools.browser_automation import (
    BrowserClickTool,
    BrowserOpenTool,
    BrowserTypeTool,
)


def test_browser_open_uses_safety_policy() -> None:
    browser = create_fake_browser_automation()

    policy = BrowserSafetyPolicy(
        blocked_domains=frozenset(
            {"danger.example"}
        )
    )

    tool = BrowserOpenTool(
        browser,
        safety_policy=policy,
    )

    result = tool.execute(
        {
            "url": "https://danger.example",
        }
    )

    assert result.success is False
    assert "blocked" in result.error
    assert browser.tabs() == ()


def test_browser_open_allows_safe_domain() -> None:
    browser = create_fake_browser_automation()

    policy = BrowserSafetyPolicy(
        allowed_domains=frozenset(
            {"example.com"}
        )
    )

    tool = BrowserOpenTool(
        browser,
        safety_policy=policy,
    )

    result = tool.execute(
        {
            "url": "https://example.com",
        }
    )

    assert result.success is True
    assert len(browser.tabs()) == 1


def test_browser_click_uses_current_page_safety() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    policy = BrowserSafetyPolicy()

    tool = BrowserClickTool(
        browser,
        safety_policy=policy,
    )

    result = tool.execute(
        {
            "locator": {
                "kind": "css",
                "value": "#button",
            }
        }
    )

    assert result.success is True


def test_browser_type_blocks_sensitive_css_field() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    policy = BrowserSafetyPolicy()

    tool = BrowserTypeTool(
        browser,
        safety_policy=policy,
    )

    result = tool.execute(
        {
            "locator": {
                "kind": "css",
                "value": "#password",
            },
            "text": "secret",
        }
    )

    assert result.success is False
    assert "sensitive field" in result.error


def test_browser_type_allows_normal_field() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    policy = BrowserSafetyPolicy()

    tool = BrowserTypeTool(
        browser,
        safety_policy=policy,
    )

    result = tool.execute(
        {
            "locator": {
                "kind": "css",
                "value": "#search",
            },
            "text": "OSA",
        }
    )

    assert result.success is True
