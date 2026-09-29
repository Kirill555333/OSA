from __future__ import annotations

import pytest

from osa.browser import (
    BrowserAutomationError,
    BrowserLocator,
    BrowserSafetyError,
    BrowserSafetyPolicy,
    create_fake_browser_automation,
)
from osa.tools.browser_automation import create_browser_action_tools


def test_browser_action_factory_has_stable_order() -> None:
    browser = create_fake_browser_automation()

    tools = create_browser_action_tools(browser)

    assert [tool.name for tool in tools] == [
        "browser_open",
        "browser_find",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_read",
        "browser_wait",
        "browser_observe",
        "browser_screenshot",
    ]


def test_browser_safety_rejects_non_http_urls() -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("file:///tmp/test.txt")

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("javascript:alert(1)")


def test_browser_safety_rejects_embedded_credentials() -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(BrowserSafetyError):
        policy.validate_url(
            "https://user:password@example.com"
        )


def test_browser_safety_rejects_local_network_targets() -> None:
    policy = BrowserSafetyPolicy()

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("http://127.0.0.1:8080")

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("http://localhost:3000")


def test_browser_safety_blocks_blocked_domain_subdomains() -> None:
    policy = BrowserSafetyPolicy(
        blocked_domains=("example.com",),
    )

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("https://example.com")

    with pytest.raises(BrowserSafetyError):
        policy.validate_url("https://sub.example.com")


def test_browser_safety_allows_explicit_safe_domain() -> None:
    policy = BrowserSafetyPolicy(
        allowed_domains=("example.com",),
    )

    policy.validate_url("https://example.com/path")
    policy.validate_url("https://sub.example.com/path")


def test_fake_browser_observe_returns_page_state() -> None:
    browser = create_fake_browser_automation()

    browser.open("https://example.com")

    observation = browser.observe()

    assert observation.tab.url == "https://example.com"
    assert isinstance(observation.text_truncated, bool)
    assert isinstance(
        observation.elements_truncated,
        bool,
    )


def test_fake_browser_observe_requires_active_tab() -> None:
    browser = create_fake_browser_automation()

    with pytest.raises(BrowserAutomationError):
        browser.observe()


def test_browser_locator_is_backend_neutral() -> None:
    locator = BrowserLocator(
        kind="text",
        value="OpenAI",
        exact=True,
    )

    assert locator.kind == "text"
    assert locator.value == "OpenAI"
    assert locator.exact is True


def test_browser_current_tab_requires_active_tab() -> None:
    browser = create_fake_browser_automation()

    with pytest.raises(BrowserAutomationError):
        browser.current_tab()


def test_browser_locator_rejects_empty_value() -> None:
    with pytest.raises(
        ValueError,
        match="locator value cannot be empty",
    ):
        BrowserLocator(
            kind="text",
            value="",
        )
