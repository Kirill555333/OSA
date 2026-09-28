from __future__ import annotations

from osa.browser.automation import (
    BrowserAutomationError,
    BrowserElement,
    BrowserLocator,
    BrowserTab,
    create_fake_browser_automation,
)


def test_locator_rejects_empty_value() -> None:
    try:
        BrowserLocator(
            kind="css",
            value="   ",
        )
    except ValueError as exc:
        assert str(exc) == (
            "Browser locator value cannot be empty."
        )
    else:
        raise AssertionError("Expected ValueError")


def test_open_creates_active_tab() -> None:
    browser = create_fake_browser_automation()

    tab = browser.open("https://example.com")

    assert isinstance(tab, BrowserTab)
    assert tab.url == "https://example.com"
    assert browser.current_tab() == tab
    assert browser.tabs() == (tab,)


def test_find_supports_backend_neutral_locator() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    locator = BrowserLocator(
        kind="role",
        value="button",
    )

    element = browser.find(locator)

    assert isinstance(element, BrowserElement)
    assert element.locator == locator
    assert element.role == "button"
    assert element.visible is True
    assert element.enabled is True


def test_interactive_actions_are_recorded() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    locator = BrowserLocator(
        kind="label",
        value="Search",
    )

    browser.click(locator)
    browser.type_text(
        locator,
        "OSA",
    )
    browser.press(
        locator,
        "Enter",
    )

    assert ("click", locator) in browser.actions

    assert (
        "type_text",
        {
            "locator": locator,
            "text": "OSA",
            "clear": True,
        },
    ) in browser.actions

    assert (
        "press",
        {
            "locator": locator,
            "key": "Enter",
        },
    ) in browser.actions


def test_wait_get_text_and_screenshot() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    locator = BrowserLocator(
        kind="css",
        value="#result",
    )

    browser.wait_for(
        locator,
        state="visible",
        timeout_ms=5000,
    )

    assert browser.get_text(locator) == "fake page text"
    assert browser.screenshot() == b"fake-screenshot"

    assert any(
        action[0] == "wait_for"
        for action in browser.actions
    )


def test_navigation_controls_return_current_tab() -> None:
    browser = create_fake_browser_automation()
    tab = browser.open("https://example.com")

    assert browser.go_back() == tab
    assert browser.go_forward() == tab
    assert browser.reload() == tab

    action_names = [
        action[0]
        for action in browser.actions
    ]

    assert "go_back" in action_names
    assert "go_forward" in action_names
    assert "reload" in action_names


def test_close_active_tab() -> None:
    browser = create_fake_browser_automation()

    tab = browser.open("https://example.com")
    browser.close_tab(tab.tab_id)

    assert browser.tabs() == ()

    try:
        browser.current_tab()
    except BrowserAutomationError as exc:
        assert str(exc) == "No active browser tab."
    else:
        raise AssertionError(
            "Expected BrowserAutomationError"
        )


def test_evaluate_rejects_empty_script() -> None:
    browser = create_fake_browser_automation()

    try:
        browser.evaluate("   ")
    except ValueError as exc:
        assert str(exc) == "script cannot be empty."
    else:
        raise AssertionError("Expected ValueError")


def test_observe_returns_bounded_page_state() -> None:
    browser = create_fake_browser_automation()
    browser.open("https://example.com")

    observation = browser.observe(
        max_text_characters=5,
        max_elements=10,
    )

    assert observation.tab.url == "https://example.com"
    assert observation.text == "fake "
    assert observation.text_truncated is True
    assert observation.elements == ()


def test_observe_rejects_invalid_limits() -> None:
    browser = create_fake_browser_automation()

    try:
        browser.observe(max_text_characters=0)
    except ValueError as exc:
        assert str(exc) == (
            "max_text_characters must be greater than zero."
        )
    else:
        raise AssertionError("Expected ValueError")

    try:
        browser.observe(max_elements=0)
    except ValueError as exc:
        assert str(exc) == (
            "max_elements must be greater than zero."
        )
    else:
        raise AssertionError("Expected ValueError")


def test_health_check() -> None:
    browser = create_fake_browser_automation()

    assert browser.health_check() is True
