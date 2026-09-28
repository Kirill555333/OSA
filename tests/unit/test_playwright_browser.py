from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.browser import (
    BrowserAutomationActionError,
    BrowserAutomationError,
    BrowserLocator,
    PlaywrightBrowser,
    PlaywrightBrowserConfig,
)


class FakeLocator:
    def __init__(
        self,
        *,
        count: int = 1,
        text: str = "fake text",
        role: str | None = "button",
        tag_name: str = "button",
    ) -> None:
        self._count = count
        self._text = text
        self._role = role
        self._tag_name = tag_name
        self.first = self

    def count(self) -> int:
        return self._count

    def inner_text(self) -> str:
        return self._text

    def get_attribute(self, name: str):
        if name == "role":
            return self._role
        return None

    def evaluate(self, script: str) -> str:
        assert "tagName" in script
        return self._tag_name

    def is_visible(self) -> bool:
        return True

    def is_enabled(self) -> bool:
        return True

    def click(self, **kwargs) -> None:
        return None

    def fill(self, text: str, **kwargs) -> None:
        return None

    def press_sequentially(self, text: str, **kwargs) -> None:
        return None

    def press(self, key: str, **kwargs) -> None:
        return None

    def wait_for(self, **kwargs) -> None:
        return None


class FakePage:
    def __init__(
        self,
        url: str = "about:blank",
    ) -> None:
        self.url = url
        self._title = "Fake page"
        self._closed = False
        self.locator_object = FakeLocator()

    def title(self) -> str:
        return self._title

    def goto(self, url: str, **kwargs) -> None:
        self.url = url

    def close(self) -> None:
        self._closed = True

    def locator(self, value: str) -> FakeLocator:
        if value == "ambiguous":
            return FakeLocator(count=2)
        return self.locator_object

    def get_by_role(self, value: str, **kwargs) -> FakeLocator:
        return self.locator_object

    def get_by_text(self, value: str, **kwargs) -> FakeLocator:
        return self.locator_object

    def get_by_label(self, value: str, **kwargs) -> FakeLocator:
        return self.locator_object

    def get_by_placeholder(self, value: str, **kwargs) -> FakeLocator:
        return self.locator_object

    def get_by_test_id(self, value: str) -> FakeLocator:
        return self.locator_object

    def go_back(self, **kwargs) -> None:
        return None

    def go_forward(self, **kwargs) -> None:
        return None

    def reload(self, **kwargs) -> None:
        return None

    def screenshot(self, **kwargs) -> bytes:
        return b"fake-screenshot"

    def evaluate(self, script: str) -> object:
        return {"script": script}


class FakeContext:
    def __init__(self) -> None:
        self._pages: list[FakePage] = []

    @property
    def pages(self) -> list[FakePage]:
        return [
            page
            for page in self._pages
            if not page._closed
        ]

    def new_page(self) -> FakePage:
        page = FakePage()
        self._pages.append(page)
        return page

    def set_default_navigation_timeout(self, timeout: int) -> None:
        return None

    def set_default_timeout(self, timeout: int) -> None:
        return None

    def close(self) -> None:
        return None


class FakeBrowser:
    def __init__(self) -> None:
        self.context = FakeContext()

    def new_context(self) -> FakeContext:
        return self.context

    def close(self) -> None:
        return None


class FakeBrowserType:
    def __init__(self) -> None:
        self.browser = FakeBrowser()

    def launch(self, **kwargs) -> FakeBrowser:
        return self.browser


class FakePlaywright:
    def __init__(self) -> None:
        self.chromium = FakeBrowserType()
        self.firefox = FakeBrowserType()
        self.webkit = FakeBrowserType()
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


@dataclass
class FakeDriver:
    instance: FakePlaywright

    def start(self) -> FakePlaywright:
        return self.instance


def make_browser() -> tuple[PlaywrightBrowser, FakePlaywright]:
    playwright = FakePlaywright()

    browser = PlaywrightBrowser(
        driver_factory=lambda: FakeDriver(playwright),
    )

    return browser, playwright


def test_open_and_current_tab() -> None:
    browser, _ = make_browser()

    tab = browser.open("https://example.com")

    assert tab.url == "https://example.com"
    assert tab.title == "Fake page"
    assert browser.current_tab() == tab


def test_find_uses_backend_neutral_locator() -> None:
    browser, _ = make_browser()

    browser.open("https://example.com")

    locator = BrowserLocator(
        kind="role",
        value="button",
        exact=True,
    )

    element = browser.find(locator)

    assert element is not None
    assert element.locator == locator
    assert element.text == "fake text"
    assert element.role == "button"
    assert element.tag_name == "button"
    assert element.visible is True
    assert element.enabled is True


def test_all_locator_kinds_are_supported() -> None:
    browser, _ = make_browser()

    browser.open("https://example.com")

    for kind in (
        "css",
        "role",
        "text",
        "label",
        "placeholder",
        "test_id",
    ):
        locator = BrowserLocator(
            kind=kind,
            value="value",
        )

        assert browser.find(locator) is not None


def test_click_type_press_wait_and_text() -> None:
    browser, _ = make_browser()

    browser.open("https://example.com")

    locator = BrowserLocator(
        kind="css",
        value="#search",
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
    browser.wait_for(locator)

    assert browser.get_text(locator) == "fake text"


def test_navigation_and_screenshot() -> None:
    browser, _ = make_browser()

    browser.open("https://example.com")

    assert browser.go_back().url == "https://example.com"
    assert browser.go_forward().url == "https://example.com"
    assert browser.reload().url == "https://example.com"
    assert browser.screenshot() == b"fake-screenshot"


def test_evaluate() -> None:
    browser, _ = make_browser()

    browser.open("https://example.com")

    result = browser.evaluate(
        "document.title",
    )

    assert result == {
        "script": "document.title"
    }


def test_ambiguous_action_is_rejected() -> None:
    browser, _ = make_browser()

    browser.open("https://example.com")

    locator = BrowserLocator(
        kind="css",
        value="ambiguous",
    )

    with pytest.raises(
        BrowserAutomationActionError,
        match="ambiguous",
    ):
        browser.click(locator)


def test_close_tab() -> None:
    browser, _ = make_browser()

    tab = browser.open("https://example.com")

    browser.close_tab(tab.tab_id)

    assert browser.tabs() == ()

    with pytest.raises(
        BrowserAutomationError,
        match="No active browser tab.",
    ):
        browser.get_text()


def test_config_validation() -> None:
    with pytest.raises(
        ValueError,
        match="browser_name",
    ):
        PlaywrightBrowserConfig(
            browser_name="invalid",
        )

    with pytest.raises(
        ValueError,
        match="navigation_timeout_ms",
    ):
        PlaywrightBrowserConfig(
            navigation_timeout_ms=0,
        )


def test_health_check() -> None:
    browser, _ = make_browser()

    assert browser.health_check() is True

    browser.close()

    assert browser._browser is None
