from __future__ import annotations

from osa.browser import BrowserPage
from osa.tools import BrowserFetchTool


class FakeBrowser:
    """Fake browser backend for tool tests."""

    def __init__(
        self,
        page: BrowserPage,
    ) -> None:
        self._page = page

    def fetch(self, url: str) -> BrowserPage:
        return self._page


def test_browser_fetch_tool_returns_page_data() -> None:
    browser = FakeBrowser(
        BrowserPage(
            url="https://example.com/",
            title="Example",
            text="Hello from the web.",
            status_code=200,
        )
    )

    tool = BrowserFetchTool(browser)

    result = tool.execute(
        {"url": "https://example.com"}
    )

    assert result.success is True
    assert "Title: Example" in result.output
    assert "Status: 200" in result.output
    assert "Hello from the web." in result.output


def test_browser_fetch_tool_rejects_invalid_url_argument() -> None:
    browser = FakeBrowser(
        BrowserPage(
            url="https://example.com/",
            title="Example",
            text="Hello",
            status_code=200,
        )
    )

    tool = BrowserFetchTool(browser)

    result = tool.execute(
        {"url": 123}
    )

    assert result.success is False
    assert "must be a string" in result.error


def test_browser_fetch_tool_truncates_text() -> None:
    browser = FakeBrowser(
        BrowserPage(
            url="https://example.com/",
            title="Example",
            text="abcdefghij",
            status_code=200,
        )
    )

    tool = BrowserFetchTool(
        browser,
        max_text_characters=5,
    )

    result = tool.execute(
        {"url": "https://example.com"}
    )

    assert result.success is True
    assert "Text:\nabcde" in result.output
    assert "[Page text truncated by OSA.]" in result.output
