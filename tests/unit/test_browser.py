from __future__ import annotations

import socket
from typing import Any

import pytest

from osa.browser import (
    BrowserInterface,
    BrowserPage,
    BrowserResponseError,
    HttpBrowser,
    HttpBrowserConfig,
)


class DummyBrowser(BrowserInterface):
    """Simple browser implementation used only for testing."""

    @property
    def backend_name(self) -> str:
        return "dummy"

    def fetch(self, url: str) -> BrowserPage:
        return BrowserPage(
            url=url,
            title="Test page",
            text="Hello from OSA.",
            status_code=200,
        )

    def health_check(self) -> bool:
        return True


def test_browser_interface_contract() -> None:
    browser = DummyBrowser()

    assert isinstance(
        browser,
        BrowserInterface,
    )
    assert browser.backend_name == "dummy"
    assert browser.health_check() is True


def test_browser_page_contains_normalized_data() -> None:
    page = BrowserPage(
        url="https://example.com",
        title="Example",
        text="Hello",
        status_code=200,
    )

    assert page.url == "https://example.com"
    assert page.title == "Example"
    assert page.text == "Hello"
    assert page.status_code == 200


class FakeHeaders:
    def __init__(
        self,
        charset: str | None = "utf-8",
    ) -> None:
        self._charset = charset

    def get_content_charset(self) -> str | None:
        return self._charset


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str = "https://example.com/final",
        status: int = 200,
        charset: str | None = "utf-8",
    ) -> None:
        self._body = body
        self.status = status
        self.headers = FakeHeaders(charset)
        self._url = url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        pass

    def read(
        self,
        size: int = -1,
    ) -> bytes:
        if size < 0:
            return self._body

        return self._body[:size]

    def geturl(self) -> str:
        return self._url


def test_http_browser_extracts_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = HttpBrowser()

    body = b"""
    <html>
      <head>
        <title> Example   OSA </title>
        <style>.hidden { display: none; }</style>
      </head>
      <body>
        <h1>Hello OSA</h1>
        <p>This is <strong>a test</strong>.</p>
        <script>alert('hidden')</script>
      </body>
    </html>
    """

    def fake_getaddrinfo(
        host: str,
        port: int,
        **kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            )
        ]

    def fake_open(
        http_request: Any,
        timeout: float,
    ) -> FakeResponse:
        return FakeResponse(body)

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )
    monkeypatch.setattr(
        browser._opener,
        "open",
        fake_open,
    )

    page = browser.fetch(
        "https://example.com"
    )

    assert page.url == "https://example.com/final"
    assert page.title == "Example OSA"
    assert page.text == "Hello OSA This is a test."
    assert page.status_code == 200


def test_http_browser_rejects_non_http_url() -> None:
    browser = HttpBrowser()

    with pytest.raises(
        BrowserResponseError,
        match="Only HTTP and HTTPS",
    ):
        browser.fetch(
            "file:///tmp/test.html"
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/test",
        "http://127.0.0.1/test",
        "http://10.0.0.1/test",
        "http://192.168.1.1/test",
    ],
)
def test_http_browser_rejects_local_targets(
    url: str,
) -> None:
    browser = HttpBrowser()

    with pytest.raises(
        BrowserResponseError,
        match="(?i)local network hosts|local or private",
    ):
        browser.fetch(url)


def test_http_browser_rejects_credentials() -> None:
    browser = HttpBrowser()

    with pytest.raises(
        BrowserResponseError,
        match="credentials",
    ):
        browser.fetch(
            "https://user:password@example.com"
        )


def test_http_browser_enforces_response_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    browser = HttpBrowser(
        HttpBrowserConfig(
            max_response_bytes=10,
        )
    )

    body = b"x" * 11

    def fake_getaddrinfo(
        host: str,
        port: int,
        **kwargs: Any,
    ) -> list[tuple[Any, ...]]:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", port),
            )
        ]

    def fake_open(
        http_request: Any,
        timeout: float,
    ) -> FakeResponse:
        return FakeResponse(body)

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )
    monkeypatch.setattr(
        browser._opener,
        "open",
        fake_open,
    )

    with pytest.raises(
        BrowserResponseError,
        match="size limit",
    ):
        browser.fetch(
            "https://example.com"
        )
