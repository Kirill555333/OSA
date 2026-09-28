"""HTTP browser backend for OSA."""

from __future__ import annotations

import ipaddress
import re
import socket
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib import error, request
from urllib.parse import urlsplit

from osa.browser.interface import (
    BrowserConnectionError,
    BrowserInterface,
    BrowserPage,
    BrowserResponseError,
)


@dataclass(frozen=True, slots=True)
class HttpBrowserConfig:
    """Configuration for the HTTP browser backend."""

    timeout: float = 20.0
    max_response_bytes: int = 2_000_000
    user_agent: str = "OSA/0.2.12"

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError(
                "timeout must be greater than zero."
            )

        if self.max_response_bytes <= 0:
            raise ValueError(
                "max_response_bytes must be greater than zero."
            )


class _HtmlTextExtractor(HTMLParser):
    """Extract page title and visible text from HTML."""

    _IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
    }

    def __init__(self) -> None:
        super().__init__(
            convert_charrefs=True
        )
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []
        self._inside_title = False
        self._ignored_depth = 0

    @property
    def title(self) -> str:
        """Return the normalized page title."""
        return " ".join(
            " ".join(self._title_parts).split()
        )

    @property
    def text(self) -> str:
        """Return normalized visible page text."""
        text = " ".join(
            " ".join(self._text_parts).split()
        )

        return re.sub(
            r"\s+([,.;:!?])",
            r"\1",
            text,
        )

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_tag = tag.lower()

        if normalized_tag == "title":
            self._inside_title = True

        if normalized_tag in self._IGNORED_TAGS:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        normalized_tag = tag.lower()

        if normalized_tag == "title":
            self._inside_title = False

        if normalized_tag in self._IGNORED_TAGS:
            self._ignored_depth = max(
                0,
                self._ignored_depth - 1,
            )

    def handle_data(self, data: str) -> None:
        if self._ignored_depth > 0:
            return

        if not data.strip():
            return

        if self._inside_title:
            self._title_parts.append(data)
        else:
            self._text_parts.append(data)


class _SafeRedirectHandler(request.HTTPRedirectHandler):
    """Validate every redirect before following it."""

    def __init__(
        self,
        validator: Callable[[str], str],
    ) -> None:
        self._validator = validator

    def redirect_request(
        self,
        req: request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> request.Request | None:
        validated_url = self._validator(newurl)

        return super().redirect_request(
            req,
            fp,
            code,
            msg,
            headers,
            validated_url,
        )


class HttpBrowser(BrowserInterface):
    """Fetch HTTP(S) pages and normalize their HTML content safely."""

    def __init__(
        self,
        config: HttpBrowserConfig | None = None,
    ) -> None:
        self._config = config or HttpBrowserConfig()

        self._opener = request.build_opener(
            _SafeRedirectHandler(
                self._validate_url
            )
        )

    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return "http"

    def health_check(self) -> bool:
        """Return True when the stateless HTTP backend is configured."""
        return True

    def fetch(self, url: str) -> BrowserPage:
        """Fetch one HTTP(S) page."""
        normalized_url = self._validate_url(url)

        http_request = request.Request(
            normalized_url,
            headers={
                "User-Agent": self._config.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
            method="GET",
        )

        try:
            with self._opener.open(
                http_request,
                timeout=self._config.timeout,
            ) as response:
                body = response.read(
                    self._config.max_response_bytes + 1
                )

                if len(body) > self._config.max_response_bytes:
                    raise BrowserResponseError(
                        "Page response exceeded the configured size limit."
                    )

                status_code = response.status
                final_url = response.geturl()
                charset = self._response_charset(response)

        except BrowserResponseError:
            raise
        except error.HTTPError as exc:
            raise BrowserResponseError(
                f"Web server returned HTTP {exc.code}."
            ) from exc
        except (
            error.URLError,
            TimeoutError,
        ) as exc:
            raise BrowserConnectionError(
                f"Could not fetch page: {exc}"
            ) from exc

        try:
            html = body.decode(
                charset or "utf-8",
                errors="replace",
            )
        except LookupError as exc:
            raise BrowserResponseError(
                f"Unknown page charset: {charset}"
            ) from exc

        extractor = _HtmlTextExtractor()

        try:
            extractor.feed(html)
            extractor.close()
        except Exception as exc:
            raise BrowserResponseError(
                "Could not parse HTML page."
            ) from exc

        return BrowserPage(
            url=final_url,
            title=extractor.title,
            text=extractor.text,
            status_code=status_code,
        )

    @classmethod
    def _validate_url(cls, url: str) -> str:
        """Validate a URL and reject local or private network targets."""
        normalized = url.strip()

        if not normalized:
            raise BrowserResponseError(
                "URL cannot be empty."
            )

        parsed = urlsplit(normalized)

        if parsed.scheme not in {
            "http",
            "https",
        }:
            raise BrowserResponseError(
                "Only HTTP and HTTPS URLs are supported."
            )

        if not parsed.hostname:
            raise BrowserResponseError(
                "URL must contain a hostname."
            )

        if (
            parsed.username is not None
            or parsed.password is not None
        ):
            raise BrowserResponseError(
                "URLs containing credentials are not supported."
            )

        hostname = parsed.hostname.rstrip(".").lower()

        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
        ):
            raise BrowserResponseError(
                "Local network hosts are not allowed."
            )

        try:
            port = parsed.port or (
                443
                if parsed.scheme == "https"
                else 80
            )
        except ValueError as exc:
            raise BrowserResponseError(
                "URL contains an invalid port."
            ) from exc

        try:
            addresses = socket.getaddrinfo(
                hostname,
                port,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise BrowserConnectionError(
                f"Could not resolve host: {hostname}"
            ) from exc

        for address in addresses:
            ip = ipaddress.ip_address(
                address[4][0]
            )

            if cls._is_blocked_ip(ip):
                raise BrowserResponseError(
                    "Access to local or private network targets "
                    "is not allowed."
                )

        return normalized

    @staticmethod
    def _is_blocked_ip(
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    ) -> bool:
        """Return whether an IP is unsuitable for external web access."""
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        )

    @staticmethod
    def _response_charset(response: object) -> str | None:
        headers = getattr(
            response,
            "headers",
            None,
        )

        if headers is None:
            return None

        get_charset = getattr(
            headers,
            "get_content_charset",
            None,
        )

        if not callable(get_charset):
            return None

        return get_charset()
