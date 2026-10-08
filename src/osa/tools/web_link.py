"""Safe opening of web links in the default browser (OSA 1.5.1.3).

The address comes from the language model, so it is treated as untrusted input: only
http and https links with a real host are accepted, non-ASCII characters are
percent-encoded, and the link is handed to the system through ``webbrowser``, never
through a shell. The same code runs on macOS, Windows and Linux.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import re
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit
import webbrowser

from osa.tools.registry import ToolInterface, ToolResult

MAX_URL_LENGTH = 2048

_SCHEME_WITH_SLASHES = re.compile(r"^[a-z][a-z0-9+.\-]*://")
_SCHEME_ONLY = re.compile(r"^[a-z][a-z0-9+.\-]*:")
_HOST_AND_PORT = re.compile(r"^[^/:?#]+:\d+(?:[/?#]|$)")
_WRAPPERS = " \t\r\n\"'<>«»"
_PATH_SAFE = "/%:@!$&'()*+,;=-._~"
_QUERY_SAFE = "=&%+:/?@!$'()*,;-._~"


def normalize_url(raw: str) -> str:
    """Validate an address and return a clean, percent-encoded http(s) URL.

    Raises ValueError with a user-facing reason when the address must not be opened.
    """
    text = str(raw).strip(_WRAPPERS)

    if not text:
        raise ValueError("Адрес ссылки пуст.")

    if len(text) > MAX_URL_LENGTH:
        raise ValueError("Адрес ссылки слишком длинный.")

    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise ValueError("В адресе есть управляющие символы.")

    lowered = text.lower()

    if not lowered.startswith(("http://", "https://")):
        if _SCHEME_WITH_SLASHES.match(lowered):
            raise ValueError("Разрешены только ссылки http и https.")

        if _SCHEME_ONLY.match(lowered) and not _HOST_AND_PORT.match(lowered):
            raise ValueError("Разрешены только ссылки http и https.")

        text = f"https://{text}"

    parts = urlsplit(text)

    if parts.username is not None or parts.password is not None:
        raise ValueError("Ссылки с логином и паролем в адресе не открываются.")

    host = parts.hostname

    if not host or any(char.isspace() for char in host):
        raise ValueError("В адресе нет имени сайта.")

    try:
        port = parts.port
        ascii_host = host.encode("idna").decode("ascii")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Некорректный адрес сайта.") from exc

    if ":" in ascii_host:
        ascii_host = f"[{ascii_host}]"

    netloc = ascii_host if port is None else f"{ascii_host}:{port}"

    return urlunsplit(
        (
            parts.scheme.lower(),
            netloc,
            quote(parts.path, safe=_PATH_SAFE),
            quote(parts.query, safe=_QUERY_SAFE),
            quote(parts.fragment, safe=_QUERY_SAFE),
        )
    )


def _announce(message: str) -> None:
    print(f" [⚡ {message}]", flush=True)


class OpenUrlTool(ToolInterface):
    """Open a web page in the default browser."""

    def __init__(
        self,
        opener: Callable[[str], bool] | None = None,
        announce: Callable[[str], None] | None = None,
    ) -> None:
        self._opener = opener or self._open_in_default_browser
        self._announce = announce or _announce

    @staticmethod
    def _open_in_default_browser(url: str) -> bool:
        return bool(webbrowser.open(url, new=2))

    @property
    def name(self) -> str:
        return "open_url"

    @property
    def description(self) -> str:
        return (
            "Open a web page in the default browser. Pass a full http(s) address. For a "
            "search, pass a ready search link, for example "
            "https://www.youtube.com/results?search_query=... or "
            "https://www.google.com/search?q=..."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full http(s) address to open.",
                },
            },
            "required": ["url"],
        }

    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        try:
            url = normalize_url(str(arguments.get("url", "")))
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        self._announce(f"Открытие ссылки: {url}")

        try:
            opened = self._opener(url)
        except Exception as exc:
            return ToolResult(success=False, error=f"Не удалось открыть браузер: {exc}")

        if not opened:
            return ToolResult(
                success=False,
                error=(
                    "Не удалось открыть браузер "
                    "(нет браузера по умолчанию или графической среды)."
                ),
            )

        return ToolResult(
            success=True,
            output=f"Ссылка передана браузеру по умолчанию: {url}",
            metadata={"url": url, "verified": False},
        )
