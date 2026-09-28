from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, unquote, urljoin, urlsplit


class SearchError(RuntimeError):
    """Base error for search operations."""


class SearchConnectionError(SearchError):
    """Raised when the search backend cannot be reached."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    url: str


class SearchInterface(ABC):
    @abstractmethod
    def search(self, query: str) -> tuple[SearchResult, ...]:
        """Search the public web and return bounded results."""


class _SearchResultParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_result = False
        self._current_href: str | None = None
        self._current_text: list[str] = []
        self.results: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        attributes = dict(attrs)
        href = attributes.get("href")

        if not href:
            return

        classes = set((attributes.get("class") or "").split())
        if "result__a" not in classes:
            return

        self._in_result = True
        self._current_href = href
        self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._in_result:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._in_result:
            return

        href = self._current_href
        title = " ".join("".join(self._current_text).split())

        if href and title:
            self.results.append((title, href))

        self._in_result = False
        self._current_href = None
        self._current_text = []


class DuckDuckGoHtmlSearch(SearchInterface):
    """DuckDuckGo HTML search backed by the existing BrowserInterface."""

    def __init__(
        self,
        browser,
        endpoint: str = "https://html.duckduckgo.com/html/",
        max_results: int = 8,
    ) -> None:
        if max_results < 1:
            raise ValueError("max_results must be at least 1")

        self.browser = browser
        self.endpoint = endpoint
        self.max_results = max_results

    def search(self, query: str) -> tuple[SearchResult, ...]:
        normalized_query = " ".join(query.split()).strip()
        if not normalized_query:
            raise ValueError("Search query must not be empty")

        from urllib.parse import quote_plus

        url = f"{self.endpoint}?q={quote_plus(normalized_query)}"

        try:
            page = self.browser.fetch(url)
        except Exception as exc:
            raise SearchConnectionError(
                f"Search request failed: {exc}"
            ) from exc

        parser = _SearchResultParser()
        parser.feed(page.text)

        results: list[SearchResult] = []
        seen: set[str] = set()

        for title, href in parser.results:
            resolved = self._resolve_url(page.url, href)
            if resolved is None or resolved in seen:
                continue

            seen.add(resolved)
            results.append(
                SearchResult(
                    title=title,
                    url=resolved,
                )
            )

            if len(results) >= self.max_results:
                break

        return tuple(results)

    @staticmethod
    def _resolve_url(base_url: str, href: str) -> str | None:
        absolute = urljoin(base_url, href)
        parsed = urlsplit(absolute)

        if parsed.path.endswith("/l/"):
            query = parse_qs(parsed.query)
            uddg = query.get("uddg")
            if uddg:
                absolute = unquote(uddg[0])
                parsed = urlsplit(absolute)

        if parsed.scheme.lower() not in {"http", "https"}:
            return None

        if not parsed.netloc:
            return None

        host = parsed.hostname
        if not host:
            return None

        lowered_host = host.lower()
        if (
            lowered_host == "duckduckgo.com"
            or lowered_host.endswith(".duckduckgo.com")
        ):
            return None

        return absolute
