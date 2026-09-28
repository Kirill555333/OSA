from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from osa.browser.interface import BrowserError, BrowserInterface, BrowserPage
from osa.research.interface import (
    ResearchReport,
    ResearchSearchError,
    ResearchSource,
)
from osa.research.search import SearchError, SearchInterface


@dataclass(frozen=True, slots=True)
class WebResearchConfig:
    max_sources: int = 4
    max_source_chars: int = 12_000

    def __post_init__(self) -> None:
        if self.max_sources < 1:
            raise ValueError("max_sources must be at least 1")

        if self.max_source_chars < 1:
            raise ValueError("max_source_chars must be at least 1")


class WebResearcher:
    """Searches the public web and fetches a bounded set of sources."""

    def __init__(
        self,
        browser: BrowserInterface,
        search: SearchInterface,
        config: WebResearchConfig | None = None,
    ) -> None:
        self.browser = browser
        self.search = search
        self.config = config or WebResearchConfig()

    def research(self, query: str) -> ResearchReport:
        normalized_query = self._normalize_query(query)

        try:
            search_results = self.search.search(normalized_query)
        except SearchError as exc:
            raise ResearchSearchError(
                f"Web search failed: {exc}"
            ) from exc

        sources: list[ResearchSource] = []
        failed_urls: list[str] = []

        for result in search_results[: self.config.max_sources]:
            try:
                page = self.browser.fetch(result.url)
                sources.append(self._to_source(page, result.title))
            except BrowserError:
                failed_urls.append(result.url)

        return ResearchReport(
            query=normalized_query,
            sources=tuple(sources),
            failed_urls=tuple(failed_urls),
        )

    def _to_source(
        self,
        page: BrowserPage,
        fallback_title: str,
    ) -> ResearchSource:
        text = page.text.strip()

        if len(text) > self.config.max_source_chars:
            text = text[: self.config.max_source_chars].rstrip()

        return ResearchSource(
            url=page.url,
            title=page.title.strip() or fallback_title,
            text=text,
            status_code=page.status_code,
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized = " ".join(query.split()).strip()

        if not normalized:
            raise ValueError("Research query must not be empty")

        return normalized


def extract_research_urls(
    texts: Iterable[str],
) -> tuple[str, ...]:
    """Extract unique HTTP(S) URLs from arbitrary text fragments."""
    from osa.research.researcher import WebResearcher

    seen: set[str] = set()
    urls: list[str] = []

    for text in texts:
        for token in text.split():
            cleaned = token.strip(".,;:!?()[]{}<>\"'")
            if not (
                cleaned.startswith("http://")
                or cleaned.startswith("https://")
            ):
                continue

            if cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)

    return tuple(urls)
