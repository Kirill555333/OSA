from __future__ import annotations

from dataclasses import dataclass

from osa.research.interface import ResearchReport, ResearchSource
from osa.research.researcher import WebResearcher


class ResearchLoopError(RuntimeError):
    """Raised when the research loop cannot proceed safely."""


@dataclass(frozen=True, slots=True)
class ResearchLoopConfig:
    max_queries: int = 4
    max_sources: int = 8
    max_context_chars: int = 30_000

    def __post_init__(self) -> None:
        if self.max_queries < 1:
            raise ValueError("max_queries must be at least 1")

        if self.max_sources < 1:
            raise ValueError("max_sources must be at least 1")

        if self.max_context_chars < 1:
            raise ValueError("max_context_chars must be at least 1")


@dataclass(frozen=True, slots=True)
class ResearchLoopReport:
    original_query: str
    queries: tuple[str, ...]
    sources: tuple[ResearchSource, ...]
    failed_urls: tuple[str, ...]
    failed_queries: tuple[str, ...]

    @property
    def success(self) -> bool:
        return bool(self.sources)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def as_context(self, max_chars: int | None = None) -> str:
        limit = max_chars or 30_000

        if limit <= 0:
            return ""

        parts: list[str] = []
        used = 0

        for index, source in enumerate(self.sources, start=1):
            section = (
                f"[Source {index}]\n"
                f"Title: {source.title}\n"
                f"URL: {source.url}\n"
                f"{source.text.strip()}"
            ).strip()

            if not section:
                continue

            remaining = limit - used
            if remaining <= 0:
                break

            if len(section) > remaining:
                section = section[:remaining].rstrip()

            parts.append(section)
            used += len(section) + 2

        return "\n\n".join(parts)


class ResearchLoop:
    """Runs a bounded sequence of web-research queries."""

    def __init__(
        self,
        researcher: WebResearcher,
        config: ResearchLoopConfig | None = None,
    ) -> None:
        self.researcher = researcher
        self.config = config or ResearchLoopConfig()

    def run(
        self,
        query: str,
        follow_up_queries: tuple[str, ...] = (),
    ) -> ResearchLoopReport:
        original_query = self._normalize_query(query)

        candidates = (original_query, *follow_up_queries)
        normalized_queries: list[str] = []
        seen_queries: set[str] = set()

        for candidate in candidates:
            normalized = self._normalize_query(candidate)
            key = normalized.casefold()

            if key in seen_queries:
                continue

            seen_queries.add(key)
            normalized_queries.append(normalized)

            if len(normalized_queries) >= self.config.max_queries:
                break

        sources: list[ResearchSource] = []
        seen_urls: set[str] = set()
        failed_urls: list[str] = []
        failed_queries: list[str] = []

        for current_query in normalized_queries:
            try:
                report = self.researcher.research(current_query)
            except Exception:
                failed_queries.append(current_query)
                continue

            failed_urls.extend(report.failed_urls)

            for source in report.sources:
                key = self._url_key(source.url)

                if key in seen_urls:
                    continue

                seen_urls.add(key)
                sources.append(source)

                if len(sources) >= self.config.max_sources:
                    break

            if len(sources) >= self.config.max_sources:
                break

        return ResearchLoopReport(
            original_query=original_query,
            queries=tuple(normalized_queries),
            sources=tuple(sources),
            failed_urls=tuple(dict.fromkeys(failed_urls)),
            failed_queries=tuple(failed_queries),
        )

    def _normalize_query(self, query: str) -> str:
        normalized = " ".join(query.split()).strip()

        if not normalized:
            raise ResearchLoopError("Research query must not be empty")

        return normalized

    @staticmethod
    def _url_key(url: str) -> str:
        return url.strip().rstrip("/").casefold()
