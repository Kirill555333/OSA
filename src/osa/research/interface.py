from __future__ import annotations

from dataclasses import dataclass


class ResearchError(RuntimeError):
    """Base error for web research operations."""


class ResearchSearchError(ResearchError):
    """Raised when the web search step fails."""


class ResearchSourceError(ResearchError):
    """Raised when a research source cannot be processed."""


@dataclass(frozen=True, slots=True)
class ResearchSource:
    url: str
    title: str
    text: str
    status_code: int


@dataclass(frozen=True, slots=True)
class ResearchReport:
    query: str
    sources: tuple[ResearchSource, ...]
    failed_urls: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        return bool(self.sources)

    @property
    def source_count(self) -> int:
        return len(self.sources)

    def as_context(self, max_chars: int = 30_000) -> str:
        if max_chars <= 0:
            return ""

        sections: list[str] = []
        remaining = max_chars

        for index, source in enumerate(self.sources, start=1):
            header = (
                f"[Source {index}]\n"
                f"Title: {source.title}\n"
                f"URL: {source.url}\n"
            )
            body = source.text.strip()

            section = f"{header}{body}".strip()

            if len(section) > remaining:
                section = section[:remaining].rstrip()

            if not section:
                break

            sections.append(section)
            remaining -= len(section) + 2

            if remaining <= 0:
                break

        return "\n\n".join(sections)
