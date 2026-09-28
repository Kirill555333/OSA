from dataclasses import dataclass

import pytest

from osa.browser.interface import BrowserError, BrowserPage
from osa.research import (
    DuckDuckGoHtmlSearch,
    ResearchSearchError,
    SearchError,
    SearchResult,
    WebResearchConfig,
    WebResearcher,
)


@dataclass
class FakeBrowser:
    pages: dict[str, BrowserPage]
    failures: set[str]

    def fetch(self, url: str) -> BrowserPage:
        if url in self.failures:
            raise BrowserError(f"failed: {url}")

        return self.pages[url]


def test_duckduckgo_search_extracts_results() -> None:
    search_url = (
        "https://html.duckduckgo.com/html/?q=python+testing"
    )
    target = "https://example.com/article"

    html = f"""
    <html>
      <a class="result__a" href="{target}">Example Article</a>
      <a class="result__a" href="{target}">Duplicate</a>
      <a class="result__a" href="https://duckduckgo.com/about">Internal</a>
    </html>
    """

    browser = FakeBrowser(
        pages={
            search_url: BrowserPage(
                url=search_url,
                title="Search",
                text=html,
                status_code=200,
            ),
        },
        failures=set(),
    )

    search = DuckDuckGoHtmlSearch(browser)

    results = search.search("python testing")

    assert results == (
        SearchResult(
            title="Example Article",
            url=target,
        ),
    )


def test_duckduckgo_search_resolves_redirect() -> None:
    search_url = (
        "https://html.duckduckgo.com/html/?q=python"
    )
    target = "https://example.com/docs"

    redirect = (
        "https://duckduckgo.com/l/"
        "?uddg=https%3A%2F%2Fexample.com%2Fdocs"
    )

    browser = FakeBrowser(
        pages={
            search_url: BrowserPage(
                url=search_url,
                title="Search",
                text=(
                    f'<a class="result__a" href="{redirect}">'
                    "Docs</a>"
                ),
                status_code=200,
            ),
        },
        failures=set(),
    )

    results = DuckDuckGoHtmlSearch(browser).search("python")

    assert results[0].url == target


def test_research_uses_injected_search_backend() -> None:
    search_url = "https://example.com/article"

    browser = FakeBrowser(
        pages={
            search_url: BrowserPage(
                url=search_url,
                title="Article",
                text="Useful contents.",
                status_code=200,
            ),
        },
        failures=set(),
    )

    class FakeSearch:
        def search(self, query: str) -> tuple[SearchResult, ...]:
            assert query == "python testing"
            return (
                SearchResult(
                    title="Search result",
                    url=search_url,
                ),
            )

    report = WebResearcher(
        browser=browser,
        search=FakeSearch(),
    ).research("python testing")

    assert report.success
    assert report.source_count == 1
    assert report.sources[0].title == "Article"
    assert report.sources[0].text == "Useful contents."


def test_research_continues_when_one_source_fails() -> None:
    bad_url = "https://example.com/bad"
    good_url = "https://example.org/good"

    browser = FakeBrowser(
        pages={
            good_url: BrowserPage(
                url=good_url,
                title="Good",
                text="Useful source.",
                status_code=200,
            ),
        },
        failures={bad_url},
    )

    class FakeSearch:
        def search(self, query: str) -> tuple[SearchResult, ...]:
            return (
                SearchResult("Bad", bad_url),
                SearchResult("Good", good_url),
            )

    report = WebResearcher(
        browser=browser,
        search=FakeSearch(),
    ).research("python testing")

    assert report.success
    assert len(report.sources) == 1
    assert report.sources[0].url == good_url
    assert report.failed_urls == (bad_url,)


def test_research_raises_for_search_failure() -> None:
    class FailingSearch:
        def search(self, query: str):
            raise SearchError("backend unavailable")

    browser = FakeBrowser(pages={}, failures=set())

    researcher = WebResearcher(
        browser=browser,
        search=FailingSearch(),
    )

    with pytest.raises(ResearchSearchError):
        researcher.research("python testing")


def test_research_rejects_empty_query() -> None:
    browser = FakeBrowser(pages={}, failures=set())

    class FakeSearch:
        def search(self, query: str):
            return ()

    researcher = WebResearcher(
        browser=browser,
        search=FakeSearch(),
    )

    with pytest.raises(ValueError):
        researcher.research("   ")


def test_research_respects_source_limit() -> None:
    urls = [
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    ]

    pages = {
        url: BrowserPage(
            url=url,
            title=url,
            text="content",
            status_code=200,
        )
        for url in urls
    }

    browser = FakeBrowser(pages=pages, failures=set())

    class FakeSearch:
        def search(self, query: str):
            return tuple(
                SearchResult(url, url)
                for url in urls
            )

    report = WebResearcher(
        browser=browser,
        search=FakeSearch(),
        config=WebResearchConfig(max_sources=2),
    ).research("python testing")

    assert report.source_count == 2


def test_report_context_is_bounded() -> None:
    source_url = "https://example.com/article"

    browser = FakeBrowser(
        pages={
            source_url: BrowserPage(
                url=source_url,
                title="Article",
                text="A" * 10_000,
                status_code=200,
            ),
        },
        failures=set(),
    )

    class FakeSearch:
        def search(self, query: str):
            return (SearchResult("Article", source_url),)

    report = WebResearcher(
        browser=browser,
        search=FakeSearch(),
        config=WebResearchConfig(max_source_chars=10_000),
    ).research("python testing")

    context = report.as_context(max_chars=500)

    assert len(context) <= 500
    assert "Article" in context
    assert source_url in context
