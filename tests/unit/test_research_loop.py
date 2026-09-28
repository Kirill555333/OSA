from dataclasses import dataclass

from osa.research import (
    ResearchLoop,
    ResearchLoopConfig,
    ResearchSource,
)


@dataclass
class FakeResearcher:
    reports: dict[str, object]

    def research(self, query: str):
        value = self.reports[query]

        if isinstance(value, Exception):
            raise value

        return value


def make_report(
    query: str,
    sources: tuple[ResearchSource, ...],
):
    from osa.research.interface import ResearchReport

    return ResearchReport(
        query=query,
        sources=sources,
        failed_urls=(),
    )


def test_research_loop_runs_multiple_queries() -> None:
    source_one = ResearchSource(
        url="https://example.com/a",
        title="A",
        text="Alpha",
        status_code=200,
    )
    source_two = ResearchSource(
        url="https://example.com/b",
        title="B",
        text="Beta",
        status_code=200,
    )

    researcher = FakeResearcher(
        {
            "python testing": make_report(
                "python testing",
                (source_one,),
            ),
            "pytest fixtures": make_report(
                "pytest fixtures",
                (source_two,),
            ),
        }
    )

    report = ResearchLoop(researcher).run(
        "python testing",
        ("pytest fixtures",),
    )

    assert report.success
    assert report.queries == (
        "python testing",
        "pytest fixtures",
    )
    assert report.source_count == 2


def test_research_loop_deduplicates_queries_and_urls() -> None:
    source = ResearchSource(
        url="https://example.com/a/",
        title="A",
        text="Alpha",
        status_code=200,
    )

    researcher = FakeResearcher(
        {
            "python": make_report(
                "python",
                (source,),
            ),
        }
    )

    report = ResearchLoop(researcher).run(
        "python",
        ("python", "PYTHON"),
    )

    assert report.queries == ("python",)
    assert report.source_count == 1


def test_research_loop_respects_limits() -> None:
    sources = tuple(
        ResearchSource(
            url=f"https://example.com/{index}",
            title=str(index),
            text="content",
            status_code=200,
        )
        for index in range(5)
    )

    researcher = FakeResearcher(
        {
            "first": make_report("first", sources),
            "second": make_report("second", sources),
            "third": make_report("third", sources),
        }
    )

    loop = ResearchLoop(
        researcher,
        ResearchLoopConfig(
            max_queries=2,
            max_sources=3,
        ),
    )

    report = loop.run(
        "first",
        ("second", "third"),
    )

    assert report.queries == ("first", "second")
    assert report.source_count == 3


def test_research_loop_keeps_going_after_query_failure() -> None:
    source = ResearchSource(
        url="https://example.com/good",
        title="Good",
        text="Useful",
        status_code=200,
    )

    researcher = FakeResearcher(
        {
            "broken": RuntimeError("temporary"),
            "good": make_report(
                "good",
                (source,),
            ),
        }
    )

    report = ResearchLoop(researcher).run(
        "broken",
        ("good",),
    )

    assert report.success
    assert report.failed_queries == ("broken",)
    assert report.sources[0].url == "https://example.com/good"


def test_research_loop_context_is_bounded() -> None:
    source = ResearchSource(
        url="https://example.com/a",
        title="Article",
        text="A" * 10_000,
        status_code=200,
    )

    researcher = FakeResearcher(
        {
            "query": make_report(
                "query",
                (source,),
            ),
        }
    )

    report = ResearchLoop(researcher).run("query")

    context = report.as_context(max_chars=500)

    assert len(context) <= 500
    assert "Article" in context
