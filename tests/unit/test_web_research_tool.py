from dataclasses import dataclass

from osa.research import ResearchSource
from osa.research.loop import ResearchLoopReport
from osa.tools.research import WebResearchTool


@dataclass
class FakeResearchLoop:
    report: ResearchLoopReport
    received_query: str | None = None
    received_follow_ups: tuple[str, ...] = ()

    def run(
        self,
        query: str,
        follow_up_queries: tuple[str, ...] = (),
    ) -> ResearchLoopReport:
        self.received_query = query
        self.received_follow_ups = follow_up_queries
        return self.report


def make_success_report() -> ResearchLoopReport:
    source = ResearchSource(
        url="https://example.com/docs",
        title="Docs",
        text="Useful documentation.",
        status_code=200,
    )

    return ResearchLoopReport(
        original_query="python docs",
        queries=("python docs", "python api"),
        sources=(source,),
        failed_urls=(),
        failed_queries=(),
    )


def test_web_research_tool_returns_context() -> None:
    loop = FakeResearchLoop(
        make_success_report()
    )
    tool = WebResearchTool(loop)

    result = tool.execute(
        {
            "query": "python docs",
            "follow_up_queries": ["python api"],
        }
    )

    assert result.success is True
    assert "Useful documentation." in result.output
    assert result.metadata["source_count"] == 1
    assert loop.received_query == "python docs"
    assert loop.received_follow_ups == ("python api",)


def test_web_research_tool_requires_query() -> None:
    tool = WebResearchTool(
        FakeResearchLoop(make_success_report())
    )

    result = tool.execute({})

    assert result.success is False
    assert result.error == (
        "Argument 'query' must be a string."
    )


def test_web_research_tool_validates_follow_up_queries() -> None:
    tool = WebResearchTool(
        FakeResearchLoop(make_success_report())
    )

    result = tool.execute(
        {
            "query": "python",
            "follow_up_queries": [1],
        }
    )

    assert result.success is False
    assert result.error == (
        "Every follow-up query must be a string."
    )


def test_web_research_tool_limits_follow_up_queries() -> None:
    tool = WebResearchTool(
        FakeResearchLoop(make_success_report())
    )

    result = tool.execute(
        {
            "query": "python",
            "follow_up_queries": [
                "one",
                "two",
                "three",
                "four",
            ],
        }
    )

    assert result.success is False
    assert "at most 3 items" in result.error


def test_web_research_tool_reports_no_sources() -> None:
    report = ResearchLoopReport(
        original_query="missing",
        queries=("missing",),
        sources=(),
        failed_urls=(
            "https://example.com/missing",
        ),
        failed_queries=(),
    )

    result = WebResearchTool(
        FakeResearchLoop(report)
    ).execute(
        {"query": "missing"}
    )

    assert result.success is False
    assert result.error == (
        "Web research returned no usable sources."
    )
    assert result.metadata["source_count"] == 0


def test_web_research_tool_allows_empty_follow_up_list() -> None:
    loop = FakeResearchLoop(
        make_success_report()
    )

    result = WebResearchTool(loop).execute(
        {
            "query": "python docs",
            "follow_up_queries": [],
        }
    )

    assert result.success is True
    assert loop.received_follow_ups == ()
