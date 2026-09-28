from __future__ import annotations

from typing import Any, Mapping

from osa.research.loop import ResearchLoop, ResearchLoopError
from osa.tools.registry import ToolInterface, ToolResult


class WebResearchTool(ToolInterface):
    """Run bounded public-web research and return source context."""

    def __init__(self, research_loop: ResearchLoop) -> None:
        self._research_loop = research_loop

    @property
    def name(self) -> str:
        return "web_research"

    @property
    def description(self) -> str:
        return (
            "Research a public-web topic using search and source fetching. "
            "Use this for questions requiring current or external web information."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The main web research question or search query."
                    ),
                },
                "follow_up_queries": {
                    "type": "array",
                    "description": (
                        "Optional additional search queries to broaden the research."
                    ),
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        query = arguments.get("query")
        follow_up_queries = arguments.get(
            "follow_up_queries",
            (),
        )

        if not isinstance(query, str):
            return ToolResult(
                success=False,
                error="Argument 'query' must be a string.",
            )

        if not isinstance(
            follow_up_queries,
            (list, tuple),
        ):
            return ToolResult(
                success=False,
                error="Argument 'follow_up_queries' must be an array.",
            )

        if len(follow_up_queries) > 3:
            return ToolResult(
                success=False,
                error=(
                    "Argument 'follow_up_queries' must contain "
                    "at most 3 items."
                ),
            )

        if any(
            not isinstance(item, str)
            for item in follow_up_queries
        ):
            return ToolResult(
                success=False,
                error="Every follow-up query must be a string.",
            )

        try:
            report = self._research_loop.run(
                query,
                tuple(follow_up_queries),
            )
        except ResearchLoopError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        context = report.as_context()

        metadata = {
            "query": report.original_query,
            "queries": report.queries,
            "source_count": report.source_count,
            "failed_queries": report.failed_queries,
            "failed_urls": report.failed_urls,
        }

        if not report.success:
            return ToolResult(
                success=False,
                output=context,
                error="Web research returned no usable sources.",
                metadata=metadata,
            )

        return ToolResult(
            success=True,
            output=context,
            metadata=metadata,
        )
