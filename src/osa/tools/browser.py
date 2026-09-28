"""Browser tool for OSA."""

from __future__ import annotations

from typing import Any, Mapping

from osa.browser import (
    BrowserError,
    HttpBrowser,
)
from osa.tools.registry import (
    ToolInterface,
    ToolResult,
)


class BrowserFetchTool(ToolInterface):
    """Fetch a public web page and return readable text."""

    def __init__(
        self,
        browser: HttpBrowser,
        *,
        max_text_characters: int = 30_000,
    ) -> None:
        if max_text_characters <= 0:
            raise ValueError(
                "max_text_characters must be greater than zero."
            )

        self._browser = browser
        self._max_text_characters = max_text_characters

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "browser_fetch"

    @property
    def description(self) -> str:
        """Return a human-readable tool description."""
        return (
            "Fetch a public HTTP or HTTPS web page and return "
            "its title and readable text."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Public HTTP or HTTPS URL to fetch."
                    ),
                }
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Fetch a web page."""
        url = arguments.get("url")

        if not isinstance(url, str):
            return ToolResult(
                success=False,
                error="Argument 'url' must be a string.",
            )

        try:
            page = self._browser.fetch(url)
        except BrowserError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        text = page.text[: self._max_text_characters]

        output = (
            f"URL: {page.url}\n"
            f"Title: {page.title or '(untitled)'}\n"
            f"Status: {page.status_code}\n"
            f"Text:\n{text}"
        )

        if len(page.text) > self._max_text_characters:
            output += (
                "\n\n[Page text truncated by OSA.]"
            )

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "url": page.url,
                "status_code": page.status_code,
                "title": page.title,
            },
        )
