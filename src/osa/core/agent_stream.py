"""Normalize model streaming events for the unified Agent runtime."""

from __future__ import annotations

from osa.core.agent_runtime import (
    AgentRoundResult,
    AgentRoundRuntime,
    AgentRuntimeError,
)
from osa.models import ModelStreamEvent, ToolCall


class AgentStreamAccumulator:
    """Accumulate normalized model stream events into one agent round."""

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._tool_calls: list[ToolCall] = []
        self._finish_reason: str | None = None

    @property
    def content(self) -> str:
        """Return all accumulated text."""
        return "".join(self._content_parts)

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return all accumulated tool calls in input order."""
        return tuple(self._tool_calls)

    @property
    def finish_reason(self) -> str | None:
        """Return the latest non-null finish reason."""
        return self._finish_reason

    def add(
        self,
        event: ModelStreamEvent,
    ) -> None:
        """Accumulate one normalized model stream event."""
        if not isinstance(
            event,
            ModelStreamEvent,
        ):
            raise AgentRuntimeError(
                "event must be a ModelStreamEvent."
            )

        if event.content:
            self._content_parts.append(
                event.content
            )

        if event.tool_calls:
            self._tool_calls.extend(
                event.tool_calls
            )

        if event.finish_reason is not None:
            self._finish_reason = event.finish_reason

    def result(self) -> AgentRoundResult:
        """Build the normalized round result."""
        metadata = {}

        if self._finish_reason is not None:
            metadata["finish_reason"] = self._finish_reason

        return AgentRoundRuntime.from_stream_event(
            self.content,
            self.tool_calls,
            metadata=metadata or None,
        )
