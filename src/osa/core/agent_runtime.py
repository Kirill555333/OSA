"""Internal runtime contracts and orchestration for Agent rounds."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from osa.models import ModelResponse, ToolCall
from osa.tools import ToolResult


class AgentRuntimeError(RuntimeError):
    """Raised when an agent runtime round is invalid."""


@dataclass(frozen=True, slots=True)
class AgentRoundResult:
    """Normalized result of one model interaction round."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    model_response: ModelResponse | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise AgentRuntimeError(
                "round content must be a string."
            )

        if not isinstance(self.tool_calls, tuple):
            raise AgentRuntimeError(
                "round tool_calls must be a tuple."
            )

        for tool_call in self.tool_calls:
            if not isinstance(tool_call, ToolCall):
                raise AgentRuntimeError(
                    "round tool_calls must contain ToolCall values."
                )

        if self.model_response is not None and not isinstance(
            self.model_response,
            ModelResponse,
        ):
            raise AgentRuntimeError(
                "model_response must be a ModelResponse or None."
            )

        if self.metadata is not None and not isinstance(
            self.metadata,
            dict,
        ):
            raise AgentRuntimeError(
                "round metadata must be a dict or None."
            )

    @property
    def has_tool_calls(self) -> bool:
        """Return True when the model requested tool execution."""
        return bool(self.tool_calls)

    @property
    def completed(self) -> bool:
        """Return True when no further tool execution is requested."""
        return not self.tool_calls


@dataclass(frozen=True, slots=True)
class AgentToolExecution:
    """Normalized outcome for one model tool call."""

    tool_call: ToolCall
    result: ToolResult | None = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.tool_call,
            ToolCall,
        ):
            raise AgentRuntimeError(
                "tool_call must be a ToolCall."
            )

        if self.result is not None and not isinstance(
            self.result,
            ToolResult,
        ):
            raise AgentRuntimeError(
                "result must be a ToolResult or None."
            )

        if self.error is not None and not isinstance(
            self.error,
            Exception,
        ):
            raise AgentRuntimeError(
                "error must be an Exception or None."
            )

        if self.result is None and self.error is None:
            raise AgentRuntimeError(
                "tool execution must contain result or error."
            )

        if self.result is not None and self.error is not None:
            raise AgentRuntimeError(
                "tool execution cannot contain both result and error."
            )

    @property
    def succeeded(self) -> bool:
        """Return True when the tool returned a successful result."""
        return (
            self.result is not None
            and self.result.success
        )


class AgentRoundRuntime:
    """Build normalized round results for shared Agent orchestration."""

    @staticmethod
    def from_model_response(
        response: ModelResponse,
    ) -> AgentRoundResult:
        """Normalize a regular model response."""
        if not isinstance(
            response,
            ModelResponse,
        ):
            raise AgentRuntimeError(
                "response must be a ModelResponse."
            )

        return AgentRoundResult(
            content=response.content,
            tool_calls=response.tool_calls,
            model_response=response,
        )

    @staticmethod
    def from_stream_event(
        content: str,
        tool_calls: tuple[ToolCall, ...] = (),
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRoundResult:
        """Normalize accumulated streaming round data."""
        return AgentRoundResult(
            content=content,
            tool_calls=tool_calls,
            metadata=metadata,
        )


class AgentRoundExecutor:
    """Execute tool calls shared by regular and streaming Agent rounds."""

    def __init__(
        self,
        execute_tool_call: Callable[[ToolCall], ToolResult],
    ) -> None:
        if not callable(execute_tool_call):
            raise AgentRuntimeError(
                "execute_tool_call must be callable."
            )

        self._execute_tool_call = execute_tool_call

    @property
    def execute_tool_call(self) -> Callable[[ToolCall], ToolResult]:
        """Return the configured tool-call execution callback."""
        return self._execute_tool_call

    def execute(
        self,
        tool_calls: Iterable[ToolCall],
    ) -> tuple[AgentToolExecution, ...]:
        """
        Execute every tool call in deterministic input order.

        Exceptions from one tool call are captured so later tool calls in the
        same model round can still execute. The original exception object is
        preserved for the caller's existing error semantics.
        """
        try:
            normalized_calls = tuple(tool_calls)
        except TypeError as exc:
            raise AgentRuntimeError(
                "tool_calls must be iterable."
            ) from exc

        results: list[AgentToolExecution] = []

        for tool_call in normalized_calls:
            if not isinstance(
                tool_call,
                ToolCall,
            ):
                raise AgentRuntimeError(
                    "tool_calls must contain ToolCall values."
                )

            try:
                result = self._execute_tool_call(
                    tool_call
                )
            except Exception as exc:
                results.append(
                    AgentToolExecution(
                        tool_call=tool_call,
                        error=exc,
                    )
                )
                continue

            if not isinstance(
                result,
                ToolResult,
            ):
                raise AgentRuntimeError(
                    "tool execution callback must return ToolResult."
                )

            results.append(
                AgentToolExecution(
                    tool_call=tool_call,
                    result=result,
                )
            )

        return tuple(results)
