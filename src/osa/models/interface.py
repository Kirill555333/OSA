"""Common interfaces and data structures for OSA language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


MessageRole = Literal["system", "user", "assistant", "tool"]


class ModelError(RuntimeError):
    """Base exception for model-related errors."""


class ModelConnectionError(ModelError):
    """Raised when OSA cannot connect to a model backend."""


class ModelResponseError(ModelError):
    """Raised when a model backend returns an invalid response."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from a model to execute a tool."""

    id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Description of a tool exposed to a language model."""

    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One message exchanged with a language model."""

    role: MessageRole
    content: str
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Input passed from OSA to a language model."""

    messages: tuple[ChatMessage, ...]
    temperature: float = 0.2
    max_tokens: int = 512
    tools: tuple[ToolDefinition, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized response returned by a language model."""

    content: str
    model_name: str
    finish_reason: str | None = None
    usage: Mapping[str, Any] = field(default_factory=dict)
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelStreamEvent:
    """One normalized event produced during model streaming."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None


class ModelInterface(ABC):
    """Abstract interface implemented by every OSA language model backend."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable model name."""
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        """Generate a response from the supplied conversation."""
        raise NotImplementedError

    def generate_stream(
        self,
        request: ModelRequest,
    ) -> Iterator[str]:
        """Generate response text incrementally when supported."""
        for event in self.generate_stream_events(request):
            if event.content:
                yield event.content

    def generate_stream_events(
        self,
        request: ModelRequest,
    ) -> Iterator[ModelStreamEvent]:
        """Generate normalized text and tool-call events incrementally."""
        if request.tools:
            response = self.generate(request)

            yield ModelStreamEvent(
                content=response.content,
                tool_calls=response.tool_calls,
                finish_reason=response.finish_reason,
            )
            return

        for chunk in self.generate_stream(request):
            if chunk:
                yield ModelStreamEvent(
                    content=chunk
                )

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the model backend is available."""
        raise NotImplementedError
