"""Tool abstractions and registry for OSA."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

from osa.models.interface import ToolDefinition


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Normalized result returned by an OSA tool."""

    success: bool
    output: str = ""
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ToolError(RuntimeError):
    """Base exception raised by OSA tools."""


class ToolInterface(ABC):
    """Abstract interface implemented by every OSA tool."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique tool name."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of the tool."""
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema describing tool arguments."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Execute the tool with the supplied arguments."""
        raise NotImplementedError


class ToolRegistry:
    """Store and retrieve tools available to OSA."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolInterface] = {}

    def register(self, tool: ToolInterface) -> None:
        """Register a tool."""
        name = tool.name.strip()

        if not name:
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")

        self._tools[name] = tool

    def get(self, name: str) -> ToolInterface:
        """Return a registered tool by name."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered.") from exc

    def names(self) -> tuple[str, ...]:
        """Return registered tool names."""
        return tuple(self._tools)

    def describe(self) -> tuple[dict[str, str], ...]:
        """Return basic descriptions of all registered tools."""
        return tuple(
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self._tools.values()
        )

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return tool definitions suitable for a language model."""
        return tuple(
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in self._tools.values()
        )

    def __len__(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)
