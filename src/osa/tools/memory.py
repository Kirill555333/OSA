"""Long-term memory tools for OSA."""

from __future__ import annotations

from typing import Any, Mapping

from osa.memory import LongTermMemory
from osa.tools.registry import ToolInterface, ToolResult


class RememberTool(ToolInterface):
    """Save a user-approved memory in long-term storage."""

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "remember"

    @property
    def description(self) -> str:
        """Return a human-readable description."""
        return (
            "Save an important piece of information in OSA's long-term memory. "
            "Use this when the user explicitly asks OSA to remember something."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Information that OSA should remember.",
                },
                "category": {
                    "type": "string",
                    "description": "Category such as user, project, preference, or platform.",
                    "default": "general",
                },
                "importance": {
                    "type": "integer",
                    "description": "Importance from 1 to 10.",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Save a memory."""
        content = arguments.get("content")
        category = arguments.get("category", "general")
        importance = arguments.get("importance", 5)

        if not isinstance(content, str):
            return ToolResult(
                success=False,
                error="Argument 'content' must be a string.",
            )

        if not isinstance(category, str):
            return ToolResult(
                success=False,
                error="Argument 'category' must be a string.",
            )

        if not isinstance(importance, int) or isinstance(
            importance,
            bool,
        ):
            return ToolResult(
                success=False,
                error="Argument 'importance' must be an integer.",
            )

        try:
            record = self._memory.save(
                content,
                category=category,
                importance=importance,
            )
        except ValueError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=(
                f"Memory saved successfully. "
                f"Memory ID: {record.id}."
            ),
            metadata={
                "memory_id": record.id,
                "category": record.category,
                "importance": record.importance,
            },
        )


class RecallTool(ToolInterface):
    """Search OSA's long-term memory."""

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "recall"

    @property
    def description(self) -> str:
        """Return relevant memories from OSA's long-term memory."""

        return (
            "Search OSA's long-term memory for information relevant to a query."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Words or phrases to search for in memory.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Search long-term memory."""
        query = arguments.get("query")
        limit = arguments.get("limit", 5)

        if not isinstance(query, str):
            return ToolResult(
                success=False,
                error="Argument 'query' must be a string.",
            )

        if not isinstance(limit, int) or isinstance(
            limit,
            bool,
        ):
            return ToolResult(
                success=False,
                error="Argument 'limit' must be an integer.",
            )

        if not 1 <= limit <= 20:
            return ToolResult(
                success=False,
                error="Argument 'limit' must be between 1 and 20.",
            )

        try:
            records = self._memory.search(
                query,
                limit=limit,
            )
        except ValueError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        if not records:
            return ToolResult(
                success=True,
                output="No matching memories were found.",
                metadata={"count": 0},
            )

        lines = []

        for record in records:
            lines.append(
                (
                    f"[ID {record.id}] "
                    f"[category: {record.category}] "
                    f"[importance: {record.importance}] "
                    f"{record.content}"
                )
            )

        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"count": len(records)},
        )


class ForgetTool(ToolInterface):
    """Delete a specific memory from OSA's long-term storage."""

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    @property
    def name(self) -> str:
        """Return the unique tool name."""
        return "forget"

    @property
    def description(self) -> str:
        """Return a specific memory by ID from OSA's long-term memory."""

        return (
            "Delete a specific long-term memory by its memory ID. "
            "Use recall first when the memory ID is unknown."
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        """Return the JSON schema for tool arguments."""
        return {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "ID of the memory to delete.",
                    "minimum": 1,
                }
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        }

    def execute(
        self,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Delete a memory."""
        memory_id = arguments.get("memory_id")

        if not isinstance(memory_id, int) or isinstance(
            memory_id,
            bool,
        ):
            return ToolResult(
                success=False,
                error="Argument 'memory_id' must be an integer.",
            )

        if memory_id <= 0:
            return ToolResult(
                success=False,
                error="Argument 'memory_id' must be greater than zero.",
            )

        try:
            self._memory.delete(memory_id)
        except KeyError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
            )

        return ToolResult(
            success=True,
            output=f"Memory {memory_id} was deleted.",
            metadata={"memory_id": memory_id},
        )
