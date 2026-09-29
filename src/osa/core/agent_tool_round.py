"""Shared tool-call conversion for Agent tool rounds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osa.actions.contracts import ActionRequest
from osa.core.agent_action import (
    AgentActionBridge,
    AgentActionBridgeError,
)
from osa.models import ToolCall


class AgentToolRoundError(RuntimeError):
    """Raised when a tool call cannot enter the unified action path."""


class AgentToolRound:
    """Build unified action requests from agent and model tool calls."""

    @staticmethod
    def request(
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionRequest:
        """Build one TOOL ActionRequest from explicit tool-call fields."""
        try:
            return AgentActionBridge.tool_request(
                tool_name,
                arguments,
                request_id=request_id,
                metadata=metadata,
            )
        except AgentActionBridgeError as exc:
            raise AgentToolRoundError(
                f"Invalid tool request: {exc}"
            ) from exc

    @classmethod
    def request_from_tool_call(
        cls,
        tool_call: ToolCall,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionRequest:
        """Convert one model ToolCall into one unified ActionRequest."""
        try:
            if not isinstance(
                tool_call,
                ToolCall,
            ):
                raise AgentToolRoundError(
                    "Invalid model tool call: tool_call must be a ToolCall."
                )

            return cls.request(
                tool_call.name,
                tool_call.arguments,
                request_id=tool_call.id,
                metadata=metadata,
            )
        except AgentToolRoundError:
            raise
        except Exception as exc:
            raise AgentToolRoundError(
                f"Invalid model tool call: {exc}"
            ) from exc
