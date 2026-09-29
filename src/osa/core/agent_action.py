"""Agent-to-unified-action bridge for OSA."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.models import ToolCall


class AgentActionBridgeError(RuntimeError):
    """Raised when an agent action cannot be converted safely."""


class AgentActionBridge:
    """Convert agent tool calls and explicit actions into ActionRequest."""

    @staticmethod
    def action_request(
        *,
        kind: ActionKind,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> ActionRequest:
        """Build a validated unified action request."""
        if not isinstance(kind, ActionKind):
            raise AgentActionBridgeError(
                "kind must be an ActionKind."
            )

        if not isinstance(name, str) or not name.strip():
            raise AgentActionBridgeError(
                "Action name must be a non-empty string."
            )

        if arguments is None:
            normalized_arguments: dict[str, Any] = {}
        elif isinstance(arguments, Mapping):
            normalized_arguments = dict(arguments)
        else:
            raise AgentActionBridgeError(
                "Action arguments must be a mapping."
            )

        if metadata is None:
            normalized_metadata: dict[str, Any] = {}
        elif isinstance(metadata, Mapping):
            normalized_metadata = dict(metadata)
        else:
            raise AgentActionBridgeError(
                "Action metadata must be a mapping."
            )

        normalized_metadata["source"] = "agent"

        normalized_request_id = request_id

        if normalized_request_id is not None:
            if (
                not isinstance(normalized_request_id, str)
                or not normalized_request_id.strip()
            ):
                raise AgentActionBridgeError(
                    "request_id must be a non-empty string."
                )

            normalized_request_id = (
                normalized_request_id.strip()
            )

        request_kwargs: dict[str, Any] = {
            "kind": kind,
            "name": name.strip(),
            "arguments": normalized_arguments,
            "metadata": normalized_metadata,
        }

        if normalized_request_id is not None:
            request_kwargs["request_id"] = normalized_request_id

        try:
            return ActionRequest(**request_kwargs)
        except Exception as exc:
            raise AgentActionBridgeError(
                f"Failed to build ActionRequest: {exc}"
            ) from exc

    @classmethod
    def tool_request(
        cls,
        tool_name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> ActionRequest:
        """Build a unified TOOL action request."""
        return cls.action_request(
            kind=ActionKind.TOOL,
            name=tool_name,
            arguments=arguments,
            metadata=metadata,
            request_id=request_id,
        )

    @classmethod
    def from_tool_call(
        cls,
        tool_call: ToolCall,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ActionRequest:
        """Convert a model ToolCall into a unified TOOL action request."""
        if not isinstance(tool_call, ToolCall):
            raise AgentActionBridgeError(
                "tool_call must be a ToolCall."
            )

        call_id = tool_call.id

        if (
            not isinstance(call_id, str)
            or not call_id.strip()
        ):
            raise AgentActionBridgeError(
                "Tool call id must be a non-empty string."
            )

        return cls.tool_request(
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            metadata=metadata,
            request_id=call_id,
        )
