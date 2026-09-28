"""Resolve natural-language task steps into validated tool actions."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

from osa.models import (
    ChatMessage,
    ModelError,
    ModelInterface,
    ModelRequest,
    ToolDefinition,
)
from osa.tasks.manager import Task
from osa.tools import ToolRegistry


class TaskActionError(RuntimeError):
    """Raised when a task step cannot be resolved into a tool action."""


@dataclass(frozen=True, slots=True)
class TaskAction:
    """Validated action selected for one task."""

    tool_name: str
    arguments: Mapping[str, Any]


class TaskActionResolver:
    """Use the model to map a task description to one registered tool."""

    _SUBMIT_TOOL_NAME = "submit_action"

    def __init__(
        self,
        model: ModelInterface,
        tool_registry: ToolRegistry,
        *,
        temperature: float = 0.0,
        max_tokens: int = 384,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        self._model = model
        self._tool_registry = tool_registry
        self._temperature = temperature
        self._max_tokens = max_tokens

    def resolve(
        self,
        task: Task,
        outputs: Mapping[str, str],
    ) -> TaskAction:
        """Resolve one task into a validated tool action."""
        available_tools = self._tool_registry.definitions()

        if not available_tools:
            raise TaskActionError(
                "No tools are available for task execution."
            )

        messages = self._build_messages(
            task,
            outputs,
            available_tools,
        )

        request = ModelRequest(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tools=(
                self._submit_tool_definition(
                    available_tools
                ),
            ),
        )

        try:
            response = self._model.generate(request)
        except ModelError as exc:
            raise TaskActionError(
                f"Task action resolution failed: {exc}"
            ) from exc

        for tool_call in response.tool_calls:
            if tool_call.name != self._SUBMIT_TOOL_NAME:
                continue

            return self._parse_action(
                tool_call.arguments,
                available_tools,
            )

        content = response.content.strip()

        if not content:
            raise TaskActionError(
                "Model returned no task action."
            )

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise TaskActionError(
                "Model returned invalid task action data."
            ) from exc

        if not isinstance(payload, Mapping):
            raise TaskActionError(
                "Task action must be an object."
            )

        return self._parse_action(
            payload,
            available_tools,
        )

    def _build_messages(
        self,
        task: Task,
        outputs: Mapping[str, str],
        available_tools: tuple[ToolDefinition, ...],
    ) -> tuple[ChatMessage, ...]:
        """Build the isolated action-resolution prompt."""
        tool_names = ", ".join(
            tool.name
            for tool in available_tools
        )

        context_lines = [
            f"- {task_id}: {output}"
            for task_id, output in outputs.items()
        ]

        output_context = (
            "\nPrevious task outputs:\n"
            + "\n".join(context_lines)
            if context_lines
            else ""
        )

        system = (
            "You are OSA's task action resolver. "
            "Choose exactly one available tool for the supplied task. "
            "Do not execute the tool. "
            "Do not invent tools. "
            "Return the selection only by calling submit_action. "
            f"Available tools: {tool_names}."
        )

        return (
            ChatMessage(
                role="system",
                content=system,
            ),
            ChatMessage(
                role="user",
                content=(
                    f"Task: {task.description}"
                    f"{output_context}"
                ),
            ),
        )

    def _submit_tool_definition(
        self,
        available_tools: tuple[ToolDefinition, ...],
    ) -> ToolDefinition:
        """Build the synthetic submit_action tool definition."""
        tool_names = [
            tool.name
            for tool in available_tools
        ]

        return ToolDefinition(
            name=self._SUBMIT_TOOL_NAME,
            description=(
                "Select exactly one available tool and provide "
                "its execution arguments."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "enum": tool_names,
                    },
                    "arguments": {
                        "type": "object",
                        "description": (
                            "Arguments for the selected tool."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": [
                    "tool_name",
                    "arguments",
                ],
                "additionalProperties": False,
            },
        )

    def _parse_action(
        self,
        payload: Mapping[str, Any],
        available_tools: tuple[ToolDefinition, ...],
    ) -> TaskAction:
        """Validate a model-produced task action."""
        tool_name = payload.get("tool_name")
        arguments = payload.get(
            "arguments",
            {},
        )

        if not isinstance(tool_name, str):
            raise TaskActionError(
                "Task action field 'tool_name' must be a string."
            )

        tool_name = tool_name.strip()

        available_names = {
            tool.name
            for tool in available_tools
        }

        if tool_name not in available_names:
            raise TaskActionError(
                f"Unknown tool selected for task: '{tool_name}'."
            )

        if not isinstance(arguments, Mapping):
            raise TaskActionError(
                "Task action field 'arguments' must be an object."
            )

        return TaskAction(
            tool_name=tool_name,
            arguments=dict(arguments),
        )
