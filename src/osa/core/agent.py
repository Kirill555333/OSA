"""Core agent implementation for OSA."""

from __future__ import annotations

from typing import Any, Mapping

from osa.core.context import ConversationContext
from osa.models import (
    ChatMessage,
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
)
from osa.tools import ToolRegistry, ToolResult


class AgentError(RuntimeError):
    """Base exception raised by the OSA agent."""


class Agent:
    """Coordinate conversation state, model interaction, and tools."""

    def __init__(
        self,
        model: ModelInterface,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        tool_registry: ToolRegistry | None = None,
        max_tool_rounds: int = 8,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        if max_tool_rounds <= 0:
            raise ValueError("max_tool_rounds must be greater than zero.")

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_tool_rounds = max_tool_rounds
        self._context = ConversationContext(system_prompt)
        self._tool_registry = tool_registry or ToolRegistry()

    @property
    def model(self) -> ModelInterface:
        """Return the model used by the agent."""
        return self._model

    @property
    def context(self) -> ConversationContext:
        """Return the current conversation context."""
        return self._context

    @property
    def tools(self) -> ToolRegistry:
        """Return the registry containing available tools."""
        return self._tool_registry

    def chat(self, user_input: str) -> ModelResponse:
        """Process a user message, including any required tool calls."""
        user_input = user_input.strip()

        if not user_input:
            raise ValueError("user_input cannot be empty.")

        self._context.add(
            ChatMessage(
                role="user",
                content=user_input,
            )
        )

        try:
            return self._run_agent_loop()
        except AgentError:
            self._context.remove_last()
            raise

    def execute_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Execute a registered tool."""
        try:
            tool = self._tool_registry.get(tool_name)
        except KeyError as exc:
            raise AgentError(str(exc)) from exc

        return tool.execute(arguments)

    def _run_agent_loop(self) -> ModelResponse:
        """Run the model/tool loop until a final response is generated."""
        for _ in range(self._max_tool_rounds):
            request = ModelRequest(
                messages=self._context.messages(),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=self._tool_registry.definitions(),
            )

            try:
                response = self._model.generate(request)
            except ModelError as exc:
                raise AgentError(
                    f"Model request failed: {exc}"
                ) from exc

            if not response.tool_calls:
                self._context.add(
                    ChatMessage(
                        role="assistant",
                        content=response.content,
                    )
                )
                return response

            self._context.add(
                ChatMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            for tool_call in response.tool_calls:
                result = self.execute_tool(
                    tool_call.name,
                    tool_call.arguments,
                )

                self._context.add(
                    ChatMessage(
                        role="tool",
                        content=self._tool_result_content(result),
                        tool_call_id=tool_call.id,
                    )
                )

        raise AgentError(
            f"Maximum tool rounds exceeded ({self._max_tool_rounds})."
        )

    @staticmethod
    def _tool_result_content(result: ToolResult) -> str:
        """Convert a tool result into model-readable text."""
        if result.success:
            return result.output

        return f"Tool execution failed: {result.error or 'Unknown error.'}"

    def reset(self) -> None:
        """Reset the conversation while preserving the system prompt."""
        system_message: ChatMessage | None = None

        for message in self._context.messages():
            if message.role == "system":
                system_message = message
                break

        self._context.clear()

        if system_message is not None:
            self._context.add(system_message)
