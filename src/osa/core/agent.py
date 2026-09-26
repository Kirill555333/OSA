"""Core agent implementation for OSA."""

from __future__ import annotations

import time
from typing import Any, Mapping

from osa.core.context import ConversationContext
from osa.models import (
    ChatMessage,
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
)
from osa.permissions import (
    ConfirmationHandler,
    ConfirmationRequest,
    PermissionLevel,
    PermissionPolicy,
)
from osa.tools import ToolRegistry, ToolResult
from osa.utils import EventLogger


class AgentError(RuntimeError):
    """Base exception raised by the OSA agent."""


class PermissionDeniedError(AgentError):
    """Raised when a tool execution is denied."""


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
        event_logger: EventLogger | None = None,
        permission_policy: PermissionPolicy | None = None,
        confirmation_handler: ConfirmationHandler | None = None,
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
        self._event_logger = event_logger

        self._permission_policy = (
            permission_policy
            or PermissionPolicy()
        )

        self._confirmation_handler = (
            confirmation_handler
            or ConfirmationHandler()
        )

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

    @property
    def permissions(self) -> PermissionPolicy:
        """Return the permission policy used by the agent."""
        return self._permission_policy

    def chat(self, user_input: str) -> ModelResponse:
        """Process a user message, including any required tool calls."""
        user_input = user_input.strip()

        if not user_input:
            raise ValueError("user_input cannot be empty.")

        previous_context = self._context.messages()
        started_at = time.perf_counter()

        self._context.add(
            ChatMessage(
                role="user",
                content=user_input,
            )
        )

        self._log(
            "agent.chat.started",
            input_length=len(user_input),
            context_messages=len(previous_context),
        )

        try:
            response = self._run_agent_loop()

            self._log(
                "agent.chat.completed",
                duration_ms=self._duration_ms(started_at),
                response_length=len(response.content),
            )

            return response

        except AgentError as exc:
            self._context.restore(previous_context)

            self._log(
                "agent.chat.failed",
                duration_ms=self._duration_ms(started_at),
                error=str(exc),
            )

            raise

    def execute_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> ToolResult:
        """Execute a registered tool after permission checks."""
        decision = self._permission_policy.decide(tool_name)

        self._log(
            "permission.decision",
            tool=tool_name,
            level=decision.level.value,
            reason=decision.reason,
        )

        if decision.level == PermissionLevel.DENY:
            raise PermissionDeniedError(
                f"Permission denied for tool '{tool_name}': "
                f"{decision.reason}"
            )

        if decision.level == PermissionLevel.CONFIRM:
            confirmed = self._confirmation_handler.request(
                ConfirmationRequest(
                    tool_name=tool_name,
                    description=(
                        f"OSA wants to execute tool '{tool_name}'. "
                        "Do you allow this action?"
                    ),
                )
            )

            if not confirmed:
                raise PermissionDeniedError(
                    f"Permission confirmation was not granted for "
                    f"tool '{tool_name}'."
                )

        try:
            tool = self._tool_registry.get(tool_name)
        except KeyError as exc:
            raise AgentError(str(exc)) from exc

        started_at = time.perf_counter()

        self._log(
            "tool.call",
            tool=tool_name,
            arguments=dict(arguments),
        )

        try:
            result = tool.execute(arguments)
        except Exception as exc:
            self._log(
                "tool.failed",
                tool=tool_name,
                duration_ms=self._duration_ms(started_at),
                error=str(exc),
            )
            raise

        self._log(
            "tool.result",
            tool=tool_name,
            success=result.success,
            duration_ms=self._duration_ms(started_at),
            output_length=len(result.output),
            error=result.error,
        )

        return result

    def _run_agent_loop(self) -> ModelResponse:
        """Run the model/tool loop until a final response is generated."""
        for round_number in range(
            1,
            self._max_tool_rounds + 1,
        ):
            request = ModelRequest(
                messages=self._context.messages(),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=self._tool_registry.definitions(),
            )

            started_at = time.perf_counter()

            try:
                response = self._model.generate(request)
            except ModelError as exc:
                self._log(
                    "model.failed",
                    duration_ms=self._duration_ms(started_at),
                    error=str(exc),
                    round=round_number,
                )

                raise AgentError(
                    f"Model request failed: {exc}"
                ) from exc

            self._log(
                "model.response",
                duration_ms=self._duration_ms(started_at),
                finish_reason=response.finish_reason,
                content_length=len(response.content),
                tool_call_count=len(response.tool_calls),
                round=round_number,
            )

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
                try:
                    result = self.execute_tool(
                        tool_call.name,
                        tool_call.arguments,
                    )
                except PermissionDeniedError as exc:
                    self._context.add(
                        ChatMessage(
                            role="tool",
                            content=f"Tool execution denied: {exc}",
                            tool_call_id=tool_call.id,
                        )
                    )
                    continue

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

        return (
            "Tool execution failed: "
            f"{result.error or 'Unknown error.'}"
        )

    def _log(
        self,
        event: str,
        **data: Any,
    ) -> None:
        """Write an event when observability is enabled."""
        if self._event_logger is not None:
            self._event_logger.log(
                event,
                **data,
            )

    @staticmethod
    def _duration_ms(started_at: float) -> float:
        """Return elapsed time in milliseconds."""
        return round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )

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
