"""Core agent implementation for OSA."""

from __future__ import annotations

from collections.abc import Iterator
import time
from typing import Any, Mapping

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.actions.pipeline import ActionSafetyPipeline
from osa.actions.router import ActionRouter
from osa.actions.tool import ToolRegistryActionAdapter
from osa.core.agent_action_policy import (
    AgentModeActionPolicy,
    LegacyConfirmationActionHandler,
    LegacyPermissionActionPolicy,
)
from osa.core.agent_observability import AgentObservability
from osa.core.agent_recovery import (
    AgentRecoveryIntegration,
    AgentRecoveryIntegrationError,
)
from osa.core.agent_runtime import (
    AgentRoundExecutor,
    AgentRoundRuntime,
    AgentRuntimeError,
)
from osa.core.agent_stream import AgentStreamAccumulator
from osa.core.agent_tool_round import AgentToolRound, AgentToolRoundError
from osa.core.context import ConversationContext
from osa.core.modes import (
    AgentMode,
    AgentModePolicy,
)
from osa.memory.integration import (
    MemoryIntegration,
    MemoryIntegrationResult,
)
from osa.tasks.autonomous import (
    AutonomousLoop,
    AutonomousRunReport,
)
from osa.recovery import ErrorRecovery, RecoveryConfig
from osa.memory.automatic import AutomaticMemory
from osa.recovery_contracts import (
    RecoveryFailureKind,
    RecoveryResult,
)

from osa.models import (
    ChatMessage,
    ModelError,
    ModelInterface,
    ModelRequest,
    ModelResponse,
    ToolCall,
)
from osa.permissions import (
    ConfirmationHandler,
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
        memory_integration: MemoryIntegration | None = None,
        mode: AgentMode | str = AgentMode.CHAT,
        mode_policy: AgentModePolicy | None = None,
        autonomous_loop: AutonomousLoop | None = None,
        automatic_memory: AutomaticMemory | None = None,
        recovery: ErrorRecovery | None = None,
        recovery_integration: AgentRecoveryIntegration | None = None,
    ) -> None:
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        if max_tool_rounds <= 0:
            raise ValueError(
                "max_tool_rounds must be greater than zero."
            )

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

        self._memory_integration = memory_integration
        self._mode_policy = mode_policy or AgentModePolicy()
        self._mode = self._normalize_mode(mode)
        self._autonomous_loop = autonomous_loop
        self._recovery = (
            recovery
            or ErrorRecovery(
                RecoveryConfig(
                    max_model_retries=1,
                    model_retry_delay_seconds=0.25,
                )
            )
        )
        self._automatic_memory = automatic_memory
        self._recovery_integration = recovery_integration

        self._current_round_number: int | None = None

        self._action_router = ActionRouter(
            {
                ActionKind.TOOL: ToolRegistryActionAdapter(
                    self._tool_registry
                ),
            }
        )

        self._action_safety_pipeline = ActionSafetyPipeline(
            router=self._action_router,
            mode_policy=AgentModeActionPolicy(
                self._mode_policy,
                lambda: self._mode,
            ),
            permission_policy=LegacyPermissionActionPolicy(
                self._permission_policy,
                logger=self._log,
            ),
            confirmation_handler=LegacyConfirmationActionHandler(
                self._confirmation_handler
            ),
        )

        self._round_executor = AgentRoundExecutor(
            self._execute_model_tool_call
        )

        self._observability = AgentObservability(
            self._log
        )

    @property
    def mode(self) -> AgentMode:
        """Return the current execution mode."""
        return self._mode

    @property
    def mode_profile(self):
        """Return the current mode profile."""
        return self._mode_policy.profile(self._mode)

    def set_mode(self, mode: AgentMode | str) -> None:
        """Change the execution mode."""
        self._mode = self._normalize_mode(mode)

    def _normalize_mode(
        self,
        mode: AgentMode | str,
    ) -> AgentMode:
        if isinstance(mode, AgentMode):
            return mode

        return self._mode_policy.profile(mode).mode

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

    @property
    def recovery_integration(
        self,
    ) -> AgentRecoveryIntegration | None:
        """Return the optional unified recovery integration."""
        return self._recovery_integration

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        """
        Execute an action through the unified recovery pipeline.

        This explicit API remains unchanged and delegates to the configured
        recovery integration.
        """
        if self._recovery_integration is None:
            raise AgentError(
                "Unified recovery integration is not configured."
            )

        try:
            return self._recovery_integration.execute(
                request,
                run_id=run_id,
                task_id=task_id,
            )
        except AgentRecoveryIntegrationError as exc:
            raise AgentError(
                str(exc)
            ) from exc

    def chat(self, user_input: str) -> ModelResponse:
        """Process a user message, including memory and tool calls."""
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
            memory_result = self._retrieve_memory(
                user_input
            )

            response = self._run_agent_loop(
                memory_result.prompt
                if memory_result is not None
                else None
            )

            self._capture_automatic_memory(
                user_input
            )

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

    def chat_stream(
        self,
        user_input: str,
    ) -> Iterator[str]:
        """Process a user message and stream text with tool support."""
        user_input = user_input.strip()

        if not user_input:
            raise ValueError(
                "user_input cannot be empty."
            )

        previous_context = self._context.messages()
        started_at = time.perf_counter()

        self._context.add(
            ChatMessage(
                role="user",
                content=user_input,
            )
        )

        self._log(
            "agent.chat_stream.started",
            input_length=len(user_input),
            context_messages=len(previous_context),
        )

        try:
            memory_result = self._retrieve_memory(
                user_input
            )

            memory_context = (
                memory_result.prompt
                if memory_result is not None
                else None
            )

            for round_number in range(
                1,
                self._max_tool_rounds + 1,
            ):
                self._current_round_number = round_number

                model_request = ModelRequest(
                    messages=self._model_messages(
                        memory_context
                    ),
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    tools=self._available_tool_definitions(),
                )

                accumulator = AgentStreamAccumulator()

                try:
                    for event in self._model.generate_stream_events(
                        model_request
                    ):
                        accumulator.add(event)

                        if event.content:
                            yield event.content

                except ModelError as exc:
                    raise AgentError(
                        f"Model streaming failed: {exc}"
                    ) from exc

                round_result = accumulator.result()

                if round_result.completed:
                    self._context.add(
                        ChatMessage(
                            role="assistant",
                            content=round_result.content,
                        )
                    )

                    self._capture_automatic_memory(
                        user_input
                    )

                    self._log(
                        "agent.chat_stream.completed",
                        duration_ms=self._duration_ms(
                            started_at
                        ),
                        response_length=len(
                            round_result.content
                        ),
                    )

                    return

                self._context.add(
                    ChatMessage(
                        role="assistant",
                        content=round_result.content,
                        tool_calls=round_result.tool_calls,
                    )
                )

                executions = self._execute_model_tool_calls(
                    round_result.tool_calls
                )

                for execution in executions:
                    tool_call = execution.tool_call

                    if execution.error is not None:
                        if isinstance(
                            execution.error,
                            PermissionDeniedError,
                        ):
                            self._context.add(
                                ChatMessage(
                                    role="tool",
                                    content=(
                                        "Tool execution denied: "
                                        f"{execution.error}"
                                    ),
                                    tool_call_id=tool_call.id,
                                )
                            )
                            continue

                        error_message = (
                            ErrorRecovery.tool_exception_message(
                                tool_call.name,
                                execution.error,
                            )
                        )

                        self._context.add(
                            ChatMessage(
                                role="tool",
                                content=error_message,
                                tool_call_id=tool_call.id,
                            )
                        )
                        continue

                    if execution.result is None:
                        raise AgentError(
                            "Tool execution returned no result or error."
                        )

                    self._context.add(
                        ChatMessage(
                            role="tool",
                            content=self._tool_result_content(
                                execution.result
                            ),
                            tool_call_id=tool_call.id,
                        )
                    )

            raise AgentError(
                "Maximum tool rounds exceeded "
                f"({self._max_tool_rounds})."
            )

        except AgentError:
            self._context.restore(
                previous_context
            )
            raise

    def reset(self) -> None:
        """Reset the conversation while preserving the system prompt."""
        system_message = next(
            (
                message
                for message in self._context.messages()
                if message.role == "system"
            ),
            None,
        )

        self._context.clear()

        if system_message is not None:
            self._context.add(system_message)

    def _capture_automatic_memory(
        self,
        user_input: str,
    ) -> None:
        """Capture a stable memory without affecting the chat result."""
        if self._automatic_memory is None:
            return

        started_at = time.perf_counter()

        try:
            result = self._automatic_memory.capture(
                user_input
            )
        except Exception as exc:
            self._log(
                "memory.automatic.failed",
                duration_ms=self._duration_ms(started_at),
                error=str(exc),
            )
            return

        self._log(
            "memory.automatic",
            duration_ms=self._duration_ms(started_at),
            saved=result.saved,
            memory_id=result.memory_id,
            reason=result.reason,
        )

    def run_autonomous(
        self,
        goal: str,
    ) -> AutonomousRunReport:
        """Run a goal through the configured autonomous loop."""
        if self._mode != AgentMode.AUTONOMOUS:
            raise AgentError(
                "Autonomous execution requires AUTONOMOUS mode."
            )

        if self._autonomous_loop is None:
            raise AgentError(
                "Autonomous execution is not configured."
            )

        return self._autonomous_loop.run(goal)

    def execute_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        request_id: str | None = None,
    ) -> ToolResult:
        """
        Execute a registered tool through unified safety and recovery.

        The public return type remains ToolResult for backward compatibility.
        """
        try:
            request = AgentToolRound.request(
                tool_name,
                arguments,
                request_id=request_id,
                metadata={
                    "execution_path": "agent.execute_tool",
                },
            )
        except AgentToolRoundError as exc:
            raise AgentError(str(exc)) from exc

        return self._execute_action_request(
            request
        )

    def _execute_model_tool_call(
        self,
        tool_call: ToolCall,
    ) -> ToolResult:
        """Execute one model tool call through the shared action path."""
        try:
            metadata = {}

            if self._current_round_number is not None:
                metadata["round"] = (
                    self._current_round_number
                )

            request = AgentToolRound.request_from_tool_call(
                tool_call,
                metadata=metadata,
            )
        except AgentToolRoundError as exc:
            raise AgentError(
                str(exc)
            ) from exc

        return self._execute_action_request(
            request
        )

    def _execute_model_tool_calls(
        self,
        tool_calls: tuple[ToolCall, ...],
    ):
        """Execute all model tool calls through the shared round executor."""
        try:
            return self._round_executor.execute(
                tool_calls
            )
        except AgentRuntimeError as exc:
            raise AgentError(
                str(exc)
            ) from exc

    def _execute_action_request(
        self,
        request: ActionRequest,
    ) -> ToolResult:
        """Execute one TOOL ActionRequest through unified infrastructure."""
        round_number = request.metadata.get(
            "round"
        )

        if not isinstance(
            round_number,
            int,
        ):
            round_number = None

        self._observability.action_requested(
            request,
            round_number=round_number,
        )

        recovery_result: RecoveryResult | None = None

        if self._recovery_integration is None:
            action_result = self._action_safety_pipeline.dispatch(
                request
            )

            self._log_action_outcome(
                request,
                action_result,
                round_number=round_number,
            )

            self._raise_for_action_denial(
                action_result,
                None,
            )
        else:
            try:
                recovery_result = self._recovery_integration.execute(
                    request
                )
            except AgentRecoveryIntegrationError as exc:
                failure_result = ActionResult.failed(
                    request.request_id,
                    str(exc),
                )

                self._observability.action_completed(
                    request,
                    failure_result,
                    recovery=None,
                    round_number=round_number,
                )

                raise AgentError(
                    str(exc)
                ) from exc

            if not recovery_result.success:
                underlying = recovery_result.result

                if isinstance(
                    underlying,
                    ActionResult,
                ):
                    self._log_action_outcome(
                        request,
                        underlying,
                        recovery=recovery_result,
                        round_number=round_number,
                    )

                    self._raise_for_action_denial(
                        underlying,
                        recovery_result,
                    )

                    return self._tool_result_from_action_result(
                        underlying,
                        recovery_result,
                    )

                failure_result = ActionResult.failed(
                    request.request_id,
                    recovery_result.error
                    or "Tool recovery failed.",
                )

                self._observability.action_completed(
                    request,
                    failure_result,
                    recovery=recovery_result,
                    round_number=round_number,
                )

                if recovery_result.failure_kind in {
                    RecoveryFailureKind.PERMISSION_DENIED,
                    RecoveryFailureKind.CONFIRMATION_DENIED,
                }:
                    self._observability.action_denied(
                        request,
                        decision=(
                            recovery_result.failure_kind.value
                        ),
                        reason=recovery_result.error,
                        round_number=round_number,
                    )

                    raise PermissionDeniedError(
                        recovery_result.error
                        or (
                            f"Action denied for tool "
                            f"'{request.name}'."
                        )
                    )

                return ToolResult(
                    success=False,
                    error=(
                        recovery_result.error
                        or "Tool recovery failed."
                    ),
                    metadata={
                        "recovery_failure_kind": (
                            recovery_result.failure_kind.value
                            if recovery_result.failure_kind is not None
                            else None
                        ),
                        "recovery_attempts": (
                            recovery_result.attempt_count
                        ),
                    },
                )

            if not isinstance(
                recovery_result.result,
                ActionResult,
            ):
                failure_result = ActionResult.failed(
                    request.request_id,
                    "Unified recovery returned an invalid action result.",
                )

                self._observability.action_completed(
                    request,
                    failure_result,
                    recovery=recovery_result,
                    round_number=round_number,
                )

                raise AgentError(
                    "Unified recovery returned an invalid action result."
                )

            action_result = recovery_result.result

            self._log_action_outcome(
                request,
                action_result,
                recovery=recovery_result,
                round_number=round_number,
            )

            self._raise_for_action_denial(
                action_result,
                recovery_result,
            )

        return self._tool_result_from_action_result(
            action_result,
            recovery_result,
        )

    def _log_action_outcome(
        self,
        request: ActionRequest,
        action_result: ActionResult,
        *,
        recovery: RecoveryResult | None = None,
        round_number: int | None = None,
    ) -> None:
        """Emit telemetry for an action outcome without changing execution."""
        stage = action_result.metadata.get(
            "pipeline_stage"
        )
        code = action_result.metadata.get(
            "pipeline_error"
        )

        if stage in {
            "mode",
            "permission",
            "safety",
            "confirmation",
        } or code in {
            "not_confirmed",
            "denied",
        }:
            self._observability.action_denied(
                request,
                decision=(
                    str(code)
                    if code is not None
                    else str(stage)
                    if stage is not None
                    else "denied"
                ),
                reason=action_result.error,
                round_number=round_number,
            )
            return

        self._observability.action_completed(
            request,
            action_result,
            recovery=recovery,
            round_number=round_number,
        )

    @staticmethod
    def _raise_for_action_denial(
        action_result: ActionResult,
        recovery_result: RecoveryResult | None,
    ) -> None:
        """Preserve legacy PermissionDeniedError semantics."""
        stage = action_result.metadata.get(
            "pipeline_stage"
        )
        code = action_result.metadata.get(
            "pipeline_error"
        )

        if stage in {
            "mode",
            "permission",
            "safety",
            "confirmation",
        }:
            raise PermissionDeniedError(
                action_result.error
                or "Action was denied by the safety pipeline."
            )

        if recovery_result is not None and (
            recovery_result.failure_kind
            in {
                RecoveryFailureKind.PERMISSION_DENIED,
                RecoveryFailureKind.CONFIRMATION_DENIED,
            }
        ):
            raise PermissionDeniedError(
                recovery_result.error
                or action_result.error
                or "Action was denied."
            )

        if code in {
            "not_confirmed",
            "denied",
        }:
            raise PermissionDeniedError(
                action_result.error
                or "Action was denied."
            )

    @staticmethod
    def _tool_result_from_action_result(
        action_result: ActionResult,
        recovery_result: RecoveryResult | None,
    ) -> ToolResult:
        """Convert an ActionResult back to the legacy ToolResult API."""
        metadata = dict(
            action_result.metadata
        )

        tool_result_data = action_result.data.get(
            "tool_result"
        )

        if isinstance(
            tool_result_data,
            Mapping,
        ):
            original_metadata = tool_result_data.get(
                "metadata"
            )

            if isinstance(
                original_metadata,
                Mapping,
            ):
                metadata.update(
                    original_metadata
                )

        if recovery_result is not None:
            metadata["recovery_attempts"] = (
                recovery_result.attempt_count
            )

            if recovery_result.failure_kind is not None:
                metadata["recovery_failure_kind"] = (
                    recovery_result.failure_kind.value
                )

        return ToolResult(
            success=action_result.success,
            output=action_result.output,
            error=action_result.error,
            metadata=metadata,
        )

    def _retrieve_memory(
        self,
        user_input: str,
    ) -> MemoryIntegrationResult | None:
        """Retrieve relevant memory without modifying conversation history."""
        if self._memory_integration is None:
            return None

        result = self._memory_integration.retrieve(
            user_input
        )

        self._log(
            "memory.retrieval",
            query_length=len(user_input),
            result_count=len(result.memories),
            memories=[
                {
                    "id": item.memory.id,
                    "score": item.score,
                }
                for item in result.memories
            ],
        )

        return result

    def _run_agent_loop(
        self,
        memory_context: str | None = None,
    ) -> ModelResponse:
        """Run the model/tool loop until a final response is generated."""
        for round_number in range(
            1,
            self._max_tool_rounds + 1,
        ):
            self._current_round_number = round_number

            request = ModelRequest(
                messages=self._model_messages(
                    memory_context
                ),
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                tools=self._available_tool_definitions(),
            )

            started_at = time.perf_counter()

            try:
                response, attempts = self._recovery.run_model(
                    lambda: self._model.generate(
                        request
                    )
                )

                if attempts > 1:
                    self._log(
                        "recovery.model_retry",
                        attempts=attempts,
                        round=round_number,
                    )

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

            round_result = AgentRoundRuntime.from_model_response(
                response
            )

            if round_result.completed:
                self._context.add(
                    ChatMessage(
                        role="assistant",
                        content=round_result.content,
                    )
                )

                return response

            self._context.add(
                ChatMessage(
                    role="assistant",
                    content=round_result.content,
                    tool_calls=round_result.tool_calls,
                )
            )

            executions = self._execute_model_tool_calls(
                round_result.tool_calls
            )

            for execution in executions:
                tool_call = execution.tool_call

                if execution.error is not None:
                    if isinstance(
                        execution.error,
                        PermissionDeniedError,
                    ):
                        self._log(
                            "recovery.tool_denied",
                            tool=tool_call.name,
                            error=str(execution.error),
                        )

                        self._context.add(
                            ChatMessage(
                                role="tool",
                                content=(
                                    "Tool execution denied: "
                                    f"{execution.error}"
                                ),
                                tool_call_id=tool_call.id,
                            )
                        )
                        continue

                    error_message = ErrorRecovery.tool_exception_message(
                        tool_call.name,
                        execution.error,
                    )

                    self._log(
                        "recovery.tool_error",
                        tool=tool_call.name,
                        error=str(execution.error),
                    )

                    self._context.add(
                        ChatMessage(
                            role="tool",
                            content=error_message,
                            tool_call_id=tool_call.id,
                        )
                    )
                    continue

                if execution.result is None:
                    raise AgentError(
                        "Tool execution returned no result or error."
                    )

                self._context.add(
                    ChatMessage(
                        role="tool",
                        content=self._tool_result_content(
                            execution.result
                        ),
                        tool_call_id=tool_call.id,
                    )
                )

        raise AgentError(
            "Maximum tool rounds exceeded "
            f"({self._max_tool_rounds})."
        )

    def _available_tool_definitions(self):
        """Return tools available in the current agent mode."""
        return tuple(
            definition
            for definition in self._tool_registry.definitions()
            if self._mode_policy.supports_tool(
                self._mode,
                definition.name,
            )
        )

    def _model_messages(
        self,
        memory_context: str | None,
    ) -> tuple[ChatMessage, ...]:
        """Build the transient model context for one request."""
        messages = self._context.messages()

        if not memory_context:
            return messages

        memory_message = ChatMessage(
            role="system",
            content=memory_context,
        )

        if (
            messages
            and messages[0].role == "system"
        ):
            return (
                messages[0],
                memory_message,
                *messages[1:],
            )

        return (
            memory_message,
            *messages,
        )

    @staticmethod
    def _tool_result_content(
        result: ToolResult,
    ) -> str:
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
    def _duration_ms(
        started_at: float,
    ) -> float:
        """Return elapsed time in milliseconds."""
        return round(
            (time.perf_counter() - started_at) * 1000,
            2,
        )
