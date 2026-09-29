"""Voice-to-unified-action integration for OSA."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from json import dumps
from typing import Any, Protocol
from uuid import uuid4

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.recovery_contracts import RecoveryResult
from osa.tasks.action import (
    Task,
    TaskActionResolver,
)


class AgentVoiceActionIntegrationError(RuntimeError):
    """Raised when voice action integration cannot complete safely."""


@dataclass(frozen=True, slots=True)
class VoiceActionResponse:
    """Minimal response object required by VoiceSession."""

    content: str


class VoiceActionResolver(Protocol):
    """Resolve one voice command into a unified action request."""

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        """Resolve a natural-language command."""
        ...


class RecoveringVoiceAgent(Protocol):
    """Minimal Agent surface required by the voice action adapter."""

    def execute_action_with_recovery(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        """Execute one action through Agent unified recovery."""
        ...


class TaskActionResolverVoiceAdapter:
    """
    Adapt the existing TaskActionResolver to voice commands.

    The existing resolver remains responsible for model-based action
    selection and tool validation. This adapter only supplies the Task
    wrapper required by that existing API and converts the selected
    TaskAction into an ActionRequest.
    """

    def __init__(
        self,
        resolver: TaskActionResolver,
    ) -> None:
        if resolver is None:
            raise ValueError(
                "resolver is required."
            )

        if not callable(
            getattr(
                resolver,
                "resolve",
                None,
            )
        ):
            raise TypeError(
                "resolver must expose resolve()."
            )

        self._resolver = resolver

    @property
    def resolver(self) -> TaskActionResolver:
        """Return the configured task action resolver."""
        return self._resolver

    def resolve(
        self,
        command: str,
    ) -> ActionRequest:
        """Resolve one voice command into a TOOL ActionRequest."""
        if not isinstance(
            command,
            str,
        ):
            raise TypeError(
                "command must be a string."
            )

        normalized_command = command.strip()

        if not normalized_command:
            raise ValueError(
                "command cannot be empty."
            )

        task_id = str(uuid4())

        task = Task(
            task_id=task_id,
            description=normalized_command,
        )

        try:
            action = self._resolver.resolve(
                task,
                {},
            )
        except Exception as exc:
            raise AgentVoiceActionIntegrationError(
                f"Voice action resolution failed: {exc}"
            ) from exc

        tool_name = getattr(
            action,
            "tool_name",
            None,
        )
        arguments = getattr(
            action,
            "arguments",
            None,
        )

        if not isinstance(
            tool_name,
            str,
        ) or not tool_name.strip():
            raise AgentVoiceActionIntegrationError(
                "Voice action resolver returned an invalid tool name."
            )

        if not isinstance(
            arguments,
            Mapping,
        ):
            raise AgentVoiceActionIntegrationError(
                "Voice action resolver returned invalid arguments."
            )

        try:
            return ActionRequest(
                kind=ActionKind.TOOL,
                name=tool_name,
                arguments=dict(arguments),
                metadata={
                    "source": "voice",
                    "voice_task_id": task_id,
                },
            )
        except Exception as exc:
            raise AgentVoiceActionIntegrationError(
                f"Voice action request construction failed: {exc}"
            ) from exc


class AgentVoiceActionAdapter:
    """
    Expose an Agent-like chat surface for VoiceSession.

    VoiceSession calls `chat(command)`, while this adapter redirects the
    command into:

        resolver -> ActionRequest -> Agent.execute_action_with_recovery

    The returned response intentionally exposes `.content`, matching the
    response contract consumed by VoiceSession.
    """

    def __init__(
        self,
        agent: RecoveringVoiceAgent,
        resolver: VoiceActionResolver,
    ) -> None:
        if agent is None:
            raise ValueError(
                "agent is required."
            )

        if resolver is None:
            raise ValueError(
                "resolver is required."
            )

        if not callable(
            getattr(
                agent,
                "execute_action_with_recovery",
                None,
            )
        ):
            raise TypeError(
                "agent must expose execute_action_with_recovery()."
            )

        if not callable(
            getattr(
                resolver,
                "resolve",
                None,
            )
        ):
            raise TypeError(
                "resolver must expose resolve()."
            )

        self._agent = agent
        self._resolver = resolver

    @property
    def agent(self) -> RecoveringVoiceAgent:
        """Return the configured Agent."""
        return self._agent

    @property
    def resolver(self) -> VoiceActionResolver:
        """Return the configured voice action resolver."""
        return self._resolver

    def chat(
        self,
        user_input: str,
    ) -> VoiceActionResponse:
        """
        Resolve and execute one voice command through unified recovery.

        The method deliberately does not call Agent.chat().
        """
        if not isinstance(
            user_input,
            str,
        ):
            raise TypeError(
                "user_input must be a string."
            )

        command = user_input.strip()

        if not command:
            raise ValueError(
                "user_input cannot be empty."
            )

        try:
            request = self._resolver.resolve(
                command,
            )
        except AgentVoiceActionIntegrationError:
            raise
        except Exception as exc:
            raise AgentVoiceActionIntegrationError(
                f"Voice action resolution failed: {exc}"
            ) from exc

        if not isinstance(
            request,
            ActionRequest,
        ):
            raise AgentVoiceActionIntegrationError(
                "Voice action resolver returned an invalid ActionRequest."
            )

        voice_task_id = request.metadata.get(
            "voice_task_id"
        )

        task_id = (
            voice_task_id
            if isinstance(
                voice_task_id,
                str,
            ) and voice_task_id.strip()
            else request.request_id
        )

        run_id = f"voice-{uuid4().hex}"

        try:
            result = self._agent.execute_action_with_recovery(
                request,
                run_id=run_id,
                task_id=task_id,
            )
        except Exception as exc:
            raise AgentVoiceActionIntegrationError(
                f"Voice unified action execution failed: {exc}"
            ) from exc

        if not isinstance(
            result,
            RecoveryResult,
        ):
            raise AgentVoiceActionIntegrationError(
                "Agent returned an invalid recovery result."
            )

        return VoiceActionResponse(
            content=self._result_to_text(
                result
            )
        )

    @staticmethod
    def _result_to_text(
        result: RecoveryResult,
    ) -> str:
        """Convert a recovery result into text suitable for VoiceSession."""
        if not result.success:
            error = result.error

            if isinstance(
                error,
                str,
            ) and error.strip():
                return error.strip()

            return "The voice action failed."

        value: Any = result.result

        if value is None:
            return "Action completed."

        if isinstance(
            value,
            str,
        ):
            normalized = value.strip()

            return (
                normalized
                if normalized
                else "Action completed."
            )

        try:
            return dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            return str(value)


def create_agent_voice_action_adapter(
    agent: RecoveringVoiceAgent,
    task_action_resolver: TaskActionResolver,
) -> AgentVoiceActionAdapter:
    """Build the standard voice-to-unified-action adapter."""
    resolver = TaskActionResolverVoiceAdapter(
        task_action_resolver,
    )

    return AgentVoiceActionAdapter(
        agent,
        resolver,
    )


__all__ = [
    "AgentVoiceActionAdapter",
    "AgentVoiceActionIntegrationError",
    "RecoveringVoiceAgent",
    "TaskActionResolverVoiceAdapter",
    "VoiceActionResolver",
    "VoiceActionResponse",
    "create_agent_voice_action_adapter",
]
