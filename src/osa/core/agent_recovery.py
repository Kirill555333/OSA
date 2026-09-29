from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol
from uuid import uuid4

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
)
from osa.recovery_contracts import (
    RecoveryResult,
)


class AgentRecoveryIntegrationError(RuntimeError):
    """Raised when agent recovery integration is invalid."""


class AgentRecoveryExecutor(Protocol):
    """Unified recovery executor used by the Agent integration layer."""

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        ...


class AgentRecoveryIntegration:
    """
    Agent-facing integration seam for the unified recovery pipeline.

    The integration does not execute actions itself. It delegates to the
    injected unified recovery executor, which owns recovery, safety, routing,
    execution, and verification.
    """

    def __init__(
        self,
        executor: AgentRecoveryExecutor,
    ) -> None:
        if executor is None:
            raise ValueError(
                "executor is required."
            )

        self._executor = executor

    @property
    def executor(self) -> AgentRecoveryExecutor:
        """Return the configured unified recovery executor."""
        return self._executor

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        """
        Execute an ActionRequest through unified recovery.

        The original request is delegated unchanged. No backend is called
        directly by this integration layer.
        """
        if not isinstance(
            request,
            ActionRequest,
        ):
            raise TypeError(
                "request must be an ActionRequest."
            )

        result = self._executor.execute(
            request,
            run_id=run_id,
            task_id=task_id,
        )

        if not isinstance(
            result,
            RecoveryResult,
        ):
            raise AgentRecoveryIntegrationError(
                "Unified recovery executor returned "
                "an invalid result."
            )

        return result

    def execute_tool(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        request_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        """
        Build a TOOL ActionRequest and execute it through unified recovery.

        A request ID is generated only when the caller does not provide one.
        """
        if not isinstance(
            tool_name,
            str,
        ):
            raise TypeError(
                "tool_name must be a string."
            )

        normalized_tool_name = tool_name.strip()

        if not normalized_tool_name:
            raise ValueError(
                "tool_name cannot be empty."
            )

        if not isinstance(
            arguments,
            Mapping,
        ):
            raise TypeError(
                "arguments must be a mapping."
            )

        normalized_request_id = (
            request_id
            if request_id is not None
            else str(uuid4())
        )

        if not isinstance(
            normalized_request_id,
            str,
        ):
            raise TypeError(
                "request_id must be a string or None."
            )

        normalized_request_id = (
            normalized_request_id.strip()
        )

        if not normalized_request_id:
            raise ValueError(
                "request_id cannot be empty."
            )

        request = ActionRequest(
            request_id=normalized_request_id,
            kind=ActionKind.TOOL,
            name=normalized_tool_name,
            arguments=dict(arguments),
        )

        return self.execute(
            request,
            run_id=run_id,
            task_id=task_id,
        )
