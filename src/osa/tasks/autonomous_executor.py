"""Unified autonomous task executor for the OSA action pipeline."""

from __future__ import annotations

from osa.actions.contracts import ActionRequest
from osa.core.execution_context import (
    ExecutionContext,
    ExecutionContextError,
)
from osa.tasks.action_bridge import (
    AutonomousActionBridge,
    AutonomousActionResolver,
)
from osa.tasks.autonomous import AutonomousTaskResult


class UnifiedAutonomousExecutorError(RuntimeError):
    """Raised when the unified autonomous executor is misconfigured."""


class UnifiedAutonomousExecutor:
    """
    Resolve and execute autonomous tasks through the unified action pipeline.

    The executor owns composition only:
        task -> ActionRequest -> ExecutionContext -> ActionSafetyPipeline
        -> task result
    """

    def __init__(
        self,
        resolver: AutonomousActionResolver,
        bridge: AutonomousActionBridge,
    ) -> None:
        if resolver is None:
            raise ValueError(
                "resolver is required."
            )

        if bridge is None:
            raise ValueError(
                "bridge is required."
            )

        self._resolver = resolver
        self._bridge = bridge

    @property
    def resolver(self) -> AutonomousActionResolver:
        """Return the configured autonomous action resolver."""
        return self._resolver

    @property
    def bridge(self) -> AutonomousActionBridge:
        """Return the configured autonomous action bridge."""
        return self._bridge

    def execute(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        """
        Resolve one task into an ActionRequest and execute it through the
        canonical autonomous execution context.
        """
        normalized_run_id = self._normalize_identifier(
            run_id,
            "run_id",
        )
        normalized_task_id = self._normalize_identifier(
            task_id,
            "task_id",
        )

        try:
            request = self._resolver.resolve(
                normalized_run_id,
                normalized_task_id,
            )
        except Exception as exc:
            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                error=(
                    "autonomous_action_resolution_failed: "
                    f"{exc}"
                ),
            )

        if not isinstance(
            request,
            ActionRequest,
        ):
            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                error=(
                    "autonomous_action_resolution_failed: "
                    "resolver returned an invalid ActionRequest."
                ),
            )

        try:
            context = ExecutionContext.from_autonomous_action_request(
                request,
                run_id=normalized_run_id,
                task_id=normalized_task_id,
            )
        except ExecutionContextError as exc:
            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                error=(
                    "autonomous_action_context_invalid: "
                    f"{exc}"
                ),
            )

        return self._bridge.execute(
            context.run_id,
            context.task_id,
            request,
        )

    @staticmethod
    def _normalize_identifier(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(
            value,
            str,
        ):
            raise TypeError(
                f"{field_name} must be a string."
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized


__all__ = [
    "UnifiedAutonomousExecutor",
    "UnifiedAutonomousExecutorError",
]
