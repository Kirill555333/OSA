"""Bridge autonomous task actions into the unified action pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from osa.actions.contracts import (
    ActionRequest,
    ActionResult,
)
from osa.actions.pipeline import ActionSafetyPipeline
from osa.tasks.autonomous import AutonomousTaskResult


class AutonomousActionBridgeError(RuntimeError):
    """Raised when an autonomous action cannot be bridged safely."""


class AutonomousActionResolver(Protocol):
    """Resolve an autonomous task into one unified action request."""

    def resolve(
        self,
        run_id: str,
        task_id: str,
    ) -> ActionRequest:
        ...


class CallbackAutonomousActionResolver:
    """Callback-backed resolver for deterministic dependency injection."""

    def __init__(
        self,
        callback,
    ) -> None:
        if not callable(callback):
            raise TypeError(
                "callback must be callable."
            )

        self._callback = callback

    def resolve(
        self,
        run_id: str,
        task_id: str,
    ) -> ActionRequest:
        return self._callback(
            run_id,
            task_id,
        )


class AutonomousActionBridge:
    """
    Convert autonomous task execution into unified action execution.

    The bridge does not implement authorization or routing itself.
    Those responsibilities remain inside ActionSafetyPipeline.
    """

    def __init__(
        self,
        pipeline: ActionSafetyPipeline,
    ) -> None:
        if pipeline is None:
            raise ValueError(
                "pipeline is required."
            )

        self._pipeline = pipeline

    @property
    def pipeline(self) -> ActionSafetyPipeline:
        """Return the unified safety pipeline."""
        return self._pipeline

    def execute(
        self,
        run_id: str,
        task_id: str,
        request: ActionRequest,
    ) -> AutonomousTaskResult:
        """
        Execute one autonomous action through the unified pipeline.
        """
        normalized_run_id = self._normalize_identifier(
            run_id,
            "run_id",
        )
        normalized_task_id = self._normalize_identifier(
            task_id,
            "task_id",
        )

        if not isinstance(request, ActionRequest):
            raise AutonomousActionBridgeError(
                "request must be an ActionRequest."
            )

        try:
            result = self._dispatch_pipeline(
                request
            )
        except Exception as exc:
            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                error=(
                    "autonomous_action_execution_failed: "
                    f"{exc}"
                ),
            )

        if not isinstance(result, ActionResult):
            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                error=(
                    "autonomous_action_invalid_result: "
                    "pipeline returned an invalid result type."
                ),
            )

        if result.request_id != request.request_id:
            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="failed",
                error=(
                    "autonomous_action_invalid_result: "
                    "request_id mismatch."
                ),
            )

        if result.success:
            return AutonomousTaskResult(
                task_id=normalized_task_id,
                status="completed",
                output=result.output,
            )

        return AutonomousTaskResult(
            task_id=normalized_task_id,
            status="failed",
            output=result.output or None,
            error=(
                result.error
                or "autonomous_action_failed"
            ),
        )

    def _dispatch_pipeline(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        """
        Use the canonical pipeline entry point.

        The execute() fallback preserves compatibility with older test
        doubles and injected pipeline-like objects.
        """
        dispatch = getattr(
            self._pipeline,
            "dispatch",
            None,
        )

        if callable(dispatch):
            return dispatch(request)

        execute = getattr(
            self._pipeline,
            "execute",
            None,
        )

        if callable(execute):
            return execute(request)

        raise AutonomousActionBridgeError(
            "pipeline must provide dispatch()."
        )

    @staticmethod
    def _normalize_identifier(
        value: str,
        field_name: str,
    ) -> str:
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must be a string."
            )

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized
