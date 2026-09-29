from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

from osa.actions.contracts import ActionRequest, ActionResult
from osa.observability.context import ObservabilityContext
from osa.observability.events import ObservabilityEvent
from osa.observability.logger import (
    InMemoryObservabilityLogger,
    ObservabilityLogger,
)


class ActionPipelineLike(Protocol):
    """Minimal contract required from an action safety pipeline."""

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        ...


class ObservableActionSafetyPipeline:
    """
    Observability wrapper around the existing action safety pipeline.

    The wrapped pipeline remains responsible for all policy, confirmation,
    routing and execution decisions. This class only records lifecycle events.
    """

    def __init__(
        self,
        pipeline: ActionPipelineLike,
        *,
        logger: ObservabilityLogger | None = None,
        context_provider: (
            Callable[[ActionRequest], ObservabilityContext] | None
        ) = None,
    ) -> None:
        if pipeline is None:
            raise ValueError(
                "pipeline cannot be None."
            )

        if not hasattr(
            pipeline,
            "execute",
        ):
            raise ValueError(
                "pipeline must provide execute(request)."
            )

        self._pipeline = pipeline

        if logger is None:
            logger = InMemoryObservabilityLogger()

        self._logger = logger

        self._context_provider = (
            context_provider
            if context_provider is not None
            else self._default_context_provider
        )

    @property
    def pipeline(self) -> ActionPipelineLike:
        """Return the wrapped action pipeline."""
        return self._pipeline

    @property
    def logger(self) -> ObservabilityLogger:
        """Return the configured observability logger."""
        return self._logger

    def execute(
        self,
        request: ActionRequest,
    ) -> ActionResult:
        if not isinstance(
            request,
            ActionRequest,
        ):
            raise TypeError(
                "request must be an ActionRequest."
            )

        context = self._context_provider(
            request
        )

        if not isinstance(
            context,
            ObservabilityContext,
        ):
            raise TypeError(
                "context_provider must return an "
                "ObservabilityContext."
            )

        started_at = time.perf_counter()

        self._safe_log(
            ObservabilityEvent(
                event="action.started",
                source="action_pipeline",
                request_id=request.request_id,
                run_id=context.run_id,
                task_id=context.task_id,
                action_kind=request.kind,
                action_name=request.name,
                status="started",
                metadata={
                    "arguments_present": bool(
                        request.arguments
                    ),
                },
            )
        )

        try:
            result = self._pipeline.execute(
                request
            )
        except Exception as exc:
            self._safe_log(
                ObservabilityEvent(
                    event="action.failed",
                    source="action_pipeline",
                    request_id=request.request_id,
                    run_id=context.run_id,
                    task_id=context.task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=self._duration_ms(
                        started_at
                    ),
                    error=str(exc),
                )
            )
            raise

        if not isinstance(
            result,
            ActionResult,
        ):
            self._safe_log(
                ObservabilityEvent(
                    event="action.failed",
                    source="action_pipeline",
                    request_id=request.request_id,
                    run_id=context.run_id,
                    task_id=context.task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=self._duration_ms(
                        started_at
                    ),
                    error="invalid_action_result",
                )
            )
            return ActionResult.failed(
                request.request_id,
                error="invalid_action_result",
            )

        if result.request_id != request.request_id:
            self._safe_log(
                ObservabilityEvent(
                    event="action.failed",
                    source="action_pipeline",
                    request_id=request.request_id,
                    run_id=context.run_id,
                    task_id=context.task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=self._duration_ms(
                        started_at
                    ),
                    error="action_result_request_id_mismatch",
                )
            )
            return ActionResult.failed(
                request.request_id,
                error="action_result_request_id_mismatch",
            )

        if result.success:
            self._safe_log(
                ObservabilityEvent(
                    event="action.completed",
                    source="action_pipeline",
                    request_id=request.request_id,
                    run_id=context.run_id,
                    task_id=context.task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="completed",
                    duration_ms=self._duration_ms(
                        started_at
                    ),
                    metadata={
                        "output_present": bool(
                            result.output
                        ),
                    },
                )
            )
        else:
            self._safe_log(
                ObservabilityEvent(
                    event="action.failed",
                    source="action_pipeline",
                    request_id=request.request_id,
                    run_id=context.run_id,
                    task_id=context.task_id,
                    action_kind=request.kind,
                    action_name=request.name,
                    status="failed",
                    duration_ms=self._duration_ms(
                        started_at
                    ),
                    error=result.error,
                )
            )

        return result

    @staticmethod
    def _default_context_provider(
        request: ActionRequest,
    ) -> ObservabilityContext:
        metadata = request.metadata

        return ObservabilityContext(
            run_id=metadata.get(
                "run_id"
            ),
            task_id=metadata.get(
                "task_id"
            ),
            request_id=request.request_id,
        )

    def _safe_log(
        self,
        event: ObservabilityEvent,
    ) -> None:
        """
        Never allow observability failure to alter action execution.

        Logging is deliberately best-effort.
        """
        try:
            self._logger.log(
                event
            )
        except Exception:
            return None

    @staticmethod
    def _duration_ms(
        started_at: float,
    ) -> float:
        return round(
            (
                time.perf_counter()
                - started_at
            )
            * 1000,
            3,
        )


__all__ = [
    "ActionPipelineLike",
    "ObservableActionSafetyPipeline",
]
