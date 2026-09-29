from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from osa.actions.contracts import ActionRequest
from osa.recovery_contracts import (
    RecoveryAttempt,
    RecoveryFailureKind,
    RecoveryRequest,
    RecoveryResult,
)
from osa.recovery_controller import RecoveryController
from osa.recovery_policy import RecoveryPolicy


class RecoveryActionIntegrationError(RuntimeError):
    """Raised when action recovery integration is invalid."""


class ActionExecutionCallable(Protocol):
    """Existing unified action execution boundary."""

    def __call__(
        self,
        request: ActionRequest,
    ) -> Any:
        ...


class ActionResultClassifier(Protocol):
    """Convert an action result into a recovery attempt."""

    def __call__(
        self,
        request: RecoveryRequest,
        result: Any,
    ) -> RecoveryAttempt:
        ...


class DefaultActionResultClassifier:
    """
    Conservative classifier for ActionResult-like objects.

    A valid result must expose a boolean ``success`` attribute.

    Generic failed results are classified as BACKEND_ERROR.

    Unified ActionSafetyPipeline failures preserve their semantic safety
    category when the pipeline exposes its stable metadata:
    - permission + denied -> PERMISSION_DENIED
    - confirmation + not_confirmed -> CONFIRMATION_DENIED
    - policy/configuration denials -> NON_RETRYABLE
    """

    def __call__(
        self,
        request: RecoveryRequest,
        result: Any,
    ) -> RecoveryAttempt:
        success = getattr(
            result,
            "success",
            None,
        )

        if not isinstance(
            success,
            bool,
        ):
            raise RecoveryActionIntegrationError(
                "action result must expose a boolean "
                "'success' attribute"
            )

        output = getattr(
            result,
            "output",
            None,
        )

        error = getattr(
            result,
            "error",
            None,
        )

        if error is not None and not isinstance(
            error,
            str,
        ):
            error = str(error)

        failure_kind = (
            None
            if success
            else self._failure_kind(
                result
            )
        )

        return RecoveryAttempt(
            attempt=request.attempt,
            max_attempts=request.max_attempts,
            success=success,
            failure_kind=failure_kind,
            error=(
                None
                if success
                else (
                    error
                    or "action_execution_failed"
                )
            ),
            output_present=(
                output is not None
            ),
        )

    @staticmethod
    def _failure_kind(
        result: Any,
    ) -> RecoveryFailureKind:
        """Map stable unified pipeline metadata to recovery semantics."""
        metadata = getattr(
            result,
            "metadata",
            None,
        )

        if not isinstance(
            metadata,
            Mapping,
        ):
            return RecoveryFailureKind.BACKEND_ERROR

        stage = metadata.get(
            "pipeline_stage"
        )
        code = metadata.get(
            "pipeline_error"
        )

        if (
            stage == "permission"
            and code == "denied"
        ):
            return RecoveryFailureKind.PERMISSION_DENIED

        if (
            stage == "confirmation"
            and code == "not_confirmed"
        ):
            return RecoveryFailureKind.CONFIRMATION_DENIED

        if stage in {
            "mode",
            "safety",
            "confirmation",
        }:
            return RecoveryFailureKind.NON_RETRYABLE

        if code in {
            "policy_exception",
            "invalid_policy_result",
            "confirmation_not_configured",
            "confirmation_exception",
        }:
            return RecoveryFailureKind.NON_RETRYABLE

        return RecoveryFailureKind.BACKEND_ERROR


class RecoverableActionExecutor:
    """
    Recovery wrapper around the existing unified action executor.

    Recovery never calls Browser, Desktop, or Tool backends directly.
    Every retry re-enters the injected unified action executor.
    """

    def __init__(
        self,
        action_executor: ActionExecutionCallable,
        *,
        max_attempts: int = 1,
        policy: RecoveryPolicy | None = None,
        classifier: ActionResultClassifier | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if not callable(action_executor):
            raise RecoveryActionIntegrationError(
                "action_executor must be callable"
            )

        if max_attempts < 1:
            raise RecoveryActionIntegrationError(
                "max_attempts must be at least 1"
            )

        if classifier is None:
            classifier = DefaultActionResultClassifier()

        if not callable(classifier):
            raise RecoveryActionIntegrationError(
                "classifier must be callable"
            )

        self._action_executor = action_executor
        self._max_attempts = max_attempts
        self._classifier = classifier

        self._controller = RecoveryController(
            self._execute_action,
            attempt_builder=self._build_attempt,
            policy=policy,
            sleeper=sleeper,
        )

    @property
    def action_executor(self) -> ActionExecutionCallable:
        return self._action_executor

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    @property
    def classifier(self) -> ActionResultClassifier:
        return self._classifier

    @property
    def controller(self) -> RecoveryController:
        return self._controller

    def execute(
        self,
        request: ActionRequest,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> RecoveryResult:
        if not isinstance(
            request,
            ActionRequest,
        ):
            raise RecoveryActionIntegrationError(
                "request must be an ActionRequest"
            )

        recovery_request = RecoveryRequest(
            run_id=run_id,
            task_id=task_id,
            request_id=request.request_id,
            action_kind=request.kind,
            action_name=request.name,
            attempt=1,
            max_attempts=self._max_attempts,
            metadata=self._recovery_metadata(
                request
            ),
        )

        return self._controller.execute(
            recovery_request
        )

    def _execute_action(
        self,
        recovery_request: RecoveryRequest,
    ) -> Any:
        request = recovery_request.metadata.get(
            "_action_request"
        )

        if not isinstance(
            request,
            ActionRequest,
        ):
            raise RecoveryActionIntegrationError(
                "recovery context does not contain "
                "the original ActionRequest"
            )

        return self._action_executor(
            request
        )

    def _build_attempt(
        self,
        recovery_request: RecoveryRequest,
        result: Any,
    ) -> RecoveryAttempt:
        return self._classifier(
            recovery_request,
            result,
        )

    @staticmethod
    def _recovery_metadata(
        request: ActionRequest,
    ) -> dict[str, Any]:
        metadata = dict(
            getattr(
                request,
                "metadata",
                {},
            )
            or {}
        )

        metadata["_action_request"] = request

        return metadata
