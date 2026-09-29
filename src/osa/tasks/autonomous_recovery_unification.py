from __future__ import annotations

from typing import Protocol

from osa.recovery_contracts import (
    RecoveryDecision,
    RecoveryFailureKind,
    RecoveryRequest,
    RecoveryResult,
)
from osa.recovery_controller import RecoveryController
from osa.recovery_policy import RecoveryPolicy

from osa.tasks.autonomous import (
    AutonomousBackend,
    AutonomousTaskResult,
)


class AutonomousRecoveryUnificationError(RuntimeError):
    """Raised when unified autonomous recovery cannot be completed safely."""


class AutonomousRecoveryAuthorizer(Protocol):
    """Authorize an autonomous retry immediately before execution."""

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ):
        ...


class SafetyAwareAutonomousRecoveryPolicy:
    """
    Combine recovery policy with a fresh autonomous safety authorization.

    A retry is allowed only when both the recovery policy and the current
    autonomous authorization allow it.
    """

    def __init__(
        self,
        policy: RecoveryPolicy,
        authorizer: AutonomousRecoveryAuthorizer,
    ) -> None:
        if policy is None:
            raise ValueError("policy is required.")

        if authorizer is None:
            raise ValueError("authorizer is required.")

        self._policy = policy
        self._authorizer = authorizer

    @property
    def policy(self) -> RecoveryPolicy:
        return self._policy

    @property
    def authorizer(self) -> AutonomousRecoveryAuthorizer:
        return self._authorizer

    def decide(
        self,
        request: RecoveryRequest,
        attempt,
    ) -> RecoveryDecision:
        decision = self._policy.decide(
            request,
            attempt,
        )

        if not isinstance(
            decision,
            RecoveryDecision,
        ):
            raise AutonomousRecoveryUnificationError(
                "Recovery policy returned an invalid decision."
            )

        if not decision.retry:
            return decision

        try:
            authorization = self._authorizer.authorize(
                request.run_id,
                request.task_id,
            )
        except Exception as exc:
            return RecoveryDecision(
                retry=False,
                reason=(
                    "Autonomous retry authorization failed: "
                    f"{exc}"
                ),
                failure_kind=(
                    RecoveryFailureKind.PERMISSION_DENIED
                ),
                delay_seconds=0.0,
                next_attempt=None,
            )

        if not getattr(
            authorization,
            "allowed",
            False,
        ):
            return RecoveryDecision(
                retry=False,
                reason=(
                    getattr(
                        authorization,
                        "reason",
                        None,
                    )
                    or "Autonomous retry was denied."
                ),
                failure_kind=(
                    RecoveryFailureKind.PERMISSION_DENIED
                ),
                delay_seconds=0.0,
                next_attempt=None,
            )

        return decision


class UnifiedAutonomousRecoveryBackend:
    """
    Adapt the existing AutonomousBackend to the unified RecoveryController.

    Recovery never calls a concrete execution backend directly. It re-enters
    the same autonomous task boundary through the injected controller.
    """

    def __init__(
        self,
        backend: AutonomousBackend,
        controller: RecoveryController,
        *,
        max_attempts: int = 3,
    ) -> None:
        if backend is None:
            raise ValueError("backend is required.")

        if controller is None:
            raise ValueError("controller is required.")

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        self._backend = backend
        self._controller = controller
        self._max_attempts = max_attempts

    @property
    def backend(self) -> AutonomousBackend:
        return self._backend

    @property
    def controller(self) -> RecoveryController:
        return self._controller

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def create_run(
        self,
        goal: str,
    ) -> str:
        return self._backend.create_run(goal)

    def ready_task_ids(
        self,
        run_id: str,
    ):
        return self._backend.ready_task_ids(run_id)

    def execute_task(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        if not isinstance(
            run_id,
            str,
        ) or not run_id.strip():
            raise AutonomousRecoveryUnificationError(
                "run_id must be a non-empty string."
            )

        if not isinstance(
            task_id,
            str,
        ) or not task_id.strip():
            raise AutonomousRecoveryUnificationError(
                "task_id must be a non-empty string."
            )

        request = RecoveryRequest(
            run_id=run_id,
            task_id=task_id,
            attempt=1,
            max_attempts=self._max_attempts,
            metadata={
                "source": "autonomous",
                "task_id": task_id,
            },
        )

        try:
            recovery_result = self._controller.execute(
                request
            )
        except Exception as exc:
            raise AutonomousRecoveryUnificationError(
                "Unified autonomous recovery failed: "
                f"{exc}"
            ) from exc

        if not isinstance(
            recovery_result,
            RecoveryResult,
        ):
            raise AutonomousRecoveryUnificationError(
                "RecoveryController returned an invalid result."
            )

        result = recovery_result.result

        if isinstance(
            result,
            AutonomousTaskResult,
        ):
            return result

        if recovery_result.success:
            raise AutonomousRecoveryUnificationError(
                "RecoveryController reported success without "
                "an AutonomousTaskResult."
            )

        return AutonomousTaskResult(
            task_id=task_id,
            status="failed",
            error=(
                recovery_result.error
                or "Unified autonomous recovery failed."
            ),
        )

    def is_complete(
        self,
        run_id: str,
    ) -> bool:
        return self._backend.is_complete(run_id)

    def has_failed(
        self,
        run_id: str,
    ) -> bool:
        return self._backend.has_failed(run_id)
