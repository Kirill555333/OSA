"""Recovery integration for unified autonomous action execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_executor import UnifiedAutonomousExecutor


@dataclass(frozen=True)
class AutonomousRecoveryDecision:
    """Decision returned by an autonomous recovery policy."""

    retry: bool
    reason: str | None = None


class AutonomousRecoveryPolicy(Protocol):
    """Decide whether a failed autonomous task should be retried."""

    def decide(
        self,
        run_id: str,
        task_id: str,
        result: AutonomousTaskResult,
        attempt: int,
    ) -> AutonomousRecoveryDecision:
        ...


class NeverRetryAutonomousRecoveryPolicy:
    """Default fail-closed recovery policy."""

    def decide(
        self,
        run_id: str,
        task_id: str,
        result: AutonomousTaskResult,
        attempt: int,
    ) -> AutonomousRecoveryDecision:
        return AutonomousRecoveryDecision(
            retry=False,
            reason="autonomous_recovery_disabled",
        )


class CallbackAutonomousRecoveryPolicy:
    """Callback-backed recovery policy."""

    def __init__(
        self,
        callback,
    ) -> None:
        if not callable(callback):
            raise TypeError(
                "callback must be callable."
            )

        self._callback = callback

    def decide(
        self,
        run_id: str,
        task_id: str,
        result: AutonomousTaskResult,
        attempt: int,
    ) -> AutonomousRecoveryDecision:
        return self._callback(
            run_id,
            task_id,
            result,
            attempt,
        )


class AutonomousRecoveryExecutor:
    """
    Execute autonomous tasks with bounded, policy-controlled recovery.

    Only failed results are eligible for retry. Successful results return
    immediately and are never executed again.
    """

    def __init__(
        self,
        executor: UnifiedAutonomousExecutor,
        *,
        policy: AutonomousRecoveryPolicy | None = None,
        max_attempts: int = 1,
    ) -> None:
        if executor is None:
            raise ValueError(
                "executor is required."
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )

        self._executor = executor
        self._policy = (
            policy
            or NeverRetryAutonomousRecoveryPolicy()
        )
        self._max_attempts = max_attempts

    @property
    def executor(self) -> UnifiedAutonomousExecutor:
        """Return the underlying unified autonomous executor."""
        return self._executor

    @property
    def policy(self) -> AutonomousRecoveryPolicy:
        """Return the configured recovery policy."""
        return self._policy

    @property
    def max_attempts(self) -> int:
        """Return the maximum number of execution attempts."""
        return self._max_attempts

    def execute(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousTaskResult:
        """
        Execute a task and apply bounded recovery after failures.
        """
        normalized_run_id = self._normalize_identifier(
            run_id,
            "run_id",
        )
        normalized_task_id = self._normalize_identifier(
            task_id,
            "task_id",
        )

        last_result: AutonomousTaskResult | None = None

        for attempt in range(
            1,
            self._max_attempts + 1,
        ):
            result = self._executor.execute(
                normalized_run_id,
                normalized_task_id,
            )
            last_result = result

            if result.status == "completed":
                return result

            if result.status != "failed":
                return AutonomousTaskResult(
                    task_id=normalized_task_id,
                    status="failed",
                    output=result.output,
                    error=(
                        "autonomous_recovery_invalid_result: "
                        f"unsupported task status '{result.status}'."
                    ),
                )

            if attempt >= self._max_attempts:
                return result

            try:
                decision = self._policy.decide(
                    normalized_run_id,
                    normalized_task_id,
                    result,
                    attempt,
                )
            except Exception as exc:
                return AutonomousTaskResult(
                    task_id=normalized_task_id,
                    status="failed",
                    output=result.output,
                    error=(
                        "autonomous_recovery_policy_failed: "
                        f"{exc}"
                    ),
                )

            if not isinstance(
                decision,
                AutonomousRecoveryDecision,
            ):
                return AutonomousTaskResult(
                    task_id=normalized_task_id,
                    status="failed",
                    output=result.output,
                    error=(
                        "autonomous_recovery_policy_failed: "
                        "invalid recovery decision."
                    ),
                )

            if not decision.retry:
                return result

        if last_result is not None:
            return last_result

        return AutonomousTaskResult(
            task_id=normalized_task_id,
            status="failed",
            error="autonomous_recovery_no_result",
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
