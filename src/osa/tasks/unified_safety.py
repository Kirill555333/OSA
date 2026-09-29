"""Safety integration for autonomous execution."""

from __future__ import annotations

from osa.tasks.autonomous import AutonomousTaskAuthorizer
from osa.tasks.safety import (
    AutonomousAuthorization,
    AutonomousSafetyGate,
)


class UnifiedAutonomousSafetyError(RuntimeError):
    """Raised when autonomous safety integration is misconfigured."""


class UnifiedAutonomousTaskAuthorizer(
    AutonomousTaskAuthorizer
):
    """
    Adapt the existing autonomous safety gate to the unified executor path.

    The task-level safety gate runs before task execution.
    The ActionSafetyPipeline remains responsible for final action-level
    mode, permission, safety, and confirmation checks.
    """

    def __init__(
        self,
        safety_gate: AutonomousSafetyGate,
    ) -> None:
        if safety_gate is None:
            raise ValueError(
                "safety_gate is required."
            )

        if not hasattr(
            safety_gate,
            "authorize",
        ):
            raise TypeError(
                "safety_gate must provide authorize()."
            )

        self._safety_gate = safety_gate

    @property
    def safety_gate(self) -> AutonomousSafetyGate:
        """Return the underlying autonomous safety gate."""
        return self._safety_gate

    def authorize(
        self,
        run_id: str,
        task_id: str,
    ) -> AutonomousAuthorization:
        normalized_run_id = self._normalize_identifier(
            run_id,
            "run_id",
        )
        normalized_task_id = self._normalize_identifier(
            task_id,
            "task_id",
        )

        try:
            authorization = self._safety_gate.authorize(
                normalized_run_id,
                normalized_task_id,
            )
        except Exception as exc:
            return AutonomousAuthorization(
                allowed=False,
                reason=(
                    "autonomous_safety_gate_error: "
                    f"{exc}"
                ),
            )

        if not isinstance(
            authorization,
            AutonomousAuthorization,
        ):
            return AutonomousAuthorization(
                allowed=False,
                reason=(
                    "autonomous_safety_gate_error: "
                    "invalid authorization result."
                ),
            )

        if not authorization.allowed:
            return authorization

        return AutonomousAuthorization(
            allowed=True,
            reason=authorization.reason,
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
