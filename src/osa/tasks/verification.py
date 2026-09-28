"""Verification primitives for OSA task execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from osa.tasks.action import TaskAction
from osa.tasks.manager import Task
from osa.tools import ToolResult


class TaskVerificationError(RuntimeError):
    """Raised when a task result cannot be verified."""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Result of verifying one task execution."""

    verified: bool
    message: str = ""

    def __post_init__(self) -> None:
        message = self.message.strip()

        if self.verified and not message:
            raise ValueError(
                "A successful verification requires a message."
            )

        if not self.verified and not message:
            raise ValueError(
                "A failed verification requires a message."
            )


TaskVerifierFunction = Callable[
    [Task, TaskAction, ToolResult, Mapping[str, str]],
    VerificationResult,
]


class TaskVerifier:
    """Verify task results using explicit tool-specific rules."""

    def __init__(
        self,
        rules: Mapping[str, TaskVerifierFunction] | None = None,
    ) -> None:
        self._rules = dict(rules or {})

    def register(
        self,
        tool_name: str,
        verifier: TaskVerifierFunction,
    ) -> None:
        """Register an explicit verifier for one tool."""
        normalized_name = tool_name.strip()

        if not normalized_name:
            raise ValueError(
                "tool_name cannot be empty."
            )

        if normalized_name in self._rules:
            raise ValueError(
                f"Verifier for tool '{normalized_name}' "
                "is already registered."
            )

        self._rules[normalized_name] = verifier

    def verify(
        self,
        task: Task,
        action: TaskAction,
        result: ToolResult,
        outputs: Mapping[str, str],
    ) -> VerificationResult:
        """Verify one tool result."""
        if not result.success:
            return VerificationResult(
                verified=False,
                message=(
                    result.error
                    or (
                        f"Tool '{action.tool_name}' "
                        "reported failure."
                    )
                ),
            )

        verifier = self._rules.get(
            action.tool_name
        )

        if verifier is None:
            return VerificationResult(
                verified=False,
                message=(
                    f"No verification rule is registered "
                    f"for tool '{action.tool_name}'."
                ),
            )

        try:
            verification = verifier(
                task,
                action,
                result,
                outputs,
            )
        except Exception as exc:
            raise TaskVerificationError(
                f"Verification for tool '{action.tool_name}' "
                f"failed: {exc}"
            ) from exc

        if not isinstance(
            verification,
            VerificationResult,
        ):
            raise TaskVerificationError(
                "Verifier must return VerificationResult."
            )

        return verification
