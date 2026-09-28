from __future__ import annotations

import pytest

from osa.tasks import (
    TaskAction,
    TaskManager,
    TaskVerifier,
    VerificationResult,
)
from osa.tools import ToolResult


def create_task():
    from osa.planning import Plan, PlanStep

    manager = TaskManager()

    run = manager.create_run(
        Plan(
            goal="Verify task",
            steps=(
                PlanStep(
                    step_id="one",
                    description="Run one tool",
                ),
            ),
        )
    )

    return run.tasks["one"]


def test_verifier_requires_explicit_rule() -> None:
    verifier = TaskVerifier()
    task = create_task()

    result = verifier.verify(
        task,
        TaskAction(
            tool_name="echo",
            arguments={},
        ),
        ToolResult(
            success=True,
            output="hello",
        ),
        {},
    )

    assert result.verified is False
    assert "No verification rule" in result.message


def test_verifier_accepts_successful_rule() -> None:
    def verify(
        task,
        action,
        result,
        outputs,
    ) -> VerificationResult:
        assert result.output == "hello"

        return VerificationResult(
            verified=True,
            message="Output verified.",
        )

    verifier = TaskVerifier(
        {
            "echo": verify,
        }
    )

    result = verifier.verify(
        create_task(),
        TaskAction(
            tool_name="echo",
            arguments={},
        ),
        ToolResult(
            success=True,
            output="hello",
        ),
        {},
    )

    assert result.verified is True
    assert result.message == "Output verified."


def test_verifier_rejects_failed_tool_result() -> None:
    verifier = TaskVerifier(
        {
            "echo": lambda *args: VerificationResult(
                verified=True,
                message="should not run",
            )
        }
    )

    result = verifier.verify(
        create_task(),
        TaskAction(
            tool_name="echo",
            arguments={},
        ),
        ToolResult(
            success=False,
            output="",
            error="tool failed",
        ),
        {},
    )

    assert result.verified is False
    assert result.message == "tool failed"


def test_verification_result_requires_message() -> None:
    with pytest.raises(ValueError):
        VerificationResult(
            verified=True,
            message="",
        )

    with pytest.raises(ValueError):
        VerificationResult(
            verified=False,
            message="",
        )


def test_verifier_rejects_invalid_verifier_return() -> None:
    verifier = TaskVerifier(
        {
            "echo": lambda *args: True,
        }
    )

    with pytest.raises(
        RuntimeError,
        match="must return VerificationResult",
    ):
        verifier.verify(
            create_task(),
            TaskAction(
                tool_name="echo",
                arguments={},
            ),
            ToolResult(
                success=True,
                output="hello",
            ),
            {},
        )
