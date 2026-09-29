"""0.8.10.2 Autonomous executor and execution-context integration tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from osa.actions.contracts import ActionKind, ActionRequest
from osa.core.execution_context import (
    ExecutionContext,
    ExecutionContextError,
)
from osa.tasks.action_bridge import AutonomousActionResolver
from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_executor import UnifiedAutonomousExecutor


@dataclass
class FixedResolver:
    """Return a deterministic autonomous ActionRequest."""

    request: ActionRequest
    expected_run_id: str | None = None
    expected_task_id: str | None = None

    def resolve(
        self,
        run_id: str,
        task_id: str,
    ) -> ActionRequest:
        if self.expected_run_id is not None:
            assert run_id == self.expected_run_id

        if self.expected_task_id is not None:
            assert task_id == self.expected_task_id

        return self.request


class RecordingBridge:
    """Capture the exact request and authoritative autonomous scope."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                str,
                str,
                ActionRequest,
            ]
        ] = []

    def execute(
        self,
        run_id: str,
        task_id: str,
        request: ActionRequest,
    ) -> AutonomousTaskResult:
        self.calls.append(
            (
                run_id,
                task_id,
                request,
            )
        )

        return AutonomousTaskResult(
            task_id=task_id,
            status="completed",
            output="42",
        )


def _request(
    *,
    request_id: str = "autonomous-action-010",
    metadata: dict | None = None,
) -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={
            "value": 42,
        },
        request_id=request_id,
        metadata={} if metadata is None else metadata,
    )


def test_unified_autonomous_executor_validates_canonical_context() -> None:
    request = _request(
        metadata={
            "custom": "value",
        }
    )

    resolver: AutonomousActionResolver = FixedResolver(
        request,
        expected_run_id="run-010",
        expected_task_id="task-010",
    )
    bridge = RecordingBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-010",
        "task-010",
    )

    assert result.status == "completed"
    assert result.task_id == "task-010"

    assert len(bridge.calls) == 1

    run_id, task_id, forwarded = bridge.calls[0]

    assert run_id == "run-010"
    assert task_id == "task-010"
    assert forwarded is request

    context = ExecutionContext.from_autonomous_action_request(
        forwarded,
        run_id=run_id,
        task_id=task_id,
    )

    assert context.request_id == (
        "autonomous-action-010"
    )
    assert context.run_id == "run-010"
    assert context.task_id == "task-010"
    assert context.source == "autonomous"
    assert context.round_number is None
    assert context.metadata == {
        "custom": "value",
    }


def test_autonomous_context_is_not_written_into_action_request() -> None:
    request = _request()

    resolver = FixedResolver(
        request,
        expected_run_id="run-011",
        expected_task_id="task-011",
    )
    bridge = RecordingBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-011",
        "task-011",
    )

    assert result.status == "completed"

    forwarded = bridge.calls[0][2]

    assert forwarded is request
    assert forwarded.metadata == {}


def test_invalid_autonomous_round_fails_before_bridge() -> None:
    request = _request(
        request_id="autonomous-invalid-round-010",
        metadata={
            "round": 1,
        },
    )

    resolver = FixedResolver(
        request,
        expected_run_id="run-012",
        expected_task_id="task-012",
    )
    bridge = RecordingBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-012",
        "task-012",
    )

    assert result.status == "failed"
    assert result.task_id == "task-012"
    assert result.error is not None
    assert result.error.startswith(
        "autonomous_action_context_invalid:"
    )

    assert bridge.calls == []


def test_request_identity_remains_authoritative() -> None:
    request = _request(
        request_id="autonomous-authoritative-010",
        metadata={
            "request_id": "spoofed-request",
            "source": "voice",
        },
    )

    resolver = FixedResolver(
        request,
        expected_run_id="run-013",
        expected_task_id="task-013",
    )
    bridge = RecordingBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "run-013",
        "task-013",
    )

    assert result.status == "completed"

    forwarded = bridge.calls[0][2]

    assert forwarded.request_id == (
        "autonomous-authoritative-010"
    )

    assert forwarded.metadata == {
        "request_id": "spoofed-request",
        "source": "voice",
    }

    context = ExecutionContext.from_autonomous_action_request(
        forwarded,
        run_id="run-013",
        task_id="task-013",
    )

    assert context.request_id == (
        "autonomous-authoritative-010"
    )
    assert context.source == "autonomous"
    assert context.metadata == {}


def test_autonomous_context_rejects_round_metadata() -> None:
    request = _request(
        request_id="autonomous-correlation-010",
        metadata={
            "request_id": "spoofed-request",
            "run_id": "metadata-run",
            "task_id": "metadata-task",
            "source": "voice",
            "round": 1,
            "custom": "preserve-me",
        },
    )

    with pytest.raises(
        ExecutionContextError,
        match="round_number",
    ):
        ExecutionContext.from_autonomous_action_request(
            request,
            run_id="run-014",
            task_id="task-014",
        )

    assert request.metadata == {
        "request_id": "spoofed-request",
        "run_id": "metadata-run",
        "task_id": "metadata-task",
        "source": "voice",
        "round": 1,
        "custom": "preserve-me",
    }


def test_autonomous_source_is_authoritative() -> None:
    request = _request(
        metadata={
            "source": "voice",
            "custom": "preserve-me",
        }
    )

    context = ExecutionContext.from_autonomous_action_request(
        request,
        run_id="run-015",
        task_id="task-015",
    )

    assert context.source == "autonomous"
    assert context.run_id == "run-015"
    assert context.task_id == "task-015"
    assert context.round_number is None
    assert context.metadata == {
        "custom": "preserve-me",
    }


def test_normalized_scope_reaches_bridge() -> None:
    request = _request(
        request_id="autonomous-normalized-010",
    )

    resolver = FixedResolver(
        request,
        expected_run_id="run-normalized",
        expected_task_id="task-normalized",
    )
    bridge = RecordingBridge()

    executor = UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )

    result = executor.execute(
        "  run-normalized  ",
        "  task-normalized  ",
    )

    assert result.status == "completed"

    assert bridge.calls == [
        (
            "run-normalized",
            "task-normalized",
            request,
        )
    ]
