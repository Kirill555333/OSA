"""0.8.10.4 Autonomous recovery execution-context propagation tests."""

from __future__ import annotations

from dataclasses import dataclass

from osa.actions.contracts import (
    ActionKind,
    ActionRequest,
    ActionResult,
)
from osa.core.execution_context import ExecutionContext
from osa.recovery_contracts import RecoveryResult
from osa.tasks.action_bridge import (
    AutonomousActionBridge,
    AutonomousActionResolver,
)
from osa.tasks.autonomous import AutonomousTaskResult
from osa.tasks.autonomous_recovery import (
    AutonomousRecoveryDecision,
    AutonomousRecoveryExecutor,
    CallbackAutonomousRecoveryPolicy,
)
from osa.tasks.autonomous_executor import UnifiedAutonomousExecutor


@dataclass
class FixedResolver:
    """Return one immutable action request for every recovery attempt."""

    request: ActionRequest

    def resolve(
        self,
        run_id: str,
        task_id: str,
    ) -> ActionRequest:
        return self.request


class RecordingBridge:
    """Record every attempt reaching the unified autonomous bridge."""

    def __init__(
        self,
        outputs: list[AutonomousTaskResult],
    ) -> None:
        self.outputs = list(outputs)
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

        return self.outputs.pop(0)


def _request() -> ActionRequest:
    return ActionRequest(
        kind=ActionKind.TOOL,
        name="demo",
        arguments={
            "value": 42,
        },
        request_id="autonomous-recovery-context-010",
    )


def _unified(
    request: ActionRequest,
    bridge: RecordingBridge,
) -> UnifiedAutonomousExecutor:
    resolver: AutonomousActionResolver = FixedResolver(
        request
    )

    return UnifiedAutonomousExecutor(
        resolver,
        bridge,
    )


def test_autonomous_recovery_preserves_scope_across_retries() -> None:
    request = _request()

    bridge = RecordingBridge(
        [
            AutonomousTaskResult(
                task_id="task-010",
                status="failed",
                error="temporary-1",
            ),
            AutonomousTaskResult(
                task_id="task-010",
                status="failed",
                error="temporary-2",
            ),
            AutonomousTaskResult(
                task_id="task-010",
                status="completed",
                output="42",
            ),
        ]
    )

    unified = _unified(
        request,
        bridge,
    )

    decisions: list[tuple[str, str, int]] = []

    def decide(
        run_id: str,
        task_id: str,
        result: AutonomousTaskResult,
        attempt: int,
    ) -> AutonomousRecoveryDecision:
        decisions.append(
            (
                run_id,
                task_id,
                attempt,
            )
        )

        return AutonomousRecoveryDecision(
            retry=True,
            reason="temporary",
        )

    recovery = AutonomousRecoveryExecutor(
        unified,
        policy=CallbackAutonomousRecoveryPolicy(
            decide
        ),
        max_attempts=3,
    )

    result = recovery.execute(
        "run-010",
        "task-010",
    )

    assert result.status == "completed"
    assert result.output == "42"

    assert decisions == [
        (
            "run-010",
            "task-010",
            1,
        ),
        (
            "run-010",
            "task-010",
            2,
        ),
    ]

    assert len(bridge.calls) == 3

    for run_id, task_id, forwarded in bridge.calls:
        assert run_id == "run-010"
        assert task_id == "task-010"
        assert forwarded is request
        assert forwarded.request_id == (
            "autonomous-recovery-context-010"
        )

    contexts = [
        ExecutionContext.from_autonomous_action_request(
            forwarded,
            run_id=run_id,
            task_id=task_id,
        )
        for run_id, task_id, forwarded in bridge.calls
    ]

    assert all(
        context.run_id == "run-010"
        for context in contexts
    )
    assert all(
        context.task_id == "task-010"
        for context in contexts
    )
    assert all(
        context.source == "autonomous"
        for context in contexts
    )
    assert all(
        context.round_number is None
        for context in contexts
    )


def test_autonomous_recovery_does_not_change_request_identity() -> None:
    request = _request()

    bridge = RecordingBridge(
        [
            AutonomousTaskResult(
                task_id="task-011",
                status="failed",
                error="temporary",
            ),
            AutonomousTaskResult(
                task_id="task-011",
                status="completed",
                output="recovered",
            ),
        ]
    )

    unified = _unified(
        request,
        bridge,
    )

    recovery = AutonomousRecoveryExecutor(
        unified,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda run_id, task_id, result, attempt:
                AutonomousRecoveryDecision(
                    retry=True,
                    reason="retry",
                )
        ),
        max_attempts=2,
    )

    result = recovery.execute(
        "run-011",
        "task-011",
    )

    assert result.status == "completed"
    assert result.output == "recovered"

    assert [
        forwarded.request_id
        for _, _, forwarded in bridge.calls
    ] == [
        "autonomous-recovery-context-010",
        "autonomous-recovery-context-010",
    ]

    assert [
        id(forwarded)
        for _, _, forwarded in bridge.calls
    ] == [
        id(request),
        id(request),
    ]


def test_autonomous_success_is_not_retried() -> None:
    request = _request()

    bridge = RecordingBridge(
        [
            AutonomousTaskResult(
                task_id="task-012",
                status="completed",
                output="done",
            ),
        ]
    )

    unified = _unified(
        request,
        bridge,
    )

    recovery = AutonomousRecoveryExecutor(
        unified,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda run_id, task_id, result, attempt:
                AutonomousRecoveryDecision(
                    retry=True,
                    reason="must-not-be-used",
                )
        ),
        max_attempts=3,
    )

    result = recovery.execute(
        "run-012",
        "task-012",
    )

    assert result.status == "completed"
    assert result.output == "done"
    assert len(bridge.calls) == 1


def test_autonomous_retry_preserves_canonical_context_on_final_failure() -> None:
    request = _request()

    bridge = RecordingBridge(
        [
            AutonomousTaskResult(
                task_id="task-013",
                status="failed",
                error="failure-1",
            ),
            AutonomousTaskResult(
                task_id="task-013",
                status="failed",
                error="failure-2",
            ),
        ]
    )

    unified = _unified(
        request,
        bridge,
    )

    recovery = AutonomousRecoveryExecutor(
        unified,
        policy=CallbackAutonomousRecoveryPolicy(
            lambda run_id, task_id, result, attempt:
                AutonomousRecoveryDecision(
                    retry=True,
                    reason="retry",
                )
        ),
        max_attempts=2,
    )

    result = recovery.execute(
        "run-013",
        "task-013",
    )

    assert result.status == "failed"
    assert result.error == "failure-2"

    assert len(bridge.calls) == 2

    for run_id, task_id, forwarded in bridge.calls:
        context = ExecutionContext.from_autonomous_action_request(
            forwarded,
            run_id=run_id,
            task_id=task_id,
        )

        assert context.request_id == (
            "autonomous-recovery-context-010"
        )
        assert context.run_id == "run-013"
        assert context.task_id == "task-013"
        assert context.source == "autonomous"
        assert context.round_number is None
